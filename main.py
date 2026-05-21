import sys 
import os
from dataset import LRS2Dataset, collate_fn, collate_fn_video
from sample import build_samples 
from AudioModel import AudioModel, VideoAudioModel, BaselineModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch 
from train import train_one_epoch, val_one_epoch, test_one_epoch, train_one_epoch_video, test_one_epoch_video, val_one_epoch_video
from torchaudio.models.decoder import ctc_decoder


TRAIN_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Data/train"
PRETRAIN_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Data/pretrain"
VAL_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Data/val"
TEST_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Data/test"


"""@function 
    dataloader initialization. 
    in : boolean for video dataset 
    out :  train_loader, val_loader, test_loader
"""
def create_vocab(vocab : str): 
    char_to_int = {}; 
    int_to_char = {};   
    char_to_int["<blank>"] = 0; 
    int_to_char[0] = "<blank>"; 

    i = 1; 
    for c in vocab:     
        char_to_int[c] = i; 
        int_to_char[i] = c; 
        i += 1; 

    return char_to_int, int_to_char; 


char_to_int, int_to_char = create_vocab("abcdefghijklmnopqrstuvwxyz0123456789 '"); 


"""@function 
    dataloader initialization. 
    in : boolean for video dataset 
    out :  train_loader, val_loader, test_loader
"""
def build_loaders(video = False): 

    # pretrain_samples = build_samples(PRETRAIN_PATH); 
    train_samples = build_samples(TRAIN_PATH); 
    test_samples = build_samples(TEST_PATH); 
    val_samples = build_samples(VAL_PATH); 

    # pretrain = LRS2Dataset(pretrain_samples, char_to_int,video); 
    train = LRS2Dataset(train_samples, char_to_int, video); 
    val = LRS2Dataset(val_samples, char_to_int, video); 
    test = LRS2Dataset(test_samples, char_to_int, video);

    # print(pretrain.__len__()); 
    print(train.__len__()); 
    print(val.__len__()); 
    print(test.__len__()); 

    if (video):  
        # pretrain_loader = DataLoader(pretrain, batch_size = 8, shuffle = True, collate_fn = collate_fn_video); 
        train_loader = DataLoader(train, batch_size =8, shuffle = True, collate_fn = collate_fn_video); 
        test_loader = DataLoader(test, batch_size=8, shuffle = False, collate_fn = collate_fn_video); 
        val_loader = DataLoader(val, batch_size = 8, shuffle = False, collate_fn = collate_fn_video); 
    
    else: 
        # pretrain_loader = DataLoader(pretrain, batch_size = 8, shuffle = True, collate_fn = collate_fn); 
        train_loader = DataLoader(train, batch_size =8, shuffle = True, collate_fn = collate_fn); 
        test_loader = DataLoader(test, batch_size=8, shuffle = False, collate_fn = collate_fn); 
        val_loader = DataLoader(val, batch_size = 8, shuffle = False, collate_fn = collate_fn); 

    return train_loader, val_loader,  test_loader; 




"""@function 
    Train process for CNN
    in : boolean for video dataset 
    out :  train_loader, val_loader, test_loader
"""

def train_baseline(train_loader, val_loader, lr = 1e-3, epochs = 25, save_path = "baseline.pt"): 
    model = BaselineModel(spec_bins=80, vocab_size=len(char_to_int)); 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); 
    criterion = nn.CTCLoss(blank = 0, zero_infinity = True); 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr); 

    patience = 3; 
    delta = 0.001 

    best_loss = float("inf"); 
    epochs_no_improvement = 0; 

    for epoch in range(epochs): 
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device); 
        val_loss = val_one_epoch(model, val_loader, criterion, device); 

        print(f"Baseline Epoch: {epoch + 1}/ {epochs}"); 
        print(f"Train Loss: {train_loss:.4f}"); 
        print(f"Val Loss: {val_loss:.4f}"); 

        if (best_loss - val_loss) > delta: 
            best_loss = val_loss 
            epochs_no_improvement = 0; 
            torch.save(model.state_dict(), save_path); 
        else: 
            epochs_no_improvement += 1; 

        if epochs_no_improvement >= patience: 
            break; 


def eval_baseline(test_loader, output_path, decoder = None): 
    model = BaselineModel(spec_bins=80, vocab_size = len(char_to_int)); 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); 
    criterion = nn.CTCLoss(blank = 0, zero_infinity = True); 
    model.load_state_dict(torch.load("baseline.pt")); 
    model.eval(); 

    test_loss, wer, cer = test_one_epoch(model, test_loader, criterion, device, int_to_char, output_path, decoder); 

    with open(output_path, "a", encoding = "utf-8") as f: 
        f.write("\n"); 
        f.write(f"Test Loss: {test_loss:.4f}\n"); 
        f.write(f"Test WER: {wer:.4f}\n"); 
        f.write(f"Test CER: {cer:.4f}\n"); 


    print(f"Test Loss: {test_loss:.4f}"); 
    print(f"Test WER: {wer:.4f}")
    print(f"Test CERL {cer:.4f}")


"""@function 
    Train process for AudioModel
    in : train dataset, validation dataset, learning rate, epoch counter, pretrainer path (opt), filepath for otuput
    out :  best_audio_model.pt
"""
def train_audio(train_loader, val_loader, lr, epochs = 25, pretrain_path = None, save_path = "best_audio_model.pt"):  
    model = AudioModel(spec_bins = 80, hidden_dim = 256, layers=2, vocab_size = len(char_to_int)); 
    #########SPECS#############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device);

    if pretrain_path is not None: 
        model.load_state_dict(torch.load(pretrain_path, map_location = device)); 
        print("Loaded weights."); 
    
    criterion = nn.CTCLoss(blank = 0, zero_infinity= True);
    optimizer = torch.optim.Adam(model.parameters(), lr = lr); 


    ######TRAINING##############

    epochs = epochs; 
    patience = 4; 
    delta = 0.001; # min threshold for earlystop

    best_loss = float("inf"); 
    epochs_no_improvement = 0; 
    for epoch in range(epochs): 
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device); 
        val_loss = val_one_epoch(model, val_loader, criterion, device); 

        print(f"Epoch: {epoch + 1}/{epochs}")
        print(f" train loss: {train_loss:.4f}"); 
        print(f" val loss: {val_loss:.4f}"); 

        if (best_loss - val_loss) > delta: 
            best_loss = val_loss 
            epochs_no_improvement = 0; 
            torch.save(model.state_dict(), save_path); 
        else: 
            epochs_no_improvement += 1 

        if (epochs_no_improvement >= patience): 
            print("Early Stopping."); 
            break; 
    
        
"""@function 
    Evaluation Process for AudioModel
    in : Test dataset, output path, and decoder test
    out :  CTC loss, wer, cer
"""
def eval_audio(test_loader, output_path, decoder = None): 
    model = AudioModel(80, 256, 2, vocab_size = len(char_to_int)); 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device); 
    criterion = nn.CTCLoss(blank = 0, zero_infinity =  True); 
    model.load_state_dict(torch.load("best_audio_model.pt", map_location = device)); 
    model.eval(); 

    test_loss, wer, cer = test_one_epoch(model, test_loader, criterion, device, int_to_char, output_path, decoder = decoder); 

    with open(output_path, "a", encoding = "utf-8") as f: 
        f.write("\n")
        f.write(f"Test Loss: {test_loss:.4f}\n"); 
        f.write(f"Test WER: {wer:.4f}\n"); 
        f.write(f"Test CER: {cer:.4f}\n");
    
    print(f"Test Loss: {test_loss:.4f}");
    print(f"Test WER: {wer:.4f}");
    print(f"Test CER: {cer:.4f}"); 




"""@function 
    Training process for audio_visual Model
    in : Train dataset, val dataset, lr, epochs, pretrain, save
    out :  best_av_model.pt
"""
def train_av(train_loader, val_loader, lr, epochs = 25, pretrain_path = None, save_path = "best_av_model.pt"): #
    

    model = VideoAudioModel( 
        spec_bins = 80, 
        hidden_dim=256, 
        layers=2, 
        vocab_size = len(char_to_int), 
        video_feature_dim=256
    ); 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device);

    if pretrain_path is not None: 
        model.load_state_dict(torch.load(pretrain_path, map_location=device)); 
        print("Loaded weights."); 
    
    criterion = nn.CTCLoss(blank = 0, zero_infinity= True);
    optimizer = torch.optim.Adam(model.parameters(), lr = lr); 


    epochs = epochs 
    patience = 4 
    delta = 0.001; 

    best_loss = float("inf"); 
    epochs_no_improvement = 0; 


    for epoch in range(epochs): 
        train_loss = train_one_epoch_video(model, train_loader, optimizer, criterion, device); 
        val_loss = val_one_epoch_video(model, val_loader, criterion, device); 
        print(f"AV Epoch: {epoch + 1}/{epochs}"); 
        print(f" train loss: {train_loss:.4f}"); 
        print(f" val loss: {val_loss:.4f}"); 


        if (best_loss - val_loss) > delta: 
            best_loss = val_loss; 
            epochs_no_improvement = 0; 
            torch.save(model.state_dict(), save_path); 
        else: 
            epochs_no_improvement += 1; 


        if epochs_no_improvement >= patience: 
            break; 


"""@function 
    Training process for audio_visual Model
    in : Train dataset, val dataset, lr, 
    out :  CTC loss, wer, cer
"""
    
def eval_av(test_loader, output_path, decoder = None): 
    model = VideoAudioModel(80, 256, 2, len(char_to_int), video_feature_dim=256); 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device); 
    criterion = nn.CTCLoss(blank = 0, zero_infinity = True); 
    model.load_state_dict(torch.load("best_av_model.pt", map_location = device)); 
    model.eval();

    test_loss, wer, cer = test_one_epoch_video(model, test_loader, criterion, device, int_to_char, output_path,  decoder = decoder); 

    with open(output_path, "a", encoding = "utf-8") as f: 
        f.write("\n")
        f.write(f"Test Loss: {test_loss:.4f}\n"); 
        f.write(f"Test WER: {wer:.4f}\n"); 
        f.write(f"Test CER: {cer:.4f}\n");
    

    print(f"Test Loss: {test_loss:.4f}");
    print(f"Test WER: {wer:.4f}");
    print(f"Test CER: {cer:.4f}"); 

       

    
"""@function 
    ONLY FOR quick evaluation. use for quickly loading test samples and making them available in main. 
    in : n/a
    out :  test_loader, test_loader_visual. 
"""


def build_test_loaders_only(): 
    test_samples = build_samples(TEST_PATH); 
    test = LRS2Dataset(test_samples, char_to_int, False);
    test_v = LRS2Dataset(test_samples, char_to_int, True);
    test_loader_V = DataLoader(test_v, batch_size=8, shuffle = False, collate_fn = collate_fn_video); 
    test_loader = DataLoader(test, batch_size=8, shuffle = False, collate_fn = collate_fn);

    return test_loader, test_loader_V



   





"""@function 
    Helper function for calling quick test cases after training. 
    in : file_path, video bool, opt decoder
    out :  file_path.txt
"""
    
def quick_test(loader,file_path, video = False, decoder = None): 
    if video: 
        eval_av(loader, file_path, decoder); 
    else: 
        eval_audio(loader, file_path, decoder); 
        

    


if __name__ == "__main__": 
    # train_loader,val_loader, test_loader = build_loaders(False);
    # print("===================CNN AUDIO BASELINE================="); 
    #  train_baseline(train_loader, val_loader);
    # train_loader_V,val_loader_V, test_loader_V = build_loaders(True);
    # print("Training Audio Model:")
    # train_audio(train_loader, val_loader, 1e-3, epochs = 25); 
       #=========================================
    # print("Training Audio-Visual Model"); 
    # train_av(train_loader_V, val_loader_V, 1e-3, epochs = 25); 
    #=========================================

   
    test_loader, test_loader_V = build_test_loaders_only(); # ALREADY HAVE CHECKPOINTS TO RUN THESE SEGMENTS! 

    print("Evaluating Audio Model CNN-Only Baseline + Greedy Decode")
    eval_baseline(test_loader, "baseline.txt", None); 


    print("Evaluating Audio Model + Greedy Decode (Baseline)");
    quick_test(test_loader, "audio_model_greedy.txt", False, None); 


    ###########################CREATE DECODERS HERE #######################################
    no_lm_decoder_a = ctc_decoder(
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = None, 
        nbest = 1, 
        beam_size = 25, 
        blank_token = "<blank>", sil_token=" "
    )

    no_lm_decoder_b = ctc_decoder(
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = None, 
        nbest = 1, 
        beam_size = 10, 
        blank_token = "<blank>", sil_token=" "
    )

    no_lm_decoder_c = ctc_decoder(
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = None, 
        nbest = 1, 
        beam_size = 5, 
        blank_token = "<blank>", sil_token=" "
    )

    lm_decoder_a = ctc_decoder( 
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 0.5, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )
    lm_decoder_b= ctc_decoder( 
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 0.75, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )
    lm_decoder_c = ctc_decoder( 
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 1.0, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )

    lm_decoder_d = ctc_decoder( 
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 1.5, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )

    lm_decoder_e = ctc_decoder( 
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 2.0, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )

    lm_decoder_w_vocab_a = ctc_decoder(
        lexicon = "lexicon.txt", 
        tokens = [int_to_char[i] for i in range(len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 1.0, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )

    lm_decoder_w_vocab_b = ctc_decoder(
        lexicon = "lexicon.txt", 
        tokens = [int_to_char[i] for i in range(len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 1.5, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )

    lm_decoder_w_vocab_c = ctc_decoder(
        lexicon = "lexicon.txt", 
        tokens = [int_to_char[i] for i in range(len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 2.0, 
        word_score = 0.0, 
        blank_token = "<blank>", 
        sil_token = " "
    )
        
    lm_decoder_w_vocab_d = ctc_decoder(
        lexicon = "lexicon.txt", 
        tokens = [int_to_char[i] for i in range(len(int_to_char))], 
        lm = "kenlm.bin", 
        nbest = 1, 
        beam_size = 25, 
        lm_weight = 2.0, 
        word_score = 0.5, 
        blank_token = "<blank>", 
        sil_token = " "
    )


    ############################END CREATING DECODERS ##############################


    #######################################TESTING ALL DIFFERENT SCENARIOS FOR EACH CONFIGURATION#######################################


    print("Evaluating Audio Model + Beam Decoder (A)");
    quick_test(test_loader,"audio_model_beam_a.txt", False, no_lm_decoder_a);

    print("Evaluating Audio Model + Beam Decoder (B)");
    quick_test(test_loader,"audio_model_beam_b.txt", False, no_lm_decoder_b); 


    print("Evaluating Audio Model + Beam Decoder (C)");
    quick_test(test_loader,"audio_model_beam_c.txt", False, no_lm_decoder_c); 

    



    print("Evaluating Audio Model + KenLM(A)"); 
    quick_test(test_loader, "audio_model_kenlm_nolexicon_A.txt", False, lm_decoder_a);

    print("Evaluating Audio Model + KenLM(B)"); 
    quick_test(test_loader,"audio_model_kenlm_nolexicon_B.txt", False, lm_decoder_b); 
    
    print("Evaluating Audio Model + KenLM(C)"); 
    quick_test(test_loader,"audio_model_kenlm_nolexicon_C.txt", False, lm_decoder_c);

    print("Evaluating Audio Model + KenLM(D)"); 
    quick_test(test_loader,"audio_model_kenlm_nolexicon_D.txt", False, lm_decoder_d); 

    print("Evaluating Audio Model + KenLM(E)"); 
    quick_test(test_loader,"audio_model_kenlm_nolexicon_E.txt", False, lm_decoder_e); 


    
    print("Evaluating Audio Model + KenLM+Lexicon (A)");
    quick_test(test_loader,"audio_model_kenlm_lexicon_A.txt", False, lm_decoder_w_vocab_a); 

    print("Evaluating Audio Model + KenLM+Lexicon (B)");
    quick_test(test_loader,"audio_model_kenlm_lexicon_B.txt", False, lm_decoder_w_vocab_b); 

    print("Evaluating Audio Model + KenLM+Lexicon (C)");
    quick_test(test_loader,"audio_model_kenlm_lexicon_C.txt", False, lm_decoder_w_vocab_c); 

    print("Evaluating Audio Model + KenLM+Lexicon (D)");
    quick_test(test_loader,"audio_model_kenlm_lexicon_D.txt", False, lm_decoder_w_vocab_d); 


 

    print("Evaluating AV Model + Greedy Decode (Baseline)"); 
    quick_test(test_loader_V,"av_model_greedy.txt", True, None);  

    print("Evaluating AV Model + Beam Decoder (A)");
    quick_test(test_loader_V,"av_model_beam_a.txt", True, no_lm_decoder_a);

    print("Evaluating AV Model + Beam Decoder (B)");
    quick_test(test_loader_V,"av_model_beam_b.txt", True, no_lm_decoder_b); 


    print("Evaluating AV Model + Beam Decoder (C)");
    quick_test(test_loader_V,"av_model_beam_c.txt", True, no_lm_decoder_c); 

    print("Evaluating AV Model + KenLM(A)"); 
    quick_test(test_loader_V,"av_model_kenlm_nolexicon_A.txt", True, lm_decoder_a);

    print("Evaluating AV Model + KenLM(B)"); 
    quick_test(test_loader_V,"av_model_kenlm_nolexicon_B.txt", True, lm_decoder_b); 
    
    print("Evaluating AV Model + KenLM(C)"); 
    quick_test(test_loader_V,"av_model_kenlm_nolexicon_C.txt", True, lm_decoder_c);

    print("Evaluating Audio Model + KenLM(D)"); 
    quick_test(test_loader_V,"av_model_kenlm_nolexicon_D.txt", True, lm_decoder_d); 

    print("Evaluating AV Model + KenLM(E)"); 
    quick_test(test_loader_V,"av_model_kenlm_nolexicon_E.txt", True, lm_decoder_e); 


    print("Evaluating AV Model + KenLM+Lexicon (A)");
    quick_test(test_loader_V,"av_model_kenlm_lexicon_A.txt", True, lm_decoder_w_vocab_a); 

    print("Evaluating AV Model + KenLM+Lexicon (B)");
    quick_test(test_loader_V,"av_model_kenlm_lexicon_B.txt", True, lm_decoder_w_vocab_b); 

    print("Evaluating AV Model + KenLM+Lexicon (C)");
    quick_test(test_loader_V,"av_model_kenlm_lexicon_C.txt", True, lm_decoder_w_vocab_c); 

    print("Evaluating AV Model + KenLM+Lexicon (D)");
    quick_test(test_loader_V,"av_model_kenlm_lexicon_D.txt", True, lm_decoder_w_vocab_d); 



