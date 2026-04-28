import sys 
import os
from dataset import LRS2Dataset, collate_fn, collate_fn_video
from sample import build_samples 
from AudioModel import AudioModel, VideoAudioModel 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch 
from train import train_one_epoch, val_one_epoch, test_one_epoch, train_one_epoch_video, test_one_epoch_video, val_one_epoch_video
from torchaudio.models.decoder import ctc_decoder


TRAIN_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Project/train"
PRETRAIN_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Project/pretrain"
VAL_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Project/val"
TEST_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Project/test"


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

def build_loaders(video = False): 

    # pretrain_samples = build_samples(PRETRAIN_PATH); 
    train_samples = build_samples(TRAIN_PATH); 
    test_samples = build_samples(TEST_PATH); 
    val_samples = build_samples(VAL_PATH); 

    # pretrain = LRS2Dataset(pretrain_samples, char_to_int); 
    train = LRS2Dataset(train_samples, char_to_int, video); 
    val = LRS2Dataset(val_samples, char_to_int, video); 
    test = LRS2Dataset(test_samples, char_to_int, video);

    # print(pretrain.__len__()); 
    print(train.__len__()); 
    print(val.__len__()); 
    print(test.__len__()); 

    if (video):  
        train_loader = DataLoader(train, batch_size =8, shuffle = True, collate_fn = collate_fn_video); 
        test_loader = DataLoader(test, batch_size=8, shuffle = False, collate_fn = collate_fn_video); 
        val_loader = DataLoader(val, batch_size = 8, shuffle = False, collate_fn = collate_fn_video); 
    
    else: 
        # pretrain_loader = DataLoader(pretrain, batch_size = 8, shuffle = True, collate_fn = collate_fn); 
        train_loader = DataLoader(train, batch_size =8, shuffle = True, collate_fn = collate_fn); 
        test_loader = DataLoader(test, batch_size=8, shuffle = False, collate_fn = collate_fn); 
        val_loader = DataLoader(val, batch_size = 8, shuffle = False, collate_fn = collate_fn); 

    return  train_loader, test_loader, val_loader; 


def train_audio(train_loader, val_loader):  

    #create a vocab. 
    model = AudioModel(spec_bins = 80, hidden_dim = 256, layers=2, vocab_size = len(char_to_int)); 
    #########SPECS#############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device);
    criterion = nn.CTCLoss(blank = 0, zero_infinity= True);
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3); 


    ######TRAINING##############

    epochs = 25; 
    patience = 4; 
    delta = 0.001; 

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
            torch.save(model.state_dict(), "best_audio_model.pt"); 
        else: 
            epochs_no_improvement += 1 

        if (epochs_no_improvement >= patience): 
            print("Early Stopping."); 
            break; 
    
        

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






def train_av(train_loader, val_loader): #
    

    model = VideoAudioModel( 
        spec_bins = 80, 
        hidden_dim=256, 
        layers=2, 
        vocab_size = len(char_to_int), 
        video_feature_dim=256
    ); 

    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device);
    criterion = nn.CTCLoss(blank = 0, zero_infinity= True);
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3); 


    epochs = 25 
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
            torch.save(model.state_dict(), "best_av_model.pt"); 
        else: 
            epochs_no_improvement += 1; 


        if epochs_no_improvement >= patience: 
            break; 


    
def eval_av(test_loader, output_path, decoder = None): 
    model = VideoAudioModel(80, 256, 2, len(char_to_int), video_feature_dim=256); 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device); 
    criterion = nn.CTCLoss(blank = 0, zero_infinity = True); 
    model.load_state_dict(torch.load("best_av_model.pt", map_location = device)); 

    test_loss, wer, cer = test_one_epoch_video(model, test_loader, criterion, device, int_to_char, output_path,  decoder = decoder); 

    with open(output_path, "a", encoding = "utf-8") as f: 
        f.write("\n")
        f.write(f"Test Loss: {test_loss:.4f}\n"); 
        f.write(f"Test WER: {wer:.4f}\n"); 
        f.write(f"Test CER: {cer:.4f}\n");
    

    print(f"Test Loss: {test_loss:.4f}");
    print(f"Test WER: {wer:.4f}");
    print(f"Test CER: {cer:.4f}"); 

       


    # #Decoder Output vs. Non-Decoder Output
    # if beam_decoder is not None: 
    #     test_loss, wer, cer = test_one_epoch_video(model, test_loader, criterion, device, int_to_char, decoder=beam_decoder); 
    #     print(f"Test Loss: {test_loss:.4f}"); 
    #     print(f"Test WER: {wer:.4f}");
    #     print(f"Test CER: {cer:.4f}");
    # else: 
    #     test_loss,wer, cer = test_one_epoch_video(model, test_loader, criterion, device, int_to_char); 
    #     print(f"Test Loss: {test_loss:.4f}"); 
    #     print(f"Test WER: {wer:.4f}");
    #     print(f"Test CER: {cer:.4f}");


    




   










    


def test(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 

    if (torch.cuda.is_available()): 
        print("cuda")
    else: 
        print("cpu"); 

    


if __name__ == "__main__": 
    train_loader, test_loader, val_loader = build_loaders(False);
    train_loader_V,test_loader_V,val_loader_V = build_loaders(True); 

    no_lm_decoder = ctc_decoder(
        lexicon = None, 
        tokens = [int_to_char[i] for i in range (len(int_to_char))], 
        lm = None, 
        nbest = 1, 
        beam_size = 25, 
        blank_token = "<blank>", sil_token=" "
    )

    lm_decoder = ctc_decoder( 
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

    lm_decoder_w_vocab = ctc_decoder(
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

    print("Training Audio Model:")

    train_audio(train_loader, val_loader); 

    print("Evaluating Audio Model + Greedy Decode (Baseline)");
    eval_audio(test_loader, "audio_model_greedy.txt"); 

    print("Evaluating Audio Model + Beam Decoder");
    eval_audio(test_loader, "audio_model_beam.txt",no_lm_decoder); 

    print("Evaluating Audio Model + KenLM"); 
    eval_audio(test_loader, "audio_model_kenlm_nolexicon.txt", lm_decoder); 

    print("Evaluating Audio Model + KenLM+Lexicon");
    eval_audio(test_loader, "audio_model_kenlm_lexicon.txt", lm_decoder_w_vocab); 

    print("Training Audio-Visual Model"); 
    train_av(train_loader_V, val_loader_V); 

    print("Evaluating AV Model + Greedy Decode (Baseline)"); 
    eval_av(test_loader_V, "av_model_greedy.txt"); 

    print("Evaluating AV Model + Beam Decode"); 
    eval_av(test_loader_V, "av_model_beam.txt", no_lm_decoder); 

    print("Evaluating AV Model + KenLM");
    eval_av(test_loader_V, "av_model_kenlm_nolexicon.txt", lm_decoder); 

    print("Evaluating AV Model + KenLM+Lexicon");
    eval_av(test_loader_V, "av_model_kenlm_lexicon.txt", lm_decoder_w_vocab); 


