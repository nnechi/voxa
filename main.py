import sys 
import os
from dataset import LRS2Dataset, collate_fn, collate_fn_video
from sample import build_samples 
from AudioModel import AudioModel, VideoAudioModel 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch 
from train import train_one_epoch, val_one_epoch, test_one_epoch, train_one_epoch_video, test_one_epoch_video, val_one_epoch_video


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

def __init__(video = False): 

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


def audio_proc(): 

    #create a vocab. 

    train_loader, test_loader, val_loader = __init__(False); 

    model = AudioModel(spec_bins = 80, hidden_dim = 256, layers=2, vocab_size = len(char_to_int)); 

    #########SPECS#############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device);
    criterion = nn.CTCLoss(blank = 0, zero_infinity= True);
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3); 


 

    #############training#############

    # cur_batch = next(iter(train_loader)); 
    # print(len(cur_batch)); 

    # features, labels, input_len, target_len, transcripts, sample_ids = cur_batch; 

    # print(features.shape); 
    # print(labels.shape); 
    # print(input_len); 
    # print(target_len); 
    # print(transcripts[0]); 
    # print(sample_ids[0]); 

    # features = features.to(device);
    # labels = labels.to(device); 
    # input_len = input_len.to(device); 
    # target_len = target_len.to(device); 
    


    # logits = model(features); 
    # print(logits.shape); 

    # probabilities = logits.log_softmax(dim = -1); 
    # probabilities = probabilities.permute(1,0,2); 
    # loss = criterion(probabilities, labels, input_len, target_len); 
    # print(loss.item()); 

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
    
        
    ####TESTING#########

    model.load_state_dict(torch.load("best_audio_model.pt", map_location = device)); 
    model = model.to(device);
    test_loss = test_one_epoch(model, test_loader, criterion, device, int_to_char); 
    print(f"Test Loss: {test_loss:.4f}"); 






def audio_visual_proc(): 
    train_loader, test_loader, val_loader = __init__(True); 

    model = VideoAudioModel( 
        spec_bins = 80, 
        hidden_dim=256, 
        layers=2, 
        vocab_size = len(char_to_int), 
        video_feature_dim=256
    ); 
    # batch = next(iter(train_loader)); 
    # audio, video, labels, input_len, target_len, transcripts, sample_ids = batch; 

    # print(audio.shape); 
    # print(video.shape); 
    # print(labels.shape); 
    # print(input_len); 
    # print(target_len); 
    # print(transcripts[0]); 
    # print(sample_ids[0]); 


    # audio = audio.to(device); 
    # video = video.to(device); 

    # logits = model(audio, video); 
    # print(logits.shape); 
    # probabilities = logits.log_softmax(dim=-1); 
    # probabilities = probabilities.permute(1,0,2); 

    # loss = criterion(probabilities, labels, input_len, target_len); 
    # print(loss.item()); 


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

    model.load_state_dict(torch.load("best_av_model.pt", map_location = device)); 
    model = model.to(device); 
    
    test_loss = test_one_epoch_video(model, test_loader, criterion, device, int_to_char); 
    print(f"AV Test Loss: {test_loss:.4f}"); 


    




   










    


def test(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 

    if (torch.cuda.is_available()): 
        print("cuda")
    else: 
        print("cpu"); 

    


if __name__ == "__main__": 
    test(); 
    audio_proc(); 
