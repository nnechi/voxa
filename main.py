import sys 
import os
from dataset import LRS2Dataset, collate_fn
from sample import build_samples 
from AudioModel import AudioModel, VideoAudioModel 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch 


TRAIN_PATH = "/home/nnechi/Code/Python/489/Project/train"
PRETRAIN_PATH = "/home/nnechi/Code/Python/489/Project/pretrain"
VAL_PATH = "/home/nnechi/Code/Python/489/Project/val"
TEST_PATH = "/home/nnechi/Code/Python/489/Project/test"


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
         
         
    
def main(): 

    #create a vocab. 
    char_to_int, int_to_char = create_vocab("abcdefghijklmnopqrstuvwxyz0123456789 '"); 

    pretrain_samples = build_samples(PRETRAIN_PATH); 
    train_samples = build_samples(TRAIN_PATH); 
    test_samples = build_samples(TEST_PATH); 
    val_samples = build_samples(VAL_PATH); 

    pretrain = LRS2Dataset(pretrain_samples, char_to_int); 
    train = LRS2Dataset(train_samples, char_to_int); 
    val = LRS2Dataset(val_samples, char_to_int); 
    test = LRS2Dataset(test_samples, char_to_int); 

    pretrain_loader = DataLoader(pretrain, batch_size = 8, shuffle = True, collate_fn = collate_fn); 
    train_loader = DataLoader(train, batch_size =8, shuffle = True, collate_fn = collate_fn); 
    test_loader = DataLoader(test, batch_size=8, shuffle = False, collate_fn = collate_fn); 
    val_loader = DataLoader(val, batch_size = 8, shuffle = False, collate_fn = collate_fn); 


    model = AudioModel(spec_bins = 80, hidden_dim = 256, layers=2, vocab_size = len(char_to_int)); 

    #########SPECS#############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 
    model = model.to(device);
    criterion = nn.CTCLoss(blank = 0, zero_infinity= True);
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3); 


    print(pretrain.__len__()); 
    print(train.__len__()); 
    print(val.__len__()); 
    print(test.__len__()); 

    #############training#############

    cur_batch = next(iter(train_loader)); 
    print(len(cur_batch)); 

    features, labels, input_len, target_len, transcripts, sample_ids = cur_batch; 

    print(features.shape); 
    print(labels.shape); 
    print(input_len); 
    print(target_len); 
    print(transcripts[0]); 
    print(sample_ids[0]); 

    features = features.to(device);
    labels = labels.to(device); 
    input_len = input_len.to(device); 
    target_len = target_len.to(device); 
    


    logits = model(features); 
    print(logits.shape); 

    probabilities = logits.log_softmax(dim = -1); 
    probabilities = probabilities.permute(1,0,2); 
    loss = criterion(probabilities, labels, input_len, target_len); 
    print(loss.item()); 

    





    


if __name__ == "__main__": 
    main(); 
