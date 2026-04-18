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

    pretrain = LRS2Dataset(pretrain_samples, char_to_int); 
    train = LRS2Dataset(train_samples, char_to_int); 
    test = LRS2Dataset(test_samples, char_to_int); 

    pretrain_loader = DataLoader(pretrain, batch_size = 8, shuffle = True, collate_fn = collate_fn); 
    train_loader = DataLoader(train, batch_size =8, shuffle = True, collate_fn = collate_fn); 
    test_loader = DataLoader(test, batch_size=8, shuffle = False, collate_fn = collate_fn); 


    model = AudioModel(spec_bins = 80, hidden_dim = 256, layers=2, vocab_size = len(char_to_int)); 

    #########SPECS#############

    criterion = nn.CTCLoss(blank = 0, zero_infinity= True);
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3); 


    print(pretrain.__len__()); 
    print(train.__len__()); 
    print(test.__len__()); 






    





    

    
    











   









if __name__ == "__main__": 
    main(); 
