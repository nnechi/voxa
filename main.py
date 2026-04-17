import sys 
import os
from dataset import LRS2Dataset
from sample import build_samples 
from AudioModel import AudioModel, VideoAudioModel 


TRAIN_PATH = "/home/nnechi/Code/Python/489/Project/train"
PRETRAIN_PATH = "/home/nnechi/Code/Python/489/Project/pretrain"
TEST_PATH = "/home/nnechi/Code/Python/489/Project/test"



def normalize(text : str) -> str: 
    text = text.lower().strip(); 
    text = " ".join(text.split()); 
    return text; 

def encode(text : str, char_to_idx: dict[str,int]) -> list[int]: 
    text = normalize(text); 
    encoded = []; 

    for c in text: 
        encoded.append(char_to_idx[c]); 
    return encoded; 


#greedy ctc decode (remove dups.)
def decode(encoded : list[int], idx_to_char: dict[int, str]) -> str: 
    decoded = []; 
    prev = "";

    for c in encoded: 
        if (prev != c and c != 0): 
            decoded.append(idx_to_char[c]); 
        prev = c; 

        

    return "".join(decoded); 



def main(): 

    pretrain_samples = build_samples(PRETRAIN_PATH); 
    train_samples = build_samples(TRAIN_PATH); 
    test_samples = build_samples(TEST_PATH); 

    pretrain = LRS2Dataset(pretrain_samples); 
    train = LRS2Dataset(train_samples); 
    test = LRS2Dataset(test_samples); 


    # print(pretrain.__len__()); 
    # print(train.__len__()); 
    # print(test.__len__()); 

    #create a vocab. 

    chars = "abcdefghijklmnopqrstuvwxyz0123456789 '"; 
     
    char_to_int = {}; 
    int_to_char = {}; 

    char_to_int["<blank>"] = 0; 
    int_to_char[0] = "<blank>"; 
    #generate char_to_int, int_to_char mappings. 

    i = 1; 
    for c in chars: 
        char_to_int[c] = i; 
        int_to_char[i] = c; 
        i+=1; 


    vocab_size = len(char_to_int);
    

    
    











   









if __name__ == "__main__": 
    main(); 
