""" Load the dataset and turn raw files into the training material. """
from torch.utils.data import Dataset





class LRS2Dataset(Dataset):


    """samples will be the mp4 and txt transcript matches."""
    """pair class"""
    def __init__(self, samples):
        self.samples = samples; # sample objects. 
    
    def __len__(self): 
        return len(self.samples); 
    

    def __getitem__(self,idx): 
        return self.samples[idx]; 
        
