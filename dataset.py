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
        sample = self.samples[idx]; 

        with open(sample.txt, "r", encoding="utf-8") as f: 
            transcript = f.read().strip().lower(); 

        return sample.mp4, transcript; 

        
