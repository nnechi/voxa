""" Load the dataset and turn raw files into the training material. """
import torch 
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
class LRS2Dataset(Dataset):


    """samples will be the mp4 and txt transcript matches."""
    """pair class"""
    def __init__(self, samples, char_to_int):
        self.samples = samples; # sample objects.  
        self.char_to_int = char_to_int; #encoding 
        self.vocab_size = len(self.char_to_int); #number of encodings. 
    
        self.target_sample_rate = 16000; 
    
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.target_sample_rate, 
            n_mels = 80
        ); 

    
    def __len__(self): 
        return len(self.samples); 

    def get_vocab_size(self): 
        return self.vocab_size; 

    def normalize(self, text : str) -> str: 
        text = text.lower().strip(); 

        if text.startswith("text:"): 
            text = text[len("text:"):].strip(); 
        
        if "conf:" in text: 
            text = text.split("conf:")[0].strip(); 
        
        text = " ".join(text.split()); 
        return text; 

    def encode(self, text : str, char_to_idx: dict[str,int]) -> list[int]: 
        text = self.normalize(text); 
        encoded = []; 
        for c in text: 
            if c in char_to_idx: 
                encoded.append(char_to_idx[c]);
        
        return encoded; 

    def __getitem__(self,idx): 
        sample = self.samples[idx]; #filepath. 
        audio_file, sample_rate = torchaudio.load(sample.wav); 

        if (audio_file.size(0) > 1): 
            audio_file = audio_file.mean(dim=0, keepdim=True); 
        
        if (sample_rate != self.target_sample_rate): 
            resample = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate); 
            audio_file = resample(audio_file); 
        
        features = self.mel(audio_file); 
        features = features.transpose(1,2); 

        with open(sample.txt, "r", encoding = "utf-8") as f: 
            transcript = self.normalize(" ".join(f.read().strip().lower().split())); 
        
        encoded_label = self.encode(transcript, self.char_to_int); #switch transcript to label form. 
        labels = torch.tensor(encoded_label, dtype = torch.long); 

        return features, labels, transcript, sample.id; 





#create the batches. need to get every single video to the same length. 

def collate_fn(batch):
    features, labels, transcripts, sample_ids = zip(*batch); 
    
    input_len = torch.tensor([feature.size(1) for feature in features], dtype = torch.long); 
    target_len = torch.tensor([label.size(0) for label in labels], dtype = torch.long); 

    features = [feature.squeeze(0) for feature in features];

    padded_features = pad_sequence(features, batch_first=True);

    padded_features = padded_features.unsqueeze(1); 

    flat_labels = torch.cat(labels); 

    return padded_features, flat_labels, input_len, target_len, transcripts, sample_ids; 





