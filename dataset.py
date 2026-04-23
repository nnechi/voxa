""" Load the dataset and turn raw files into the training material. """
import torch 
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import cv2 
import numpy as np
class LRS2Dataset(Dataset):


    """samples will be the mp4 and txt transcript matches."""
    """pair class"""
    def __init__(self, samples, char_to_int, video = False):
        self.samples = samples; # sample objects.  
        self.char_to_int = char_to_int; #encoding 
        self.vocab_size = len(self.char_to_int); #number of encodings. 
        
        self.video = video; 
        self.target_sample_rate = 16000; 
        self.num_video_frames = 32; 
        self.frame_height = 96; 
        self.frame_width = 96; 
    
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

        if (self.video): 
            video_frames = self.load_video_frames(sample.mp4); 


        with open(sample.txt, "r", encoding = "utf-8") as f: 
            transcript = self.normalize(f.read()); 
        
        encoded_label = self.encode(transcript, self.char_to_int); #switch transcript to label form. 
        labels = torch.tensor(encoded_label, dtype = torch.long); 


        if (self.video): 
            return features, video_frames, labels, transcript, sample.id; 
        else: 
            return features, labels, transcript, sample.id; 

    def load_video_frames(self, mp4_path: str) -> torch.Tensor: 
        captured_frames = cv2.VideoCapture(mp4_path); 
        total_frames = int(captured_frames.get(cv2.CAP_PROP_FRAME_COUNT)); 
        num_frames = self.num_video_frames; 
        
        if (total_frames <= 0): 
            captured_frames.release() 
            return torch.zeros(num_frames, 3, self.frame_height, self.frame_width); 

        frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int); 
        target_set = set(frame_indices.tolist()); 


        frames = []; 
        current_index = 0; 

        

        while True: 
            ret, frame = captured_frames.read() 

            if not ret: 
                break; 

            if current_index in target_set: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); 
                frame = cv2.resize(frame,(self.frame_width, self.frame_height)); 
                frame = frame.astype(np.float32)/255.0; 
                frame = torch.from_numpy(frame).permute(2,0,1); 
                frames.append(frame); 


            current_index+=1; 

        captured_frames.release(); 

        while (len(frames) < num_frames): 
            frames.append(torch.zeros(3, self.frame_height, self.frame_width)); 
        
        frames = frames[:num_frames]; 
        
        return torch.stack(frames); 










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



def collate_fn_video(batch): 
    audio_features, video_frames, labels, transcripts, sample_ids = zip(*batch); 

    input_len = torch.tensor([feature.size(1) for feature in audio_features], dtype = torch.long); 
    target_len = torch.tensor([label.size(0) for label in labels], dtype = torch.long); 

    audio_features = [feature.squeeze(0) for feature in audio_features]; 
    padded_audio = pad_sequence(audio_features, batch_first = True); 
    padded_audio = padded_audio.unsqueeze(1); 

    video_frames = torch.stack(video_frames); 
    flat_labels = torch.cat(labels); 

    return padded_audio, video_frames, flat_labels, input_len, target_len, transcripts, sample_ids; 




