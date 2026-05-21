""" Load the dataset and turn raw files into the training material. """
import torch 
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import cv2 
import numpy as np

"""@function  
        extracts transcripts, removes unnecessary characters and metadata, then returns all lowercase version of transcript for ease of matching 
        in : text - raw transcript label text 
        out : text - normalized consistent text 
    """
def normalize(text : str) -> str: 
    text = text.lower().strip(); 

    if text.startswith("text:"): 
        text = text[len("text:"):].strip(); 
    
    if "conf:" in text: 
        text = text.split("conf:")[0].strip(); 
    
    text = " ".join(text.split()); 
    return text; 



"""@class 
    Wrapper Class for Dataset
    Methods: 
        __len__()
        get_vocab_size()
        normalize()
        encode() 
        __getitem__() 
        load_video_frames() 

"""
class LRS2Dataset(Dataset):
    
    """@constructor 
    
        samples - array of Sample.py objects 
        char_to_int - encoder dictionary 
        video - boolean flag for processing video frames. 

    """
    def __init__(self, samples, char_to_int, video = False):
        self.samples = samples; # sample objects.  
        self.char_to_int = char_to_int; #encoding 
        self.vocab_size = len(self.char_to_int); #number of encodings. 
        
        self.video = video; 
        self.target_sample_rate = 16000; 
        self.num_video_frames = 64; #grab around 6 seconds. 
        self.frame_height = 96; 
        self.frame_width = 96; 
    
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.target_sample_rate, 
            n_mels = 80
        ); 

    """@function  
        superclass override, return length of samples to be processed. 
    """
    def __len__(self): 
        return len(self.samples); 

    """@function  
        helper for number of mappings in vocab.
    """
    def get_vocab_size(self): 
        return self.vocab_size; 




    """@function  
        Encoder function that takes a transcript label and converts to a list of integers 
        in : text, char_to_idx encoder
        out : list of integer labels 
    """
    def encode(self, text : str, char_to_idx: dict[str,int]) -> list[int]: 
        text = normalize(text); 
        encoded = []; 
        for c in text: 
            if c in char_to_idx: 
                encoded.append(char_to_idx[c]);
        
        return encoded; 


    """@function 
        Function that unpacks and organizes samples for processing. 
        in : index of samples array. 
        out : processed features, video_frames, labels, transcript, sample id. 
    """
    def __getitem__(self,idx): 
        sample = self.samples[idx]; #filepath. 
        audio_file, sample_rate = torchaudio.load(sample.wav); 

        if (audio_file.size(0) > 1): 
            audio_file = audio_file.mean(dim=0, keepdim=True); 
        
        if (sample_rate != self.target_sample_rate): 
            resample = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate); 
            audio_file = resample(audio_file); 
        
        features = self.mel(audio_file);  #process audio using a spectrogram
        features = features.transpose(1,2); 

        if (self.video): 
            video_frames = self.load_video_frames(sample.mp4); 


        with open(sample.txt, "r", encoding = "utf-8") as f: 
            transcript = normalize(f.read()); 
        
        encoded_label = self.encode(transcript, self.char_to_int); #switch transcript to label form. 
        labels = torch.tensor(encoded_label, dtype = torch.long); 


        if (self.video): 
            return features, video_frames, labels, transcript, sample.id; 
        else: 
            return features, labels, transcript, sample.id; 

    
    """@function 
        Loads video frames into a tensor format and process it. 
        in : path of the mp4. 
        out : tensor representing all frames in the mp4. 
    """

    def load_video_frames(self, mp4_path: str) -> torch.Tensor: 
        captured_frames = cv2.VideoCapture(mp4_path); 
        total_frames = int(captured_frames.get(cv2.CAP_PROP_FRAME_COUNT)); 
        num_frames = self.num_video_frames; 
        
        if (total_frames <= 0): 
            captured_frames.release() 
            return torch.zeros(num_frames, 3, self.frame_height, self.frame_width); 

        #choose the video frames to sample. 
        frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int); 
        target_set = set(frame_indices.tolist()); #all target indices to grab.


        frames = []; 
        current_index = 0; 

        

        while True: 
            ret, frame = captured_frames.read() 

            if not ret: 
                break; #no more frames to process

            if current_index in target_set: #keep only sample frames
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); 
                frame = cv2.resize(frame,(self.frame_width, self.frame_height)); 
                frame = frame.astype(np.float32)/255.0; 
                frame = torch.from_numpy(frame).permute(2,0,1); 
                frames.append(frame); #append as a tensor. 


            current_index+=1; 

        captured_frames.release(); 

        while (len(frames) < num_frames): #every sample should have the same amt of frames. 
            frames.append(torch.zeros(3, self.frame_height, self.frame_width)); 
        
        frames = frames[:num_frames]; 
        
        return torch.stack(frames); 












"""@function  
        in: batch of samples containing features, labels, transcripts, and a sample id. 
        out : normalized features, flattened transcripts, the length of ranscript, and the sample id. 
    """

def collate_fn(batch):
    features, labels, transcripts, sample_ids = zip(*batch); 
    
    input_len = torch.tensor([feature.size(1) for feature in features], dtype = torch.long); 
    target_len = torch.tensor([label.size(0) for label in labels], dtype = torch.long); 

    features = [feature.squeeze(0) for feature in features];

    padded_features = pad_sequence(features, batch_first=True);

    padded_features = padded_features.unsqueeze(1); 

    flat_labels = torch.cat(labels); 

    return padded_features, flat_labels, input_len, target_len, transcripts, sample_ids; 


"""@function  
        in: batch of samples containing features, labels, transcripts, and a sample id. 
        out : normalized features, flattened transcripts, the length of ranscript, and the sample id. 
    """

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




