import torch 
import torch.nn as nn 



class AudioModel(nn.Module): 
    def __init__(self, spec_bins, hidden_dim, layers, vocab_size): 
        super.__init__(); 


    def forward(self,x): 
        pass;


class VideoAudioModel(nn.Module): 
    def __init__(self): 
        super.__init__(); 
    

        self.front = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels = 64, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2), 


            nn.Conv2D(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=4); 
        )


    def forward(self,x): 
        pass;
