import torch 
import torch.nn as nn 



class AudioModel(nn.Module): 
    def __init__(self, spec_bins, hidden_dim, layers, vocab_size): 
        super().__init__(); 
    

        # looking at a baseline of a CNN -> BiLSTM -> Classify
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels = 64, kernel_size = 3, padding = 1), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=(1,2)), # pool frequency, not time. 


                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=(1,2))
            )
        
        out_size = 128 * (spec_bins // 4); 
        
        self.lstm = nn.LSTM( 
            input_size = out_size, 
            hidden_size = hidden_dim, 
            num_layers = layers, 
            batch_first = True, 
            bidirectional = True
        )

#final classification on vocab label size. 
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size); 

    def forward(self,x): 
        x = self.cnn(x); 
        batch_size, channels, time_steps, freq_bins = x.size(); 
        x = x.permute(0,2,1,3); 
        x = x.contiguous().view(batch_size, time_steps, channels * freq_bins); 
        x, _ = self.lstm(x); 
        x = self.classifier(x); 
        return x; 



class VideoAudioModel(nn.Module): 
    def __init__(self): 
        super().__init__(); 
    

        


    def forward(self,x): 
        pass;
