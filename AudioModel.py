import torch 
import torch.nn as nn 
import torch.nn.functional as F



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


# audio + video pipeline. 
class VideoAudioModel(nn.Module): 
    def __init__(self, spec_bins, hidden_dim, layers, vocab_size, video_feature_dim): 
        super().__init__();
    

    #audio pipeline: 
        self.audio_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels = 64, kernel_size = 3, padding = 1), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=(1,2)), # pool frequency, not time. 


                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=(1,2))
            )
        

        audio_out_size = 128 * (spec_bins // 4); 


        self.video_cnn = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels=32, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3, padding = 1), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1,1))
        ); 

        self.video_proj = nn.Linear(128, video_feature_dim);    

        total_dim = audio_out_size + video_feature_dim; 

        
        self.lstm = nn.LSTM( 
            input_size = total_dim, 
            hidden_size = hidden_dim, 
            num_layers = layers, 
            batch_first = True, 
            bidirectional = True
        )




        self.classifier = nn.Linear(hidden_dim * 2, vocab_size); 



        

    

        


    def forward(self,audio_x, video_x): 
        
        
        audio_x = self.audio_cnn(audio_x); 
        b,c,t_a,f = audio_x.size(); #batch, channels, audio steps, frequency bins. 
        audio_x = audio_x.permute(0,2,1,3); 
        audio_x = audio_x.contiguous().view(b,t_a, c*f); 


        b, t_v, ch, h, w = video_x.size();  #batch, video frame count, channels, height, width
        video_x = video_x.view(b*t_v, ch, h, w); 
        video_x = self.video_cnn(video_x); 
        video_x = video_x.view(b*t_v, 128); 
        video_x = self.video_proj(video_x); 
        video_x = video_x.view(b,t_v, -1); 



        if (t_v != t_a): 
            video_x = video_x.permute(0,2,1); 
            video_x = F.interpolate(video_x, size=t_a, mode = "linear", align_corners = False); 
            video_x = video_x.permute(0,2,1); 

        vid_input = torch.cat([audio_x, video_x], dim = -1); 
        vid_input,_ = self.lstm(vid_input); 
        logits = self.classifier(vid_input); 
        return logits; 


        
