# A Lightweight CNN-BiLSTM Audio-Visual Pipeline 

Lightweight automatic speech recognition pipeline built using PyTorch for comparing: 

audio-only vs audio-visual transcription 

Compatibility between multiple decoding strategies: 
-greedy decoding
-beam search 
-KenLM no Lex 
-KenLM lex 

The project utilizes the LRS2 dataset and evaluates models with CTC Loss, WER, and CER metrics. 


## Goal 

The project explores whether a lightweight CNN + BiLSTM architecture provides a reproducible ASR baseline. 

## Instructions: 

pip install the following : 
torch
torchaudio
torchmetrics
numpy
opencv-python 
openai-whisper
flashlight-text

And run: 
sudo apt install ffmpeg 

# Running the program: 
Separate the LRS2 data into Pretrain, Train, Validation, and Test folders in order to divide samples. 

Change TRAIN_PATH, PRETRAIN_PATH, VAL_PATH, TEST_PATH to respective path folders on your system. 

Run python3 main.py 

Will produce transcript text in text files underneath: 
audio_model_greedy.txt 
audio_model_beam.txt 
audio_model_kenlm_nolexicon.txt 
audio_model_kenlm_lexicon.txt 
av_model_greedy.txt 
av_model_beam.txt 
av_model_kenlm_nolexicon.txt 
av_model_kenlm_lexicon.txt 


Each output file will contain 

a reference transcript, 
predicted text, 
final test loss, 
WER,
CER. 

Epochs are printed in the terminal. 

You can run python3 whisper_eval.py -> whisper.txt and it can be used as a baseline. 




