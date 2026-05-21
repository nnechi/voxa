import os 
import torch 
import whisper 
from torchmetrics.text import WordErrorRate, CharErrorRate 
from sample import build_samples 
from dataset import normalize 



TEST_PATH = r"/mnt/c/Users/nnechi/Documents/Code/Project/test"
FILE_PATH = "whisper.txt"

"""@function 
    use whisper as baseline on test set. call from this file. """
def eval(model_name = "base.en"): 
    device = "cuda" if torch.cuda.is_available() else "cpu"; 
    model = whisper.load_model(model_name, device=device); 
    samples = build_samples(TEST_PATH); 

    predictions = []; 
    labels = []; 

    with open(FILE_PATH, "w", encoding = "utf-8") as out: 
        for idx, sample in enumerate(samples, start=1): 
            with open(sample.txt, "r", encoding = "utf-8") as f: 
                transcript_ref = normalize(f.read()); 
                labels.append(transcript_ref); 

            prediction = model.transcribe(sample.wav, language = "en", task = "transcribe", fp16=torch.cuda.is_available(), verbose=False); 
            prediction = normalize(prediction["text"]); 
            predictions.append(prediction); 

            out.write(f"Transcript: {transcript_ref}\n"); 
            out.write(f"Predicted Text: {prediction}\n\n");
            
        wer = WordErrorRate(); 
        cer = CharErrorRate(); 

        w = wer(predictions, labels).item(); 
        c = cer(predictions, labels).item(); 

        out.write (f"Test WER: {w:.4f}\n");
        out.write(f"Test CER: {c:.4f}\n");

    return w, c; 

if __name__ == "__main__": 
    wer, cer = eval(); 
    print(f"Test WER: {wer:.4f}\n");
    print(f"Test CER: {cer:.4f}\n"); 




            
        
        
