import torch 
from torchaudio.models.decoder import ctc_decoder 



#greedy ctc decode (remove dups.)
def decode(encoded : list[int], idx_to_char: dict[int, str]) -> str: 
    decoded = []; 
    prev = None;

    for c in encoded: 
        if (prev != c and c != 0): 
            decoded.append(idx_to_char[c]); 
        prev = c; 

        

    return "".join(decoded); 

#CTC BEAM DECODE : 
def beam_decode(logit, decoder) -> list[str]: 
    probabilities = logit.log_softmax(dim = -1).detach().cpu().unsqueeze(0); 
    results = decoder(probabilities); 

    h = results[0][0]; 
    text = "".join(decoder.idxs_to_tokens(h.tokens)); 
    text = text.replace("<blank>", ""); 
    text = " ".join(text.split()); 

    return text; 




#train loop. 
def train_one_epoch(model, loader, optimizer, criterion, device): 
    model.train(); 
    total_loss = 0.0; 
    for batch in loader: 
        features, labels, input_len, target_len, transcripts, sample_ids = batch; 

        features = features.to(device); 
        labels = labels.to(device); 
        input_len = input_len.to(device); 
        target_len = target_len.to(device); 

        optimizer.zero_grad(); 

        logits = model(features); 
        probabilities = logits.log_softmax(dim=-1); 
        probabilities = probabilities.permute(1,0,2); 

        loss = criterion(probabilities, labels, input_len, target_len);

        total_loss += loss.item(); 

        loss.backward(); 
        #add gradient clipping here for LSTM  
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1); 
        optimizer.step(); 



    return total_loss / len(loader); 


#validation loop. 
@torch.no_grad()
def val_one_epoch(model, loader, criterion, device, testing = False): 
    model.eval(); 
    total_loss = 0.0; 

    for batch in loader: 
        features, labels, input_len, target_len, transcripts, sample_ids = batch; 
        features = features.to(device); 
        labels = labels.to(device); 
        input_len = input_len.to(device); 
        target_len = target_len.to(device); 
       
        logits = model(features); 
        probabilities = logits.log_softmax(dim=-1); 
        probabilities = probabilities.permute(1,0,2); 

        loss = criterion(probabilities, labels, input_len, target_len); 
        total_loss += loss.item(); 


    return total_loss / len(loader); 




@torch.no_grad() 

def test_one_epoch(model, loader, criterion, device, int_to_char,decoder=None): 
    model.eval(); 
    total_loss = 0.0; 
    
    output_path = "audio_predictions.txt" if not decoder else "audio_beam_predictions.txt"
    with open (output_path, "w", encoding="utf-8") as out: 
        for batch in loader: 
            features, labels, input_len, target_len, transcripts, sample_ids = batch; 
            features = features.to(device); 
            labels = labels.to(device);   
            input_len = input_len.to(device);  
            target_len = target_len.to(device);  
            logits = model(features); 
            probabilities = logits.log_softmax(dim = -1); 
            probabilities = probabilities.permute(1,0,2); 
            loss = criterion(probabilities, labels, input_len, target_len); 
            total_loss += loss.item(); 
            predicted = logits.argmax(dim=-1); 


            for i in range(predicted.size(0)): 
            
                if decoder is None: 
                    predicted_text = decode(predicted[i].detach().cpu().tolist(), int_to_char); 
                else: 
                    predicted_text = beam_decode(logits[i], decoder); 
                out.write(f"Transcript: {transcripts[i]}\n"); 
                out.write(f"Predicted Text: {predicted_text}\n"); 
                out.write(f"\n"); 

    return total_loss/len(loader); 


            


def train_one_epoch_video(model, loader, optimizer, criterion, device): 
    model.train(); 
    total_loss = 0.0; 


    for batch in loader: 
        audio, video, labels, input_len, target_len, transcripts, sample_ids = batch; 
        audio = audio.to(device); 
        video = video.to(device); 
        labels = labels.to(device); 
        input_len = input_len.to(device); 
        target_len = target_len.to(device); 


        optimizer.zero_grad(); 

        logits = model(audio, video); 
        probabilities = logits.log_softmax(dim = -1); 
        probabilities = probabilities.permute(1,0,2); 

        loss = criterion(probabilities, labels, input_len, target_len); 

        total_loss += loss.item(); 

        loss.backward() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0); 

        optimizer.step(); 

    return total_loss/len(loader); 




@torch.no_grad() 

def val_one_epoch_video(model, loader, criterion, device): 
    model.eval(); 
    total_loss = 0.0; 

    for batch in loader: 
        audio, video, labels, input_len, target_len, transcripts, sample_ids = batch; 

        audio = audio.to(device); 
        video = video.to(device); 
        labels = labels.to(device); 
        input_len = input_len.to(device); 
        target_len = target_len.to(device); 

        logits = model(audio, video);  
        probabilities = logits.log_softmax(dim = -1); 
        probabilities = probabilities.permute(1,0,2); 

        loss = criterion(probabilities, labels, input_len, target_len); 


        total_loss += loss.item(); 


    return total_loss/len(loader); 




@torch.no_grad() 

def test_one_epoch_video(model, loader, criterion, device, int_to_char, decoder = None): 
    model.eval(); 
    total_loss = 0.0; 
    

    output_path = "audio_video_model_predictions.txt" if not decoder else "audio_video_beam_predictions.txt"; 
    with open (output_path, "w" , encoding = "utf-8") as out: 
        for batch in loader: 
            audio, video, labels, input_len, target_len, transcripts, sample_ids = batch; 
            audio = audio.to(device); 
            video = video.to(device); 
            labels = labels.to(device); 
            input_len = input_len.to(device); 
            target_len = target_len.to(device); 

            logits = model(audio, video); 
            probabilities = logits.log_softmax(dim=-1); 
            probabilities = probabilities.permute(1,0,2); 

            loss = criterion(probabilities, labels, input_len, target_len); 
            total_loss += loss.item(); 

            predicted = logits.argmax(dim = -1); 

        

            for i in range(predicted.size(0)): 
                if decoder is None: 
                    predicted_text = decode(predicted[i].detach().cpu().tolist(), int_to_char); 
                else : 
                    predicted_text = beam_decode(logits[i], decoder);        
                out.write(f"Transcript: {transcripts[i]}\n"); 
                out.write(f"Predicted Text: {predicted_text}\n"); 
                out.write(f"\n"); 

    

    return total_loss/len(loader); 





        