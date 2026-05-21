import torch 
from torchaudio.models.decoder import ctc_decoder 
from torchmetrics.text import WordErrorRate, CharErrorRate


"""@function 
        Greedy decoding function for final tensors to produce human-readable text. 
        in : encoded model output, decoder for idx to char 
        out : decoded string
"""   
def decode(encoded : list[int], idx_to_char: dict[int, str]) -> str: 
    decoded = []; 
    prev = None;

    for c in encoded: 
        if (prev != c and c != 0): 
            decoded.append(idx_to_char[c]); 
        prev = c; 

        
    return "".join(decoded); 

"""@function 
        Beam-search based decoding function for final tensors to produce human-readable text. 
        in : raw logits, expected final length, decoder method
        out : decoded string
"""   
def beam_decode(logit, length, decoder): 

    l = int(length); #get length of the sequence
    probabilities = logit[:l].log_softmax(dim = -1).detach().cpu().unsqueeze(0); #move the encoded unit to cpu and pad. 
    results = decoder(probabilities); 

    h = results[0][0]; 
    text = "".join(decoder.idxs_to_tokens(h.tokens)); 
    text = text.replace("<blank>", ""); #remove blank occurrences
    text = " ".join(text.split()); 

    return text; 




"""@function 
        One epoch train loop - train, calculate loss, then backpropagate. (NO VIDEO FRAMES USED)
        in : training model, data loader, training optimizer, criterion (loss function), device 
        out : average training loss 
"""   
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


"""@function 
        One epoch validation loop - validation, calculate loss, check for overfitting in main etc.  (NO VIDEO FRAMES USED)
        in : training model, validation loader, training optimizer, criterion (loss function), device 
        out : average validation loss 
"""  
@torch.no_grad()
def val_one_epoch(model, loader, criterion, device, testing = False): 
    model.eval(); 
    total_loss = 0.0; 

    for batch in loader: #loop entire batch
        features, labels, input_len, target_len, transcripts, sample_ids = batch; #pull details 
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

"""@function 
        One epoch test loop - evaluation, calculate loss, then backpropagate. (NO VIDEO FRAMES USED)
        in : training model, validation loader, training optimizer, criterion (loss function), device 
        out : average validation loss 
"""  


@torch.no_grad() 

def test_one_epoch(model, loader, criterion, device, int_to_char,output_path,decoder=None): 
    model.eval(); 
    total_loss = 0.0; 
    tr = []; 
    predictions = [];
    output_path = output_path
    with open (output_path, "w", encoding="utf-8") as out: 
        for batch in loader:  #loop through each batch, extract and unzip the features/labels and length. 
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
            predicted = logits.argmax(dim=-1); #run argmax on the character prediction dimension 


            for i in range(predicted.size(0)): 
                #run decoder on each character and write entire output to file. 
                if decoder is None: 
                    predicted_text = decode(predicted[i].detach().cpu().tolist(), int_to_char); 
                else: 
                    predicted_text = beam_decode(logits[i], int(input_len[i].item()), decoder); 
                
                out.write(f"Transcript: {transcripts[i]}\n"); 
                tr.append(transcripts[i]);
                out.write(f"Predicted Text: {predicted_text}\n"); 
                predictions.append(predicted_text)
                out.write(f"\n"); 

    wer = WordErrorRate(); 
    cer = CharErrorRate(); 

    w = wer(predictions, tr).item(); 
    c = cer(predictions, tr).item(); 
    avg_loss = total_loss/len(loader)
    return avg_loss, w, c; 


            


"""@function 
        One epoch train loop - train, calculate loss, then backpropagate. (VIDEO FRAMES USED)
        in : training model, data loader, training optimizer, criterion (loss function), device 
        out : average training loss 
"""   

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




"""@function 
        One epoch validation loop - validation, calculate loss, check for overfitting in main etc.  (VIDEO FRAMES USED)
        in : training model, validation loader, training optimizer, criterion (loss function), device 
        out : average validation loss 
"""  
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


"""@function 
        One epoch test loop - evaluation, calculate loss, then backpropagate. (VIDEO FRAMES USED)
        in : training model, validation loader, training optimizer, criterion (loss function), device 
        out : average validation loss 
"""  

@torch.no_grad() 

def test_one_epoch_video(model, loader, criterion, device, int_to_char, output_path, decoder = None): 
    model.eval(); 
    total_loss = 0.0;
    tr = []; 
    predictions = [];  
    

    output_path = output_path; 

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
                    predicted_text = beam_decode(logits[i], int(input_len[i].item()), decoder);      
                out.write(f"Transcript: {transcripts[i]}\n"); 
                tr.append(transcripts[i]); 
                out.write(f"Predicted Text: {predicted_text}\n"); 
                predictions.append(predicted_text); 
                out.write(f"\n"); 

    
    #instantialize objs 
    wer = WordErrorRate(); 
    cer = CharErrorRate(); 

    w = wer(predictions, tr).item(); 
    c = cer(predictions, tr).item(); 
    avg_loss = total_loss/len(loader)
    return avg_loss, w, c; 





        