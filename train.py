import torch 

#greedy ctc decode (remove dups.)
def decode(encoded : list[int], idx_to_         : dict[int, str]) -> str: 
    decoded = []; 
    prev = "";

    for c in encoded: 
        if (prev != c and c != 0): 
            decoded.append(idx_to_char[c]); 
        prev = c; 

        

    return "".join(decoded); 


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
def validate_one_epoch(model, loader, criterion, device): 
    model.eval(); 
    total_loss = 0.0; 

    for batch in loader: 
        features, labels, input_len, target_len, transcripts, sample_ids = batch; 
        features = features.to(device); 
        labels = labels.to(device); 
        target_len = target_len.to(device); 
        input_len = input_len.to(device); 
        logits = model(features); 
        probabilities = logits.log_softmax(dim=-1); 
        probabilities = probabilities.permute(1,0,2); 

        loss = criterion(probabilities, labels, input_len, target_len); 
        total_loss += loss.item(); 


    return total_loss / len(loader); 





        

