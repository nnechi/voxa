from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample: 
    id: str; #doc id
    mp4: str;  #mp4 file id
    txt: str;  # transcript id
    wav: str | None = None;  #audio no video file id 




"""@function 
        Create sample objects from given directory. (Raw samples, must be cleaned up later using helper functions in dataset. )
        in : file root
        out : list of samples present in the directory organized. 
"""  
def build_samples(root : str) -> list[Sample]: 
    root_path = Path(root); 

    samples = []; #generate samples. 

    for mp4_file in root_path.rglob("*.mp4"): 
        txt_file = mp4_file.with_suffix(".txt"); 
        wav_file = mp4_file.with_suffix(".wav"); 
        if (txt_file.exists() and wav_file.exists()): #if it is one complete sample. 
                samples.append(
                    Sample( 
                        id = str(mp4_file.with_suffix("").relative_to(root_path)),
                        mp4 = str(mp4_file), 
                        txt = str(txt_file), 
                        wav = str(wav_file)
                        )
                    ); 
    

    print(f"Finished samples for {root}.");

    return samples; 



     

