from dataclasses import dataclass
from pathlib import Path
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip

@dataclass
class Sample: 
    id: str; #doc id
    mp4: str;  #mp4 file id
    txt: str;  # transcript id
    view: str | None = None; #video no audio file id
    wav: str | None = None;  #audio no video file id 




def build_samples(root : str) -> list[Sample]: 
    root_path = Path(root); 


    samples = []; #generate samples. 

    for mp4_file in root_path.rglob("*.mp4"): 
        txt_file = mp4_file.with_suffix(".txt"); 
        wav_file = mp4_file.with_suffix(".wav"); 

        if (txt_file.exists()): 
            
            doc_id = str(mp4_file.with_suffix("").relative_to(root_path)); 

            if (wav_file.exists()):
                samples.append(
                    Sample( 
                        id = doc_id, 
                        mp4 = str(mp4_file), 
                        txt = str(txt_file), 
                        view = None,
                        wav = str(wav_file)
                        )
                    ); 
    

    print(f"Finished samples for {root}.");

    return samples; 

