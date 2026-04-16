from dataclasses import dataclass
from pathlib import Path



@dataclass
class Sample: 
    id: str;
    mp4: str; 
    txt: str; 
    view: str | None = None; 


def build_samples(root : str) -> list[Sample]: 
    root_path = Path(root); 


    samples = []; #generate samples. 

    for mp4_file in root_path.rglob("*.mp4"): 
        txt_file = mp4_file.with_suffix(".txt"); 

        if (txt_file.exists()): 
            
            doc_id = str(mp4_file.with_suffix("").relative_to(root_path))

            samples.append(
                Sample( 
                    id = doc_id, 
                    mp4 = str(mp4_file), 
                    txt = str(txt_file), 
                    view = None
                    )
            ); 
    

    print(f"Finished samples for {root}.");

    return samples; 



