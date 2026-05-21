from sample import build_samples
from main import TRAIN_PATH
from dataset import normalize 


def ken_lm(path, output_path = "l.txt"): 

    with open(output_path, "w", encoding = "utf-8") as out: 
        for file in path: 
            samples = build_samples(file)
            for sample in samples: 
                with open(sample.txt, "r", encoding = "utf-8") as f: 
                    line = normalize(f.read()); 
                    if line is not None: 
                        out.write(line + "\n"); 

def lex(corp = "l.txt", lex_path = "lexicon.txt"): 
    words = set(); 

    with open(corp, "r", encoding = "utf-8") as f: 
        for line in f: 
            for word in line.strip().split(): 
                words.add(word);

    with open(lex_path, "w", encoding = "utf-8") as out: 
        for word in sorted(words): 
            spelling = " ".join(list(word)); 
            out.write(f"{word} {spelling}\n"); 


if __name__ == "__main__": 
    ken_lm([TRAIN_PATH]); 
    lex(); 