from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.nn as nn
from util.utils import Glove
import json


class WordEmb:
    def __init__(self, cfg):
        self.glove = Glove(glove_dir=cfg.glove_dir,
                           glove_zip_save_path=cfg.glove_zip_save_path)
        self.embedding = nn.Embedding.from_pretrained(
            glove.embedding_matrix, freeze=True)
    
class CharEmb(nn.Embedding):
    def __init__(self, emb_dim=100):
        super(CharEmb, self).__init__(num_embeddings=70, embedding_dim=emb_dim)

class CNN_CharEmb(nn.Module):
    def __init__(self, model_dim=300):
        super(CNN_CharEmb, self).__init__()
        
        self.char_emb = CharEmb(emb_dim=100)
        self.cnn = nn.Conv1d(100,300,1,1,0)

    def forward(self, x, wordidx):
            
        x = self.char_emb(x).permute(0,2,1)
        x = self.cnn(x).permute(0,2,1)
        
        wordidx = np.array(wordidx)

        output = []
        first_word=False
        for bi, batch in enumerate(wordidx):
            batch_output=[]
            tmp_word = torch.empty(size=(1,300))
            for wi,idx in enumerate(batch):
                if wordidx[bi,wi]==0:
                    wordRepresentation = torch.max(tmp_word,dim=0,keepdim=True).values #(1,hidden_dim)
                    wordRepresentaiton = wordRepresentation.unsqueeze(0) #(1,1,hidden_dim)
                    batch_output.append(wordRepresentation)
                    first_word=False
                elif wordidx[bi,wi]==-1:
                    break
                else:
                    if first_word==False:
                        tmp_word=x[bi,wi,:].unsqueeze(0) # (1,hidden_dim)
                        first_word=True
                    else:
                        tmp_word = torch.cat((tmp_word,x[bi,wi,:].unsqueeze(0)),dim=0) # appending (1,hidden_dim)
            tmp_sentence = torch.cat(batch_output,dim=0) # (sequence_length,hidden_dim)
            output.append(tmp_sentence)

        final_output = pad_sequence(output,batch_first=True) # (Batch_Size, Character_Sequence_Length, Hidden_size)
        
        return final_output

if __name__ == "__main__":
    # download_glove_txt(".")
    with open("config/config.json", "r") as f:
        cfg = json.load(f)

    glove = Glove(
        glove_dir="/mnt/c/Users/mapoo/Documents/BiDAF_torch/glove", embedding_dim=300, glove_zip_save_path="./"
    )

    cnn = CNN_CharEmb()