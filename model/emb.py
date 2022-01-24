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


if __name__ == "__main__":
    # download_glove_txt(".")
    with open("config/config.json", "r") as f:
        cfg = json.load(f)

    glove = Glove(

        glove_dir="/mnt/c/Users/mapoo/Documents/BiDAF_torch/glove", embedding_dim=300, glove_zip_save_path="./")
    
class CharEmb(nn.Embedding):
    def __init__(self, emb_dim=100):
        super(CharEmb, self).__init__(num_embeddings=72, embedding_dim=emb_dim)

class CNN_CharEmb(nn.Module):
    def __init__(self, model_dim=300):
        super(CNN_CharEmb, self).__init__()
        
        self.char_emb = CharEmb(emb_dim=100)
        self.cnn = nn.Conv1d(100,300,1,1,0)

    def forward(self, x):
        
        output = self.char_emb(x).permute(0,2,1)
        output = self.cnn(output).permute(0,2,1)
        
        # (Batch_Size, Character_Sequence_Length, Hidden_size)
        return output



        
