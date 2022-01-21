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
        glove_dir=cfg.glove_dir, embedding_dim=cfg.embedding_dim, glove_zip_save_path=cfg.glove_zip_save_path)
