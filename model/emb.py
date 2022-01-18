import wget
import os
import zipfile
from tqdm import tqdm
import numpy as np


def download_glove_txt(save_path):
    """ 
    downloads glove.6B zip file from the url.
    the zip file contains 100d, 200d and 300d embedding
    """
    url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
    file_name = url.split("/")[-1]
    wget.download(url, os.path.join(save_path, file_name))


def unzip_glove_zip(glove_zip_file_path, save_path):
    """
    unzip downloaded glove6B.zip file

    Args:
        glove_zip_file_path (str): path of the glove zip file
        save_path (str): the directory of where extracted files to be saved
    """
    with zipfile.ZipFile(glove_zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)


def load_glove_dict_from_txt(glove_dir, embedding_dim=300):
    """
    load glove embedding dictionary from the .txt file

    Args:
        glove_dir (string): the path of glove directory
        embedding_dim (int, optional): glove embedding dimension. [50, 100, 200, 300] Defaults to 300.
    """
    with open(os.path.join(glove_dir, f"glove.6B.{embedding_dim}d.txt", "r")) as f:
        embedding_dict = {}
        print("started to read lines in txt file...")
        for line in tqdm(f):
            values = line.split()
            embedding_vector = np.asarray(values[-embedding_dim:], "float32")
            word = "".join(values[:-embedding_dim])
            embedding_dict[word] = embedding_vector

    return embedding_dict


if __name__ == "__main__":
    # download_glove_txt(".")
    unzip_glove_zip(
        "/Users/chaehyeongju/Documents/BiDAF/6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837.zip", "glove")
