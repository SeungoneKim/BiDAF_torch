import wget
import os
import zipfile
from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator


class Glove:
    def __init__(self, glove_dir, embedding_dim=300, glove_zip_file_path=None, glove_zip_save_path=None):
        """[summary]
        Args:
            glove_dir ([type]): [description]
            embedding_dim (int, optional): [description]. Defaults to 300.
            glove_zip_file_path ([type], optional): [description]. Defaults to None.
            glove_zip_save_path ([type], optional): [description]. Defaults to None.
        """
        self.glove_zip_file_path = glove_zip_file_path
        self.glove_dir = glove_dir
        self.embedding_dim = embedding_dim
        self.glove_zip_save_path = glove_zip_save_path

        # check if pickle files already exist
        if not os.path.isfile(os.path.join(glove_dir, f"glove.6B.{embedding_dim}d.pkl")):

            # if .txt file does not exist, unzip downloaded .zip file
            if not os.path.isfile(os.path.join(glove_dir, f"glove.6B.{embedding_dim}d.txt")):
                if glove_zip_file_path == None:  # check if glove zip file exists
                    print(".zip file does not exist. start downloading .zip file...")
                    self.glove_zip_save_path = "./" if glove_zip_save_path == None else glove_zip_save_path
                    self.download_glove_txt()
                    print("download finished.")

                self.unzip_glove_zip()

            # read txt file and load glove dictionary
            glove_dict = self.load_glove_dict_from_txt()

            self.vocab = self.make_vocab_from_glove_dict_and_save_as_pickle(
                glove_dict)
            self.embedding_matrix = self.make_embedding_matrix_and_save_as_pickle(
                glove_dict, self.vocab)

        else:  # when pickle files exist, you dont need to make vocab and matrix again. just load them from pickle files
            self.vocab = self.load_vocab_from_pickle()
            self.embedding_matrix = self.load_embedding_matrix_from_pickle()

    def download_glove_txt(self):
        """ 
        downloads glove.6B zip file from the url.
        the zip file contains 100d, 200d and 300d embedding
        """
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        file_name = url.split("/")[-1]
        wget.download(url, os.path.join(self.glove_zip_save_path, file_name))
        self.glove_zip_file_path = os.path.join(
            self.glove_zip_save_path, file_name)

    def unzip_glove_zip(self):
        """
        unzip downloaded glove6B.zip file
        Args:
            glove_zip_file_path (str): path of the glove zip file
            save_path (str): the directory of where extracted files to be saved
        """
        with zipfile.ZipFile(self.glove_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.glove_dir)

    def load_glove_dict_from_txt(self):
        """
        load glove embedding dictionary from the .txt file
        Args:
            glove_dir (string): the path of glove directory
            embedding_dim (int, optional): glove embedding dimension. [50, 100, 200, 300] Defaults to 300.
        """
        with open(os.path.join(self.glove_dir, f"glove.6B.{self.embedding_dim}d.txt"), "r") as f:
            embedding_dict = {}
            print("started to read lines in txt file...")
            for line in tqdm(f):
                values = line.split()
                embedding_vector = np.asarray(
                    values[-self.embedding_dim:], "float32")
                word = "".join(values[:-self.embedding_dim])
                embedding_dict[word] = embedding_vector

        return embedding_dict

    def make_vocab_from_glove_dict_and_save_as_pickle(self, glove_dict):
        """
        make torchtext.vocab from glove embedding dictionary and save it as pikle file
        Args:
            glove_dict (dict): key(token), value(embedding vector, np.array)  
        Returns:
            vocab : torchtext.vocav object. it will be used when making embedding matrix
        """
        word_list = [glove_dict.keys()]
        vocab = build_vocab_from_iterator(word_list)

        with open(os.path.join(self.glove_dir, f"glove.6B.{self.embedding_dim}d_vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)

        return vocab

    def make_embedding_matrix_and_save_as_pickle(self, glove_dict, vocab):
        """
        make embedding matrix with glove dict and vocab in np.array type and save it as pickle file
        Args:
            glove_dict (dict): key(token), value(embedding vector, np.array) 
            vocab (torchtext.vocab): contains vocab made with glove dict
        Returns:
            embedding matrix: list of numpy array, contains embedding vectors
        """
        embedding_matrix = []
        itos = vocab.get_itos()
        print("started to make an embedding matrix")
        for i in tqdm(range(len(vocab))):
            token_text = itos[i]
            if token_text in glove_dict:
                embedding_matrix.append(glove_dict[token_text])

        else:
            embedding_matrix.append(
                np.zeros(self.embedding_dim, dtype="float32"))
        print("finished making an emedding matrix")
        with open(os.path.join(self.glove_dir, f"glove.6B.{self.embedding_dim}d.pkl"), "wb") as f:
            pickle.dump(embedding_matrix, f)

        return embedding_matrix

    def load_embedding_matrix_from_pickle(self):
        print("started to load embedding matrix...")
        with open(os.path.join(self.glove_dir, f"glove.6B.{self.embedding_dim}d.pkl"), "rb") as f:
            embedding_matrix = pickle.load(f)
            print("embedding matrix is loaded successfully!")
            return embedding_matrix

    def load_vocab_from_pickle(self):
        print("started to load vocab...")
        with open(os.path.join(self.glove_dir, f"glove.6B.{self.embedding_dim}d_vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
            print("vocab is loaded successfully!")
            return vocab
