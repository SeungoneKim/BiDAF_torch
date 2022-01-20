import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from data.tokenizer import Tokenizer

class SQuAD_Dataset(Dataset):
    def __init__(self, split_type, max_len=128, truncate=False):
        self.tokenizer = Tokenizer(max_len)
        self.truncate = truncate
        self.data = load_dataset('squad','plain_text',split=split_type)
        self.data_len = len(self.data)
        
        self.question = self.data['question']
        self.answer = self.data['answers']
        self.context = self.data['context']

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        question = self.question[index]
        answer = self.answer[index]
        context = self.context[index]

        answer_start_index = answer['answer_start'][0]
        answer_end_index = answer['answer_start'][0] + len(answer['text'])-1

        encoded_question = self.tokenizer.tokenize(question,self.truncate)
        encoded_context = self.tokenizer.tokenize(context,self.truncate)

        item = {}
        item['question']= encoded_question
        item['context']= encoded_context
        item['answer']= (answer_start_index, answer_end_index)

        return item

class SQuAD_Dataset_Total():
    def __init__(self,max_len=128,truncate=False):
        self.train_dataset = SQuAD_Dataset('train',max_len,truncate)
        train_len = len(self.train_dataset)*0.9
        val_len = len(self.train_dataset)-train_len
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(self.train_dataset,
            [train_len,val_len]
        )
        self.test_dataset = SQuAD_Dataset('validation',max_len,truncate)

    def getTrainData(self):
        return self.train_dataset
    
    def getValData(self):
        return self.validation_dataset
    
    def getTestData(self):
        return self.test_dataset
