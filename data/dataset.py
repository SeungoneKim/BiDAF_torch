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

        # Input for word embedding
        encoded_question = self.tokenizer.tokenize(question,self.truncate)
        encoded_context = self.tokenizer.tokenize(context,self.truncate)

        # Input for character embedding
        encoded_character_question = self.tokenizer.tokenize_char(question)
        encoded_character_context = self.tokenizer.tokenize_char(context)

        # transform into Torch Tensor
        item = {}
        item['question']= torch.Tensor(encoded_question).to(torch.long)
        item['context']= torch.Tensor(encoded_context).to(torch.long)
        item['character_question'] = torch.Tensor(encoded_character_question[0]).to(torch.long)
        item['character_question_wordidx'] = torch.Tensor(encoded_character_question[1]).to(torch.long)
        item['character_context'] = torch.Tensor(encoded_character_context[0]).to(torch.long)
        item['character_context_wordidx'] = torch.Tensor(encoded_character_context[1]).to(torch.long)
        item['start_answer']= torch.Tensor([answer_start_index]).to(torch.long)
        item['end_answer']=torch.Tensor([answer_end_index]).to(torch.long)

        return item

class SQuAD_Dataset_Total():
    def __init__(self,max_len=128,truncate=False):
        self.total_dataset = SQuAD_Dataset('train',max_len,truncate)
        train_len = int(len(self.total_dataset)*0.9)
        val_len = len(self.total_dataset)-train_len
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(self.total_dataset,
            [train_len, val_len],
        )
        self.test_dataset = SQuAD_Dataset('validation',max_len,truncate)

    def getTrainData(self):
        return self.train_dataset
    
    def getValData(self):
        return self.validation_dataset
    
    def getTestData(self):
        return self.test_dataset

if __name__ == "__main__":
    total_ds = SQuAD_Dataset_Total(128,False)
    train_ds, val_ds, test_ds = total_ds.getTrainData(), total_ds.getValData(), total_ds.getTestData()

    print(train_ds[0])
    print()
    print(val_ds[0])
    print()
    print(test_ds[0])