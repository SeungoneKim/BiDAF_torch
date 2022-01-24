from nltk.tokenize import TreebankWordTokenizer
import pickle
import torch
import numpy as np

class Tokenizer():
    def __init__(self, max_len):
        self.tokenizer = TreebankWordTokenizer()
        with open('BiDAF_torch\glove\glove.6B.300d_vocab.pkl','rb') as pk:
            self.vocab = pickle.load(pk)
        self.vocablist = self.vocab.get_stoi()
        self.wordlist = self.vocab.get_itos()
        self.max_len = max_len

        self.characterset = "abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]"+"{"+"}"
        self.characterDictionary = {}
        for idx,char in enumerate(self.characterset):
            self.characterDictionary[char]=idx
        self.characterDictionary["<unk>"]=70


    # input = "Which laws faced significant opposition?"
    # output = ['Which', 'laws', 'faced', 'significant', 'opposition', '?']
    def encode(self, sentence):
        return self.tokenizer.tokenize(sentence)


    # input = "Which laws faced significant opposition?"
    # encoded_sentence = ['Which', 'laws', 'faced', 'significant', 'opposition', '?']
    # output = [386474, 218595, 143885, 331417, 270822, 42963]
    def tokenize(self, sentence, truncation=False):
        encoded_sentence = self.encode(sentence)
        
        final_output = []
        for word in encoded_sentence:
            try:
                final_output.append(self.vocablist[word.lower()])
            except:
                continue
        if truncation==True:
            if len(final_output) > self.max_len:
                final_output = final_output[:max_len]
        
        return final_output

    def tokenize_char(self, sentence):
        encoded_input = []
        which_word = []
        word_idx = 1

        for char in sentence:
            try:
                x = self.characterDictionary[char]
            except:
                x = self.characterDictionary['<unk>']
            encoded_input.append(x)
            # x==36 is ' '
            if x!=36:
                which_word.append(word_idx)
            else:
                which_word.append(-1)
                word_idx+=1

        return encoded_input, which_word

    def decode(self, tensor):
        decoded_output = []
        batch_sequence = np.array(tensor).astype(int) # (batch_size, sequence_length)
        for sequence in batch_sequence:
            tmp_output=""
            for token in sequence:
                tmp_output += self.wordlist[token]
                tmp_output += " "
            
            decoded_output.append(tmp_output.strip())
        
        return decoded_output


if __name__ == "__main__":
    tokenizer = Tokenizer(128)
    x = "Which laws faced significant opposition?"
    y = tokenizer.tokenize(x)
    z = tokenizer.decode(torch.Tensor(y).unsqueeze(0))
    w,u = tokenizer.tokenize_char(x)

    print(x)
    print(y)
    print(z)
    print(w)
    print(u)