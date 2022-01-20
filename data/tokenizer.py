from nltk.tokenize import TreebankWordTokenizer

class Tokenizer():
    def __init__(self, max_len):
        self.tokenizer = TreebankWordTokenizer()
        with open('BiDAF_torch\glove\glove.6B.300d_vocab.pkl','rb') as pk:
            self.vocab = pickle.load(pk)
        self.vocablist = self.vocab.get_stoi()
        self.wordlist = self.vocab.get_itos()
        self.max_len = max_len


    # input = "Which laws faced significant opposition?"
    # output = ['Which', 'laws', 'faced', 'significant', 'opposition', '?']
    def encode(self, sentence):
        return self.tokenizer.tokenize(sentence)


    # input = "Which laws faced significant opposition?"
    # encoded_sentence = ['Which', 'laws', 'faced', 'significant', 'opposition', '?']
    # output = [386474, 218595, 143885, 331417, 270822, 42963]
    def tokenize(self, sentence, truncation=False):
        encoded_sentence = self.encode(batch_sentence)
        
        final_output = []
        for word in encoded_sentence:
            final_output.append(self.vocablist[word.lower()])
        if truncation==True:
            if len(final_output) > self.max_len:
                final_output = final_output[:max_len]
        
        return final_output


    def decode(self, batch_sequence):
        decoded_output = []
        for sequence in batch_sequence:
            tmp_output=""
            for token in sequence:
                tmp_output += self.wordlist[token]
                tmp_output += " "
            
            decoded_output.append(tmp_output.strip())
        
        return decoded_output