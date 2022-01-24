from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from data.dataset import SQuAD_Dataset_Total


def make_batch(samples):
    questions = [sample['question'] for sample in samples]
    contexts = [sample['context'] for sample in samples]
    char_questions = [sample['character_question'] for sample in samples]
    char_questions_wordidx = [sample['character_question_wordidx'] for sample in samples]
    char_contexts = [sample['character_context'] for sample in samples]
    char_contexts_wordidx = [sample['character_context_wordidx'] for sample in samples]
    start_answers = [sample['start_answer'] for sample in samples]
    end_answers = [sample['end_answer'] for sample in samples]
    

    padded_questions = pad_sequence(questions,padding_value=400001)
    padded_contexts = pad_sequence(contexts,padding_value=400001)
    padded_char_questions = pad_sequence(char_questions,padding_value=71)
    padded_char_contexts = pad_sequence(char_contexts,padding_value=71)
    padded_char_questions_wordidx = pad_sequence(char_questions_wordidx,padding_value=-1)
    padded_char_contexts_wordidx = pad_sequence(char_contexts_wordidx,padding_value=-1)
    
    
    return {
        'question':padded_questions.contiguous().permute(1,0),
        'context':padded_contexts.contiguous().permute(1,0),
        'character_question' : padded_char_questions.contiguous().permute(1,0),
        'character_question_wordidx' : padded_char_questions_wordidx.contiguous().permute(1,0),
        'character_context' : padded_char_contexts.contiguous().permute(1,0),
        'character_context_wordidx' : padded_char_contexts_wordidx.contiguous().permute(1,0),
        'start_answer' : (torch.stack(start_answers,dim=1)).squeeze(0).contiguous(),
        'end_answer' : (torch.stack(end_answers,dim=1)).squeeze(0).contiguous()
    }

def SQuAD_DataLoader(train_bs, val_bs, test_bs, max_len, truncate):
    dataset = SQuAD_Dataset_Total(max_len,truncate)

    train_dataloader = DataLoader(dataset=dataset.getTrainData(),
                                batch_size=train_bs,
                                collate_fn=make_batch,
                                shuffle=True)

    val_dataloader = DataLoader(dataset=dataset.getValData(),
                            batch_size=val_bs,
                            collate_fn=make_batch,
                            shuffle=True)

    test_dataloader = DataLoader(dataset=dataset.getTestData(),
                            batch_size=test_bs,
                            collate_fn=make_batch,
                            shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    train_dl, val_dl, test_dl = SQuAD_DataLoader(4,4,4,128,False)

    idx=0
    for item in train_dl:
        if idx==2:
            break
        # (batch_size, sequence_length)
        print(item['question'].size())
        print(item['context'].size())
        print(item['character_question'].size())
        print(item['character_question_wordidx'].size())
        print(item['character_context'].size())
        print(item['character_context_wordidx'].size())
        # (batch_size)
        print(item['start_answer'].size())
        print(item['end_answer'].size())
        idx+=1