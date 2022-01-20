from torch.utils.data import DataLoader, Dataset
from data.dataset import SQuAD_Dataset_Total

def SQuAD_DataLoader(train_bs, val_bs, test_bs, max_len, truncate):
    dataset = SQuAD_Dataset_Total(max_len,truncate)

    train_dataloader = DataLoader(dataset=dataset.getTrainData(),
                                batch_size=train_bs,
                                shuffle=True)

    val_dataloader = DataLoader(dataset=dataset.getValData(),
                            batch_size=val_bs,
                            shuffle=True)

    test_dataloader = DataLoader(dataset=dataset.getTestData(),
                            batch_size=test_bs,
                            shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader