import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset

class OrcaMath(Dataset):
    def __init__(self, parquet_file, tokenizer):
        data = pd.read_parquet(parquet_file)

        self.tokenizer = tokenizer
        questions_list = data['question'].tolist()
        answers_list = data['answer'].tolist()
        questions_tokenized = self.tokenize(questions_list)
        answers_tokenized = self.tokenize(answers_list)
        self.questions = questions_tokenized
        self.answers = answers_tokenized
        
    
    def tokenize(self, text_list):
        return self.tokenizer(text_list,
                              truncation=True,
                              padding='max_length',
                              return_tensors='pt')
    

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return (self.questions[idx], self.answers[idx])