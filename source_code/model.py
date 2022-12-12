from transformers import AutoTokenizer
from transformers import BertModel, GPT2Model, RobertaModel, DistilBertModel, LukeModel
from transformers import AdamW, GPT2LMHeadModel

import torch
import torch.nn as nn

# MODEL
class Bert(torch.nn.Module):
    def __init__(self):

        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.fc = nn.Linear(768, 4)
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out

class Bert_Drop(torch.nn.Module):
    def __init__(self):

        super(Bert_Drop, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 4)
            )
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out

class Luke(torch.nn.Module):
    def __init__(self):

        super(Luke, self).__init__()

        self.base = LukeModel.from_pretrained("studio-ousia/luke-base")
        self.fc = nn.Linear(768, 4)
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out

class Luke_Drop(torch.nn.Module):
    def __init__(self):

        super(Luke_Drop, self).__init__()

        self.base = LukeModel.from_pretrained("studio-ousia/luke-base")
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 4)
            )
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out

class DB2(torch.nn.Module):
    def __init__(self):

        super(DB2, self).__init__()

        self.distil = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.base1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 4),
            nn.ReLU(inplace=True)
            )
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.base2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 4),
            nn.ReLU(inplace=True)
            )
            
        self.fc = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.ReLU(True),
            nn.Linear(768, 4)
            )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output1 = self.distil(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # print(output1[1])
        # output1 = self.base1(output1[0])
        output2 = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        # output2 = self.base2(output2[0])

        logits = torch.concat((output1[0], output2[0]), dim=2)

        out = self.fc(logits)

        return out
    
class DistilBert(torch.nn.Module):
    def __init__(self):

        super(DistilBert, self).__init__()

        self.distlibert = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.fc = nn.Linear(768, 4)
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.distlibert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out

class DistilBert_Drop(torch.nn.Module):
    def __init__(self):

        super(DistilBert_Drop, self).__init__()

        self.distlibert = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 4)
            )
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.distlibert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out

class GPT2(torch.nn.Module):
    def __init__(self):

        super(GPT2, self).__init__()

        self.bert = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(768, 4)
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        
        logits = output[0]
        out = self.fc(logits)

        return out

class GPT2_Drop(torch.nn.Module):
    def __init__(self):

        super(GPT2_Drop, self).__init__()

        self.bert = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 4)
            )
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        
        logits = output[0]
        out = self.fc(logits)

        return out

class Roberta(torch.nn.Module):
    def __init__(self):

        super(Roberta, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(768, 4)
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out

class Roberta_Drop(torch.nn.Module):
    def __init__(self):

        super(Roberta_Drop, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 4)
            )
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        out = self.fc(logits)

        return out