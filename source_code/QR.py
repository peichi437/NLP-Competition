import torch
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from transformers import AdamW

import os
import numpy as np, pandas as pd

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../source_code"))
from utils import get_output_post_fn, acc
from dataset import qrDataset

class QR:
    def __init__(self, **args):
        super(QR, self).__init__()
        self.device = args['device']
        self.tokenizer = args['tokenizer']
        self.lr = args['lr']
        self.epochs = args['epochs']
        self.work_dir = args['work_dir']
        self.submit_csv = args['submit_csv']
        self.model_name = args['model_name']
        self.batch_size = args['bsz']

    def train_one_epoch(self, dataloader, model, opt, loss_fct, epoch):
        running_loss = 0.0
        running_len = 0
        loop = tqdm(dataloader, leave=True)
        for batch_id, batch in enumerate(loop):
            # reset
            opt.zero_grad()


            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            try: token_type_ids = batch['token_type_ids'].to(self.device)
            except: token_type_ids=None

            q_start = batch['q_start'].to(self.device)
            r_start = batch['r_start'].to(self.device)
            q_end = batch['q_end'].to(self.device)
            r_end = batch['r_end'].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)

            q_start_logits = q_start_logits.squeeze(-1).contiguous()
            r_start_logits = r_start_logits.squeeze(-1).contiguous()
            q_end_logits = q_end_logits.squeeze(-1).contiguous()
            r_end_logits = r_end_logits.squeeze(-1).contiguous()

            q_start_loss = loss_fct(q_start_logits, q_start)
            r_start_loss = loss_fct(r_start_logits, r_start)
            q_end_loss = loss_fct(q_end_logits, q_end)
            r_end_loss = loss_fct(r_end_logits, r_end)

            loss = q_start_loss + r_start_loss + q_end_loss + r_end_loss

            # calculate loss
            loss.backward()
            # update parameters
            opt.step()

            running_loss += loss.item() * q_end.shape[0]
            running_len += q_end.shape[0]

            if batch_id % 50 == 0 and batch_id != 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    batch_id + 1, batch_id, running_loss / 50))
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

        print(f"Training_loss: {running_loss/running_len}")

        return model
    
    def fit(self, train_loader, valid_loader, test_loader, test, model):
        if not os.path.exists(os.path.join(self.work_dir, 'models')): os.mkdir(os.path.join(self.work_dir, 'models'))
        # Optim
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 1e-4  # 10^-4 good at mixup paper
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0
                },
            ]
        opt = AdamW(optimizer_grouped_parameters, lr=self.lr)
        loss_fct = CrossEntropyLoss(label_smoothing=0.1)
        valid_loss_min = np.inf
        for epoch in range(1, self.epochs+1):
            model.train()
            
            model = self.train_one_epoch(
                    dataloader=train_loader,
                    model=model,
                    opt=opt,
                    loss_fct=loss_fct,
                    epoch=epoch
                    )
                
            valid_loss = self.evaluate(valid_loader, model, loss_fct=loss_fct)
            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                torch.save({"epoch":epoch, "model":model.state_dict()}, os.path.join(self.work_dir, 'models', self.model_name))
                
        # test
        q_sub_output, r_sub_output, predict_pos = self.predict(test_loader, model)

        q_sub, r_sub = get_output_post_fn(test, q_sub_output, r_sub_output)
        test['q_sub'] = q_sub
        test['r_sub'] = r_sub

        # grading
        q_acc_sum = 0
        r_acc_sum = 0
        for i in range(test.shape[0]):
            q_accuracy = acc(test.iloc[i]["q'"], test.iloc[i]['q_sub'])
            r_accuracy = acc(test.iloc[i]["r'"], test.iloc[i]['r_sub'])

            q_acc_sum += q_accuracy
            r_acc_sum += r_accuracy

        print("q accuracy: ", q_acc_sum/test.shape[0])
        print("r accuracy: ", r_acc_sum/test.shape[0])

    def evaluate(self, dataloader, model, loss_fct):
        model.eval()
        total_loss, total_len = 0, 0
        running_loss = 0.0

        with torch.no_grad():
            loop = tqdm(dataloader, leave=True)
            for batch_id, batch in enumerate(loop):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                try: token_type_ids = batch['token_type_ids'].to(self.device)
                except: token_type_ids=None


                q_start = batch['q_start'].to(self.device)
                r_start = batch['r_start'].to(self.device)
                q_end = batch['q_end'].to(self.device)
                r_end = batch['r_end'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)

                q_start_logits = q_start_logits.squeeze(-1).contiguous()
                r_start_logits = r_start_logits.squeeze(-1).contiguous()
                q_end_logits = q_end_logits.squeeze(-1).contiguous()
                r_end_logits = r_end_logits.squeeze(-1).contiguous()

                q_start_loss = loss_fct(q_start_logits, q_start)
                r_start_loss = loss_fct(r_start_logits, r_start)
                q_end_loss = loss_fct(q_end_logits, q_end)
                r_end_loss = loss_fct(r_end_logits, r_end)

                loss = q_start_loss + r_start_loss + q_end_loss + r_end_loss

                running_loss += loss.item()
                total_loss += loss.item() * q_start.shape[0]
                total_len = q_start.shape[0]

                if batch_id % 30 == 0 and batch_id != 0:
                    print('Validation Epoch {} Batch {} Loss {:.4f}'.format(
                        batch_id + 1, batch_id, running_loss / 30))
                    running_loss = 0.0
        
        return total_loss/total_len

    def predict(self, dataloader, model):
        predict_pos = []

        model.eval()

        q_sub_output, r_sub_output = [],[]

        with torch.no_grad():
            loop = tqdm(dataloader, leave=True)
            for batch_id, batch in enumerate(loop):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                try: token_type_ids = batch['token_type_ids'].to(self.device)
                except: token_type_ids=None
                

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)

                q_start_logits = q_start_logits.squeeze(-1).contiguous()
                r_start_logits = r_start_logits.squeeze(-1).contiguous()
                q_end_logits = q_end_logits.squeeze(-1).contiguous()
                r_end_logits = r_end_logits.squeeze(-1).contiguous()

                q_start_prdict = torch.argmax(q_start_logits, 1).cpu().numpy()
                r_start_prdict = torch.argmax(r_start_logits, 1).cpu().numpy()
                q_end_prdict = torch.argmax(q_end_logits, 1).cpu().numpy()
                r_end_prdict = torch.argmax(r_end_logits, 1).cpu().numpy()

                for i in range(len(input_ids)):
                    predict_pos.append((q_start_prdict[i].item(), r_start_prdict[i].item(), q_end_prdict[i].item(), r_end_prdict[i].item()))

                    q_sub = self.tokenizer.decode(input_ids[i][q_start_prdict[i]:q_end_prdict[i]+1])
                    r_sub = self.tokenizer.decode(input_ids[i][r_start_prdict[i]:r_end_prdict[i]+1])
                    
                    q_sub_output.append(q_sub)
                    r_sub_output.append(r_sub)
        return q_sub_output, r_sub_output, predict_pos
    
    def submit(self, model):
        if not os.path.exists(os.path.join(self.work_dir, 'results')): os.mkdir(os.path.join(self.work_dir, 'results'))
        
        df=pd.read_csv(os.path.join(self.work_dir, self.submit_csv), encoding = "ISO-8859-1")

        df[['q','r']] = df[['q','r']].apply(lambda x: x.str.strip('\"'))
        df['r'] = df['s'] + ':' + df['r']
        test = df

        test_data_q = df['q'].tolist()
        test_data_r = df['r'].tolist()
        test_encodings = self.tokenizer(test_data_q, test_data_r, truncation=True, padding=True)
        self.tokenizer.decode(test_encodings['input_ids'][0])

        test_dataset = qrDataset(test_encodings)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model.load_state_dict(torch.load(os.path.join(self.work_dir, 'models', self.model_name))["model"])

        q_sub_output, r_sub_output, predict_pos = self.predict(dataloader=test_loader, model=model)

        q_sub, r_sub = get_output_post_fn(test, q_sub_output, r_sub_output)

        test['q_sub'] = q_sub
        test['r_sub'] = r_sub

        test = test.drop(['q','r','s'], axis=1)

        test = test.rename(columns={'q_sub':'q'})
        test = test.rename(columns={'r_sub':'r'})

        test[['q','r']] = test[['q','r']].apply(lambda x: "\"" + x + "\"")

        test.to_csv(os.path.join(self.work_dir, "results", f"submission_{self.model_name}.csv"), index=False, header=True, encoding= 'utf-8')
