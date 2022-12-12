import argparse
import torch

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .utils import *
from .QR import QR
from .dataset import qrDataset

# ARGS
def parse_arguments(arguments=None):
    """
    Parse the QR arguments
    arguments:
        arguments the arguments, optionally given as argument
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default="C:/Users/user/Downloads/Competition")
    parser.add_argument('--train_file', type=str, default="Batch_answers - train_data (no-blank).csv")
    parser.add_argument('--submit_file', type=str, default="Batch_answers - test_data(no_label).csv")
    parser.add_argument('--model_name', type=str, default="Bert")
    parser.add_argument('--tokenizer', type=str, default="bert-base-cased")
    parser.add_argument('--bsz', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=3e-4)

    return parser.parse_args(args=arguments)

# MAIN
def main(**args):
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    kwargs = argparse.Namespace(args)

    ##### HYPERPARAMS
    seed = kwargs.seed
    device = "cuda:0"
    work_dir = kwargs.work_dir
    train_csv = kwargs.train_csv
    epochs = kwargs.epochs
    lr = kwargs.lr
    batch_size = kwargs.bsz
    model_name = kwargs.model_name
    tokenizer = kwargs.tokenizer


    # Set Seed
    set_seed_1(seed=seed)
    # Set GPU / CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    args['device'] = device


    ##### LOAD DATA
    df=pd.read_csv(os.path.join(work_dir, train_csv), encoding = "ISO-8859-1")
    data = preprocessing(df=df)

    ##### SPLIT
    train, valid = train_test_split(data, test_size=0.2)
    valid, test = train_test_split(valid, test_size=0.5)

    ##### ENCODING
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    except:
        tokenizer_dict = {'Bert':'bert-base-cased', 'DistilBert':'distilbert-base-cased', 'GPT2':'gpt2', 'Roberta':'roberta-base', 'DB2':'distilbert-base-cased'}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict[kwargs.model_name])
        
    args['tokenizer'] = tokenizer

    train_data_q = train['q'].tolist()  # type: ignore
    valid_data_q = valid['q'].tolist()  # type: ignore
    test_data_q = test['q'].tolist()  # type: ignore

    train_data_r = train['r'].tolist()  # type: ignore
    valid_data_r = valid['r'].tolist()  # type: ignore
    test_data_r = test['r'].tolist()  # type: ignore

    train_encodings = tokenizer(train_data_q, train_data_r, truncation=True, padding=True)
    val_encodings = tokenizer(valid_data_q, valid_data_r, truncation=True, padding=True)
    test_encodings = tokenizer(test_data_q, test_data_r, truncation=True, padding=True)

    train_answer = train[['q_start', 'r_start',	'q_end', 'r_end']].to_dict('records')  # type: ignore
    valid_answer = valid[['q_start', 'r_start',	'q_end', 'r_end']].to_dict('records')  # type: ignore
    test_answer = test[['q_start', 'r_start',	'q_end', 'r_end']].to_dict('records')  # type: ignore

    ##### ADD TOKEN POSITION
    # Convert char_based_id to token_based_id
    # Find the corossponding token id after input being tokenized
    add_token_positions(train_encodings, train_answer)
    add_token_positions(val_encodings, valid_answer)
    add_token_positions(test_encodings, test_answer)

    ##### TO DATASET
    train_dataset = qrDataset(train_encodings)
    val_dataset = qrDataset(val_encodings)
    test_dataset = qrDataset(test_encodings)
    # Pack data into dataloader by batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    qr = QR(**args)
    # Put model on device
    model = eval(model_name)().to(device)
    qr.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        test=test,
        model=model,
    )
    qr.submit(
        model=model,
    )

if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))