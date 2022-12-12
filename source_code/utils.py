from transformers import set_seed
import os, re, torch, numpy as np, pandas as pd
import nltk
nltk.download('punkt')

# SEED
def set_seed_1(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

# PREPROCESSING
def preprocessing(df):
    df = df.drop(columns=['Unnamed: 6', 'total no.: 7987'])
    df[['q','r',"q'","r'"]] = df[['q','r',"q'","r'"]].apply(lambda x: x.str.strip('\"'))
    df['sub_q_true'] = [1 if x in y else 0 for x,y in zip(df["q'"],df["q"])]
    df['sub_r_true'] = [1 if x in y else 0 for x,y in zip(df["r'"],df["r"])]
    df['sub_both'] = df['sub_q_true']*df['sub_r_true']
    
    data = df.loc[df['sub_both'] == 1]
    data['q_start'] = [y.index(x) for x,y in zip(data["q'"],data["q"])]
    data['r_start'] = [y.index(x) for x,y in zip(data["r'"],data["r"])]
    data['q_end'] = [x+len(y)-1 for x,y in zip(data["q_start"],data["q'"])]
    data['r_end'] = [x+len(y)-1 for x,y in zip(data["r_start"],data["r'"])]
    
    return data

# ADD TOKEN POSITION
def add_token_positions(encodings, answers):
    q_start, r_start, q_end, r_end = [],[],[],[]

    for i in range(len(answers)):
        q_start.append(encodings.char_to_token(i, answers[i]['q_start'], 0))
        r_start.append(encodings.char_to_token(i, answers[i]['r_start'], 1))
        q_end.append(encodings.char_to_token(i, answers[i]['q_end'], 0))
        r_end.append(encodings.char_to_token(i, answers[i]['r_end'], 1))

        if q_start[-1] is None:
            q_start[-1] = 0
            q_end[-1] = 0
            # continue

        if r_start[-1] is None:
            r_start[-1] = 0
            r_end[-1] = 0
            # continue

        shift = 1
        while q_end[-1] is None:
            q_end[-1] = encodings.char_to_token(i, answers[i]['q_end'] - shift)
            shift += 1
        shift = 1
        while r_end[-1] is None:
            r_end[-1] = encodings.char_to_token(i, answers[i]['r_end'] - shift)
            shift += 1
    encodings.update({'q_start':q_start, 'r_start':r_start,	'q_end':q_end, 'r_end':r_end})

# GET OUTPUT POST FUNC
def get_output_post_fn(test, q_sub_output, r_sub_output):
    q_sub, r_sub = [], []
    for i in range(len(test)):

        q_sub_pred = q_sub_output[i].split()
        r_sub_pred = r_sub_output[i].split()

        if q_sub_pred is None:
            q_sub_pred = []
        q_sub_error_index = q_sub_pred.index('[SEP]') if '[SEP]' in q_sub_pred else -1

        if q_sub_error_index != -1:
            q_sub_pred = q_sub_pred[:q_sub_error_index]

        temp = r_sub_pred.copy()
        if r_sub_pred is None:
            r_sub_pred = []
        else:
            for j in range(len(temp)):
                if temp[j] == '[SEP]':
                    r_sub_pred.remove('[SEP]')
                if temp[j] == '[PAD]':
                    r_sub_pred.remove('[PAD]')

        q_sub.append(' '.join(q_sub_pred))
        r_sub.append(' '.join(r_sub_pred))

    return q_sub, r_sub

# GRADING
def nltk_token_string(sentence):
    # print(sentence)
    tokens = nltk.word_tokenize(sentence)
    for i in range(len(tokens)):
        if len(tokens[i]) == 1:
            tokens[i] = re.sub(r"[!\"#$%&\'()*\+, -.\/:;<=>?@\[\\\]^_`{|}~]", '', tokens[i])
    while '' in tokens:
        tokens.remove('')
    tokens = ' '.join(tokens)
    return tokens

def lcs(X, Y):
    X_, Y_ = [], []
    
    X_ = nltk_token_string(X)
    Y_ = nltk_token_string(Y)

    m = len(X_)
    n = len(Y_)
 
    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]
 
    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X_[i-1] == Y_[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def acc(full, sub):
    common = lcs(full, sub)
    union = len(full) + len(sub) - common
    accuracy = float(common/union)

    return accuracy