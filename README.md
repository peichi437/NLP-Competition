# NLP-Competition

## Requirements

```other
# requirements.txt
torch
pandas
numpy
scikit-learn
nltk
transformers
```

```other
pip install -r requirements.txt
```

## Models (Put in model_name)

```other
Bert
GPT2
Roberta
DistilBert
Luke
Bert_Drop
GPT2_Drop
Roberta_Drop
DistilBert_Drop
Luke_Drop
DB2
```

- DB2: DistilBert + Bert -> Fc_transfer

## Execute in your terminal

```other
python main.py \
    --work_dir {WORK_DIR} \
    --train_csv {TRAIN_CSV} \
    --submit_csv {SUBMIT_CSV} \
    --model_name {Model} \
    --bsz 8 \
    --epochs 1 \
    --lr 3e-5 \
    --seed 123 \
```

This will save your training model and submission.

## Only Predict with the model you've trained.

```other
python main.py \
    --work_dir {WORK_DIR} \
    --train_csv {TRAIN_CSV} \
    --submit_csv {SUBMIT_CSV} \
    --model_name {Model} \
    --bsz 8 \
    --epochs 1 \
    --lr 3e-5 \
    --seed 123 \
```

## Our results (Epochs=1)

### No Dropout and Label Smoothing

- Learning Rate=3e-5

| **Model Name** | **Q Accuracy** | **R Accuracy** | **Validation Loss** |
| -------------- | -------------- | -------------- | ------------------- |
| Bert           | 0.59069        | 0.56575        | 7128.481            |
| Roberta        | 0.59812        | **0.58679**    | **7101.377**        |
| DistilBert     | 0.58708        | 0.56430        | 7250.130            |
| Luke           | 0.58172        | 0.57017        | 7539.106            |
| GPT2           | **0.60258**    | 0.57964        | 7627.008            |
| DB2            | 0.55620        | 0.54457        | 7237.902            |

- Learning Rate=1e-4

| **Model Name** | **Q Accuracy** | **R Accuracy** | **Validation Loss** |
| -------------- | -------------- | -------------- | ------------------- |
| Bert           | 0.58662        | 0.55180        | 7577.477            |
| Roberta        | 0.59577        | 0.57522        | **7475.330**        |
| DistilBert     | 0.58991        | 0.54543        | 7490.690            |
| Luke           | 0.58157        | 0.57210        | 7681.781            |
| GPT2           | **0.60630**    | **0.57989**    | 7686.711            |
| DB2            | 0.56452        | 0.54086        | 7546.682            |

### Dropout without Label Smoothing

- Learning_Rate=3e-5

| **Model Name**  | **Q Accuracy** | **R Accuracy** | **Validation Loss** |
| --------------- | -------------- | -------------- | ------------------- |
| Bert_Drop       | 0.59089        | 0.56970        | 7120.010            |
| Roberta_Drop    | **0.60831**    | **0.58707**    | **7099.284**        |
| DistilBert_Drop | 0.58720        | 0.55671        | 7221.264            |
| Luke_Drop       | 0.56549        | 0.56495        | 7574.423            |
| GPT2_Drop       | 0.60348        | 0.58310        | 7719.937            |

- Learning Rate=1e-4

| **Model Name**  | **Q Accuracy** | **R Accuracy** | **Validation Loss** |
| --------------- | -------------- | -------------- | ------------------- |
| Bert_Drop       | 0.58902        | 0.56014        | 7421.714            |
| Roberta_Drop    | 0.60145        | 0.57984        | **7409.106**        |
| DistilBert_Drop | 0.58781        | 0.52936        | 7519.611            |
| Luke_Drop       | 0.57402        | 0.56926        | 7717.999            |
| GPT2_Drop       | **0.61028**    | **0.58562**    | 7757.780            |

### Label Smoothing without Dropout

- Learning Rate=3e-5

| **Model Name**           | **Q Accuracy** | **R accuracy** | **Validation Loss** |
| ------------------------ | -------------- | -------------- | ------------------- |
| Bert-label-smth0.1       | 0.59404        | 0.56429        | 10057.734           |
| Roberta-label-smth0.1    | 0.59089        | **0.58596**    | **10051.090**       |
| DistilBert-label-smth0.1 | 0.58416        | 0.56445        | 10175.184           |
| Luke-label-smth0.1       | 0.56940        | 0.57113        | 10455.308           |
| GPT2-label-smth0.1       | **0.60293**    | 0.58145        | 10873.668           |
| DB2-label-smth0.1        | 0.56212        | 0.55518        | 10175.834           |

- Learing Rate=1e-4

| **Model Name**           | **Q Accuracy** | **R accuracy** | **Validation Loss** |
| ------------------------ | -------------- | -------------- | ------------------- |
| Bert-label-smth0.1       | 0.58081        | 0.55534        | 10435.389           |
| Roberta-label-smth0.1    | 0.60248        | **0.58627**    | **10235.635**       |
| DistilBert-label-smth0.1 | 0.57781        | 0.54664        | 10411.203           |
| Luke-label-smth0.1       | 0.56505        | 0.55947        | 10567.551           |
| GPT2-label-smth0.1       | **0.60783**    | 0.57980        | 10884.886           |
| DB2-label-smth0.1        | 0.55637        | 0.55112        | 10480.545           |

### Dropout and Label Smooth

- Learning Rate=3e-5

| **Model Name**                | **Q Accuracy** | **R Accuracy** | **Validation Loss** |
| ----------------------------- | -------------- | -------------- | ------------------- |
| Bert_Drop-label-smth0.1       | 0.58934        | 0.56481        | 10072.143           |
| Roberta_Drop-label-smth0.1    | 0.60583        | **0.58727**    | **10012.728**       |
| DistilBert_Drop-label-smth0.1 | 0.58630        | 0.55877        | 10161.239           |
| Luke_Drop-label-smth0.1       | 0.56987        | 0.56450        | 10481.756           |
| GPT2_Drop-label-smth0.1       | **0.60529**    | 0.58309        | 10943.229           |

- Learing Rate=1e-4

| **Model Name**                | **Q Accuracy** | **R Accuracy** | **Validation Loss** |
| ----------------------------- | -------------- | -------------- | ------------------- |
| Bert_Drop-label-smth0.1       | 0.58204        | 0.56536        | **10360.533**       |
| Roberta_Drop-label-smth0.1    | 0.09301        | 0.12575        | 24778.624           |
| DistilBert_Drop-label-smth0.1 | 0.59076        | 0.54352        | 10363.929           |
| Luke_Drop-label-smth0.1       | 0.55035        | 0.55812        | 10529.950           |
| GPT2_Drop-label-smth0.1       | **0.60799**    | **0.58301**    | 10932.034           |

# The best one

- For Q Accuracy
   - GPT-2 with Dropout(0.5) while Learning Rate = 1e-4
- For R Accuracy
   - RoBerta with Dropout(0.5) and Label Smoothing(0.1) while Learning Rate = 3e-5
- For Validation Loss
   - RoBerta with Dropout(0.5) while Learning Rate = 3e-5
