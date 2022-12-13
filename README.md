# NLP-Competition
## Requirements
torch \
pandas \
numpy \
scikit-learn \
nltk \
transformers
## Models (Put in model_name)
Bert, GPT2, Roberta, DistilBert, Luke, Bert_Drop, GPT2_Drop, Roberta_Drop, DistilBert_Drop, Luke_Drop, DB2

* DB2: DistilBert + Bert -> Fc_transfer
## Execute in your terminal
```
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
```
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
* Learning Rate=3e-5

| Model Name | Q Accuracy | R Accuracy | Validation Loss |
|------------|------------|------------|:----------------|
| Bert       | 0.59069    | 0.56575    | 7128.481        |
| Roberta    | 0.59812    | 0.58679    | 7101.377        |
| DistilBert | 0.58708    | 0.56430    | 7250.130        |
| Luke       | 0.58172    | 0.57017    | 7539.106        |
| GPT2       | 0.60258    | 0.57964    | 7627.008        |
| DB2        | 0.55620    | 0.54457    | 7237.902        |

### Dropout without Label Smoothing
* Learning_Rate=3e-5

| Model Name      | Q Accuracy | R Accuracy | Validation Loss |
|-----------------|------------|------------|:----------------|
| Bert_Drop       | 0.59089    | 0.56970    | 7120.010        |
| Roberta_Drop    | 0.60831    | 0.58707    | 7099.284        |
| Luke_Drop       | 0.56549    | 0.56495    | 7574.423        |
| DistilBert_Drop | 0.58720    | 0.55671    | 7221.264        |
| GPT2_Drop       | 0.60348    | 0.58310    | 7719.937        |

