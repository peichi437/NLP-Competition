# NLP-Competition
# Requirements
torch \
pandas \
nltk \
transformer
# Models (Put in model_name)
Bert, GPT2, Roberta, DistilBert, Luke
# Execute in your terminal
```
python main.py \
    --work_dir {WORK_DIR} \
    --train_csv {TRAIN_CSV} \
    --submit_csv {SUBMIT_CSV} \
    --model_name Bert \
    --tokenizer bert-base-cased \
    --bsz 16 \
    --epochs 1 \
    --lr 3e-5 \
    --seed 2022 \
```
This will save your training model and submission.
# Only Predict with the model you've trained.
```
python main.py \
    --work_dir {WORK_DIR} \
    --train_csv {TRAIN_CSV} \
    --submit_csv {SUBMIT_CSV} \
    --model_name Bert \
    --tokenizer bert-base-cased \
    --bsz 16 \
    --epochs 0 \
    --lr 3e-5 \
    --seed 2022 \
```
# Our results

|  | Model_name | Q_Accuracy | R_Accuracy |
|---|------------|------------|:-----------|
|  |            |            |            |

