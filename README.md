# NLP-Competition
## Requirements
torch \
pandas \
numpy \
scikit-learn \
nltk \
transformers
## Models (Put in model_name)
Bert, GPT2, Roberta, DistilBert, Luke
## Execute in your terminal
```
python main.py \
    --work_dir {WORK_DIR} \
    --train_csv {TRAIN_CSV} \
    --submit_csv {SUBMIT_CSV} \
    --model_name {Model} \
    --tokenizer {Tokenizer} \
    --bsz 8 \
    --epochs 1 \
    --lr 1e-4 \
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
    --tokenizer {Tokenizer} \
    --bsz 8 \
    --epochs 1 \
    --lr 1e-4 \
    --seed 123 \
```
## Our results

|  | Model_name | Q_Accuracy | R_Accuracy |
|---|------------|------------|:-----------|
|  |            |            |            |

