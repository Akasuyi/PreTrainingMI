# PreTrainingMI
 Source code for "Cross-dataset transfer learning for motor imagery signal classification via multi-task learning and pre-training" 
 
 link:https://iopscience.iop.org/article/10.1088/1741-2552/acfe9c/meta 
 
 DOI:10.1088/1741-2552/acfe9c
 

## All files
- pretraining.py

pre-training deep learning(dl) models with all subjects' data from source dataset.

- finetune_whole.py

fine-tuning pre-trained models with a normal size of training set and testing.

- finetune_small.py

fine-tuning pre-trained models with a small batch of training data

(can be only one trial per class, but I recommend 3+ trials per class) and testing

- dataset.py

all datasets used in paper, data can be downloaded in corresponding urls

- models.py

three dl models used in the paper and some code for pre-training

- trainer.py

used to train dl models when pre-training and finetuning

- utils.py

auxiliary functions

- demo.py

small demo
