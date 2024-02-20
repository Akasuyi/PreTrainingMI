import os

# pre-training with three types of dl models
for model_type in [0, 1, 2]:  # shallownet, deepnet, eegnet
    for p_ds in [0, 1, 2, 3]:  # we pre-train dl model for every dataset.
        print(f'RUNNING -model_type{model_type}-p_ds{p_ds}')
        os.system(f'python pretraining.py --model_type {model_type} --p_dataset_type {p_ds} --preprocessed {False}')
        # preprocessed should set to True if you run this code for the first time

# fine-tuning with three types of dl models
for model_type in [0, 1, 2]:  # shallownet, deepnet, eegnet
    for p_ds in [0, 1, 2, 3]:  # load model weight pre-trained by p_ds(pre-trained dataset)
        for ds in [0, 1, 2, 3]:  # fine-tuning and testing using target dataset
            print(f'RUNNING -model_type{model_type}-p_ds{p_ds}-ds{ds}')
            os.system(f'python finetune_whole.py --model_type {model_type} '
                      f'--p_dataset_type {p_ds} --dataset {ds} --preprocessed {False}')  # same as line 8
            """os.system(f'python finetune_small.py --model_type {model_type} 
            --p_dataset_type {p_ds} --dataset {ds}  --preprocessed {False}')"""
