import argparse
import numpy as np

from models import *
from trainer import Trainer
from utils import get_pre_train_data, MaxNormDefaultConstraint, setup_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--model_type', type=int, default=0)
    parser.add_argument('--p_dataset_type', type=int, default=3)
    parser.add_argument('--preprocessed', dest='preprocessed', action='store_true')
    parser.add_argument('--no-preprocessed', dest='preprocessed', action='store_false')
    parser.set_defaults(preprocessed=True)

    opt = parser.parse_args()

    dropout_rate = opt.dropout
    batch_size = opt.batch_size
    weight_decay = opt.weight_decay
    learning_rate = opt.lr
    model_type = opt.model_type
    p_dataset_type = opt.p_dataset_type
    preprocessed = opt.preprocessed

    if p_dataset_type == 0:
        p_dataset_name = "HighGammaD"
        train_tasks_names_and_infos = {
            "HighGammaD": 4
        }
    elif p_dataset_type == 1:
        p_dataset_name = "BCIC4_2a"
        train_tasks_names_and_infos = {
            "BCIC4_2a": 4
        }
    elif p_dataset_type == 2:
        p_dataset_name = "BCIC4_2b"
        train_tasks_names_and_infos = {
            "BCIC4_2b": 2
        }
    elif p_dataset_type == 3:
        p_dataset_name = "MINE"
        train_tasks_names_and_infos = {
            "MINE": 2
        }
    else:
        raise NotImplementedError('wrong dataset code')

    if model_type == 0:
        save_path = {
            'backbone': 'saved_model/shallow_backbone.pkl',
            'optimizer': 'saved_model/optimizer.pkl',
            'model': 'saved_model/shallow_model.pkl',
            'whole_model': 'saved_model/shallow_whole_model.pkl'
        }
        model_name = 'shallow_net'
        dropout_rate = 0.1
    elif model_type == 1:
        save_path = {
            'backbone': 'saved_model/deep_backbone.pkl',
            'optimizer': 'saved_model/optimizer.pkl',
            'model': 'saved_model/deep_model.pkl',
            'whole_model': 'saved_model/deep_whole_model.pkl'
        }
        model_name = 'deep_net'
        dropout_rate = 0.5
    elif model_type == 2:
        save_path = {
            'backbone': 'saved_model/eeg_backbone.pkl',
            'optimizer': 'saved_model/optimizer.pkl',
            'model': 'saved_model/eeg_model.pkl',
            'whole_model': 'saved_model/eeg_whole_model.pkl'
        }
        model_name = 'eeg_net'
        dropout_rate = 0.3
    else:
        raise NotImplementedError('wrong model type')

    save_path['pre_train_data'] = f'preprocessed_data/{p_dataset_name}_pre_train_data.pkl'

    # pre-training for 20 times
    for i in range(3):
        random_seed = np.random.randint(0, 99999)
        setup_seed(random_seed)

        base_folder = "./downloaded_data"


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')  # used when debugging

        split_ratio = 0.9
        train_data_loader, dev_data_loader, class_to_id = get_pre_train_data(preprocessed, base_folder,
                                                                             train_tasks_names_and_infos, split_ratio,
                                                                             save_path=save_path, return_type='tensor',
                                                                             random_seed=random_seed, device=device,
                                                                             all_classes_in_one=True)

        num_classes = len(class_to_id.keys())

        in_channels = 3
        tasks_names_and_classes = train_tasks_names_and_infos
        size_after_pool = 2560
        num_epoch = 5000
        num_tasks = len(train_tasks_names_and_infos)
        test_batch_size = 32
        evaluate_epoch_ival = 3
        same_accuracy = 100

        F1 = 8
        D = 2
        F2 = F1 * D

        if model_type == 0:
            backbone = ShallowNet_modified(in_channels, num_classes, drop_prob=dropout_rate).to(device)
        elif model_type == 1:
            backbone = deep_net_modified(in_channels, num_classes, drop_prob=dropout_rate).to(device)
        elif model_type == 2:
            backbone = EEGNet_MI(in_channels, num_classes, F1, F2, D, dropout_rate).to(device)
        else:
            raise NotImplementedError('wrong model type')

        optimizer = torch.optim.Adam(params=backbone.parameters(), lr=learning_rate, weight_decay=weight_decay)

        loss_func = torch.nn.CrossEntropyLoss(reduction='sum')  # input:[N,C]
        classify_func = torch.nn.LogSoftmax(dim=-1)
        constrain = MaxNormDefaultConstraint()

        trainer = Trainer(backbone, None, optimizer, None, None, constrain, loss_func, classify_func,
                          save_path, num_epoch, evaluate_epoch_ival, base_folder, device)

        max_accuracy, max_step = trainer.pre_train(
            train_data_loader.get_all_dataset_in_one_x_and_y(batch_size, shuffle=True),
            dev_data_loader.get_all_dataset_in_one_x_and_y(test_batch_size),
            same_accuracy)

        # backup pre-train model
        trainer.load_backbone()
        if model_type == 0:
            save_path['backbone'] = f'backup_models/{p_dataset_name}_shallow_backbone' + str(random_seed) + '.pkl'
        elif model_type == 1:
            save_path['backbone'] = f'backup_models/{p_dataset_name}_deep_backbone' + str(random_seed) + '.pkl'
        elif model_type == 2:
            save_path['backbone'] = f'backup_models/{p_dataset_name}_eeg_backbone' + str(random_seed) + '.pkl'
        trainer.save_backbone()