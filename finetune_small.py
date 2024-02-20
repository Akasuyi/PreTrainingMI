import argparse

from dataset import *

from torch.utils.data import TensorDataset
from models import *
from utils import get_pre_train_data, get_random_train_and_test_data_balanced_class, MaxNormDefaultConstraint, setup_seed
from trainer import Trainer


def test_small_batch():
    if preprocessed:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            X_all, y_all = data['X_all'], data['y_all']
    else:
        dataset_name = test_dataset_name
        dataset = globals()[dataset_name](base_folder)
        X_all, y_all = dataset.preprocess(balance=True, select_three=True, era=True)
        data = {
            'X_all': X_all,
            'y_all': y_all,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    all_result = {}
    for subject_id, (X, Y) in enumerate(zip(X_all, y_all)):  # for each subject in this dataset
        for fine_tune_num in nums_fine_tune_data:
            result = []
            for t in range(num_test_times):
                if model_type == 0:
                    backbone = ShallowNet_modified(in_channels, num_classes, drop_prob=dropout_rate).to(device)
                elif model_type == 1:
                    backbone = deep_net_modified(in_channels, num_classes, drop_prob=dropout_rate).to(device)
                elif model_type == 2:
                    backbone = EEGNet_MI(in_channels, num_classes, F1, F2, D, dropout_rate).to(device)
                else:
                    raise NotImplementedError('wrong model type')

                model = FineTuneNet(backbone, list(test_task_name_and_infos.values())[0], size_after_pool).to(device)

                optimizer = torch.optim.Adam([
                    {'params': backbone.parameters(), 'lr': backbone_learning_rate},
                    {'params': model.test_head.parameters()}], lr=learning_rate, weight_decay=weight_decay)

                loss_func = torch.nn.CrossEntropyLoss(reduction='sum')  # input:[N,C]
                classify_func = torch.nn.LogSoftmax(dim=-1)
                constrain = MaxNormDefaultConstraint()

                trainer = Trainer(backbone, model, None, optimizer, None, constrain, loss_func, classify_func,
                                  save_path, num_epochs, None, base_folder, device)

                (X_train, y_train), (X_test, y_test) = get_random_train_and_test_data_balanced_class(X, Y,
                                                                                                     fine_tune_num)
                X_train = X_train[:, :, :time_point + 1]
                X_test = X_test[:, :, :time_point + 1]

                X_train_dev = X_train.copy()
                y_train_dev = y_train.copy()

                # split train data into trainset and devset, pack train, dev and test data into dataloader
                if fine_tune_num < 3:
                    test_size = 0.5
                else:
                    test_size = 0.2
                X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                          test_size=test_size)

                X_train = torch.tensor(X_train).to(device=device, dtype=torch.float)
                y_train = torch.tensor(y_train).to(device=device, dtype=torch.long)

                X_dev = torch.tensor(X_dev).to(device=device, dtype=torch.float)
                y_dev = torch.tensor(y_dev).to(device=device, dtype=torch.long)

                X_test = torch.tensor(X_test).to(device=device, dtype=torch.float)
                y_test = torch.tensor(y_test).to(device=device, dtype=torch.long)

                X_train_dev = torch.tensor(X_train_dev).to(device=device, dtype=torch.float)
                y_train_dev = torch.tensor(y_train_dev).to(device=device, dtype=torch.long)

                train_data_loader = torch.utils.data.DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size,
                                                                shuffle=True)
                dev_data_loader = torch.utils.data.DataLoader(TensorDataset(X_dev, y_dev), batch_size=128,
                                                              shuffle=False)
                test_data_loader = torch.utils.data.DataLoader(TensorDataset(X_test, y_test), batch_size=128,
                                                               shuffle=False)
                train_dev_data_loader = torch.utils.data.DataLoader(TensorDataset(X_train_dev, y_train_dev),
                                                                    batch_size=batch_size,
                                                                    shuffle=True)
                trainer.re_load_for_fine_tune()
                accuracy = trainer.train_with_self_validation(train_data_loader, dev_data_loader, test_data_loader,
                                                              train_dev_data_loader, same_accuracy, subject=subject_id)
                result.append(accuracy)
            all_result[(subject_id, fine_tune_num)] = result
    return all_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--model_type', type=int, default=0)
    parser.add_argument('--dataset', type=int, default=2)
    parser.add_argument('--p_dataset_type', type=int, default=0)
    parser.add_argument('--preprocessed', dest='preprocessed', action='store_true')
    parser.add_argument('--no-preprocessed', dest='preprocessed', action='store_false')
    parser.set_defaults(preprocessed=True)

    opt = parser.parse_args()

    dropout_rate = opt.dropout
    batch_size = opt.batch_size
    weight_decay = opt.weight_decay
    learning_rate = opt.lr
    model_type = opt.model_type
    dataset_type = opt.dataset
    p_dataset_type = opt.p_dataset_type
    preprocessed = opt.preprocessed

    if dataset_type == 0:
        test_task_name_and_infos = {
            "BCIC4_2a": 4
        }
        num_subjects = 9
    elif dataset_type == 1:
        test_task_name_and_infos = {
            "BCIC4_2b": 2
        }
        num_subjects = 9
    elif dataset_type == 2:
        test_task_name_and_infos = {
            "MINE": 2  # "BCIC4_2b": 2
        }
        num_subjects = 11
    elif dataset_type == 3:
        test_dataset_name = "HighGammaD"
        test_task_name_and_infos = {
            "HighGammaD": 4
        }
        num_subjects = 14
    else:
        raise NotImplementedError('wrong dataset code')

    if p_dataset_type == 0:
        p_dataset_name = "BCIC4_2a"
        train_tasks_names_and_infos = {
            "BCIC4_2a": 4
        }
    elif p_dataset_type == 1:
        p_dataset_name = "BCIC4_2b"
        train_tasks_names_and_infos = {
            "BCIC4_2b": 2
        }
    elif p_dataset_type == 2:
        p_dataset_name = "MINE"
        train_tasks_names_and_infos = {
            "MINE": 2
        }
    elif p_dataset_type == 3:
        p_dataset_name = "HighGammaD"
        train_tasks_names_and_infos = {
            "HighGammaD": 4
        }
    else:
        raise NotImplementedError('wrong dataset code')

    if model_type == 0:
        save_path = {
            'backbone': 'saved_model/shallow_backbone.pkl',
            'pre_train_optimizer': 'saved_model/pre_train_optimizer.pkl',
            'fine_tune_optimizer': 'saved_model/fine_tune_optimizer.pkl',
            'model': 'saved_model/shallow_model.pkl',
            'pre_train_data': 'data/pre_train_data.pkl',
            'test_data': 'data/test_data.pkl',
            'whole_model': 'saved_model/shallow_whole_model.pkl'
        }
        model_name = 'shallow_net'
        size_after_pool = 2560
    elif model_type == 1:
        save_path = {
            'backbone': 'saved_model/deep_backbone.pkl',
            'pre_train_optimizer': 'saved_model/pre_train_optimizer.pkl',
            'fine_tune_optimizer': 'saved_model/fine_tune_optimizer.pkl',
            'model': 'saved_model/deep_model.pkl',
            'pre_train_data': 'data/pre_train_data.pkl',
            'test_data': 'data/test_data.pkl',
            'whole_model': 'saved_model/deep_whole_model.pkl'
        }
        size_after_pool = 1800
        model_name = 'deep_net'
    elif model_type == 2:
        save_path = {
            'backbone': 'saved_model/eeg_backbone.pkl',
            'pre_train_optimizer': 'saved_model/pre_train_optimizer.pkl',
            'fine_tune_optimizer': 'saved_model/fine_tune_optimizer.pkl',
            'model': 'saved_model/eeg_model.pkl',
            'pre_train_data': 'data/pre_train_data.pkl',
            'test_data': 'data/test_data.pkl',
            'whole_model': 'saved_model/eeg_whole_model.pkl'
        }
        size_after_pool = 560
        model_name = 'eeg_net'
    else:
        raise NotImplementedError('wrong model type')

    split_ratio = 0.9

    random_seed = np.random.randint(0, 99999)
    setup_seed(random_seed)

    test_dataset_name = list(test_task_name_and_infos.keys())[0]
    base_folder = os.path.join("./downloaded_data", test_dataset_name)
    save_path['pre_train_data'] = f'preprocessed_data/{p_dataset_name}_pre_train_data.pkl'
    save_path['small_batch_data'] = f'preprocessed_data/{test_dataset_name}_small_batch_data.pkl'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')  # used when debugging

    file_path = save_path['small_batch_data']

    nums_fine_tune_data = [1, 3, 5, 10]

    num_test_times = 20

    in_channels = 3
    tasks_names_and_classes = train_tasks_names_and_infos

    num_steps = 50000
    num_tasks = len(train_tasks_names_and_infos)
    test_batch_size = 32
    same_accuracy = 100
    backbone_learning_rate = 0.0006
    time_point = 1125

    num_epochs = 2000

    F1 = 8
    D = 2
    F2 = F1 * D

    split_ratio = 0.9

    _, _, class_to_id = get_pre_train_data(True, base_folder,
                                         train_tasks_names_and_infos, split_ratio,
                                         save_path=save_path, return_type='tensor',
                                         random_seed=random_seed, device=device,
                                         all_classes_in_one=True)
    num_classes = len(class_to_id.keys())

    if model_type == 0:
        match_str = 'shallow'
    elif model_type == 1:
        match_str = 'deep'
    elif model_type == 2:
        match_str = 'eeg'
    else:
        pass

    backbone_path = []
    for path in os.listdir('backup_models'):
        if p_dataset_name in path:
            if path.split('_')[-2] == match_str:
                backbone_path.append(os.path.join('backup_models', path))
    backbone_path = backbone_path[1]
    save_path['backbone'] = backbone_path

    if p_dataset_type == dataset_type:
        pass
    else:
        result = test_small_batch()
        with open(f'result/{match_str}_net_small_batch_pre-trained_by_{p_dataset_name}_test_by{test_dataset_name}.pkl', 'wb') as f:
            pickle.dump(result, f)