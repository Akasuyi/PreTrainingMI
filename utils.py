from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

import random
import torch
from dataset import *


class PreTrainDataLoader:
    def __init__(self, all_datasets_dict, datasets_list=None):
        self.all_datasets = all_datasets_dict
        self.dataset_list = datasets_list

    def to(self, device):
        for _, dataset in self.all_datasets.items():
            for k, v in dataset.items():
                dataset[k] = v.to(device)

    def ensure_same_dim(self, data_list):
        # ensure all x in data_list have the same dim in the timestep dim
        # use smallest timesteps in all xs
        smallest_timesteps = min([x.size(-1) for x in data_list])
        for i in range(len(data_list)):
            data_list[i] = data_list[i][:, :, :smallest_timesteps]
        return data_list

    def get_one_data(self, size_per_task):
        X = []
        Y = []
        for _, dataset in self.all_datasets.items():
            x = dataset['X']
            y = dataset['y']
            selected = np.random.randint(0, x.size()[0], size=size_per_task)
            X.append(x[selected])
            Y.append(y[selected])
        X = self.ensure_same_dim(X)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        return X, Y

    def pack_all(self, batch_size=32):
        batched_data = []
        for dataset_name, dataset in self.all_datasets.items():
            x = dataset['X']
            y = dataset['y']
            assert x.size()[0] == y.size()[0]
            num_data = y.size()[0]
            num_batch = num_data // batch_size if num_data % batch_size == 0 else (num_data // batch_size) + 1
            indexes = [[i * batch_size, min((i + 1) * batch_size, num_data)] for i in range(num_batch)]
            for left, right in indexes:
                batched_data.append((dataset_name, x[left: right], y[left: right]))
        return batched_data

    def pack_data_into_dataloader(self, batch_size=32, shuffle=True):
        for i, dataset_name in enumerate(self.dataset_list):
            dataset = self.all_datasets[dataset_name]
            x = dataset['X']
            y = dataset['y']
            y_with_dataset_id = torch.zeros(y.size(0),
                                        y.size(1) + 1,
                                        dtype=torch.long, device=y.device)
            y_with_dataset_id[:, :-1] = y
            y_with_dataset_id[:, -1] = i
            dataset = TensorDataset(x, y_with_dataset_id)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            self.all_datasets[dataset_name]= data_loader

    def initialize_multi_loss_random_sample(self):
        self.dataloader_iters = {}
        for i, dataset_name in enumerate(self.dataset_list):
            dataloader = self.all_datasets[dataset_name]
            self.dataloader_iters[dataset_name] = iter(dataloader)

    def get_all_dataset_in_one_x_and_y(self, batch_size, shuffle=False):
        all_dataset_x = []
        all_dataset_y = []
        for _, dataset in self.all_datasets.items():
            x = dataset['X']
            y = dataset['y']
            all_dataset_x.append(x)
            all_dataset_y.append(y)
        all_dataset_x = self.ensure_same_dim(all_dataset_x)
        all_dataset_x = torch.cat(all_dataset_x, dim=0)
        all_dataset_y = torch.cat(all_dataset_y, dim=0)
        dataset = TensorDataset(all_dataset_x, all_dataset_y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def get_one_data_multi_loss(self):
        selected_dataset = self.dataset_list[np.random.randint(0, high=len(self.all_datasets))]
        dataloader = self.dataloader_iters[selected_dataset]
        try:
            batch = dataloader.next()
        except:
            self.dataloader_iters[selected_dataset] = iter(self.all_datasets[selected_dataset])
            dataloader = self.dataloader_iters[selected_dataset]
            batch = dataloader.next()
        return batch

    def get_dataloader_list_with_dataset_and_subject_id(self, batch_size, shuffle=False):
        all_dataset_x = []
        all_dataset_y = []
        for i, (_, dataset) in enumerate(self.all_datasets.items()):
            x = dataset['X']
            y = dataset['y']
            y_with_dataset_id = torch.zeros(y.size(0), y.size(1)+1, dtype=torch.int32)
            y_with_dataset_id[:, :-1] = y
            y_with_dataset_id[:, -1] = i
            all_dataset_x.append(x)
            all_dataset_y.append(y_with_dataset_id)
        all_dataset_x = self.ensure_same_dim(all_dataset_x)
        all_dataset_x = torch.cat(all_dataset_x, dim=0)
        all_dataset_y = torch.cat(all_dataset_y, dim=0)
        dataset = TensorDataset(all_dataset_x, all_dataset_y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader


class MaxNormDefaultConstraint(object):
    """
    Applies max L2 norm 2 to the weights until the final layer and L2 norm 0.5
    to the weights of the final layer as done in [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730

    """

    def apply(self, model):
        last_weight = None
        for name, module in list(model.named_children()):
            if hasattr(module, "weight") and (
                    not module.__class__.__name__.startswith("BatchNorm")
            ):
                module.weight.data = torch.renorm(
                    module.weight.data, 2, 0, maxnorm=2
                )
                last_weight = module.weight
        if last_weight is not None:
            last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)


def ensure_time_point(Xs, time_point):
    new_x = []
    for xs in Xs:
        tmp = []
        for x in xs:
            tmp.append(x[:, :, :time_point + 1])
        new_x.append(tmp)
    return new_x

def get_random_train_and_test_data_balanced_class(x_all_class, y_all_class, num_fine_tune_data):
    min_num_trials_per_class = min([x_all_class[i].shape[0] for i in range(len(x_all_class))])
    random_train_indexes = np.random.randint(0, high=min_num_trials_per_class,
                                             size=(num_fine_tune_data))
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for class_data, class_label in zip(x_all_class, y_all_class):
        train_x.append(class_data[random_train_indexes])
        train_y.append(class_label[random_train_indexes])

        test_mask = np.ones(class_data.shape[0], dtype=bool)
        for i in random_train_indexes:
            test_mask[i] = 0
        test_train_x = class_data[test_mask]
        test_x.append(test_train_x)
        test_y.append(class_label[test_mask])

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    return (train_x, train_y), (test_x, test_y)


def get_pre_train_data(preprocessed, base_folder, datasets_infos,
                       split_ratio, save_path=None, random_seed=666,
                       return_type='tensor', device='cpu', all_classes_in_one=False, require_subject_id=False,
                       datasets_list=None, era=True):
    if preprocessed:
        with open(save_path['pre_train_data'], 'rb') as f:
            data = pickle.load(f)
            train_data_loader = data['train']
            dev_data_loader = data['dev']
            train_data_loader.to(device)
            dev_data_loader.to(device)
            if all_classes_in_one:
                class_to_id = data['class_to_id']

    else:
        all_train_data = {}
        all_dev_data = {}
        if all_classes_in_one:
            all_classes = set()
            for dataset_name, _ in datasets_infos.items():
                dataset_folder = os.path.join(base_folder, dataset_name)
                this_dataset = globals()[dataset_name](dataset_folder)
                all_classes.update(this_dataset.classes)
            all_classes = list(all_classes)
            class_to_id = {}
            for i, c in enumerate(all_classes):
                class_to_id[c] = i
        for dataset_name, _ in datasets_infos.items():
            dataset_folder = os.path.join(base_folder, dataset_name)
            this_dataset = globals()[dataset_name](dataset_folder)
            if all_classes_in_one:
                Xs, ys = this_dataset.preprocess(sorted_class=all_classes_in_one,
                                                 sorted_class_and_id=class_to_id, return_subject=require_subject_id,
                                                 era=era)
            else:
                Xs, ys = this_dataset.preprocess(era=era)

            all_subject_x_train = []
            all_subject_y_train = []
            all_subject_x_dev = []
            all_subject_y_dev = []
            for X, y in zip(Xs, ys):  # for all subjects in a dataset
                X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=1 - split_ratio,
                                                                  shuffle=False, random_state=random_seed)
                all_subject_x_train.append(X_train)
                all_subject_y_train.append(y_train)
                all_subject_x_dev.append(X_dev)
                all_subject_y_dev.append(y_dev)

            X_train = np.concatenate(all_subject_x_train, axis=0)
            y_train = np.concatenate(all_subject_y_train, axis=0)
            X_dev = np.concatenate(all_subject_x_dev, axis=0)
            y_dev = np.concatenate(all_subject_y_dev, axis=0)

            if return_type == 'tensor':
                X_train = torch.tensor(X_train).to(device=device, dtype=torch.float)
                y_train = torch.tensor(y_train).to(device=device, dtype=torch.long)
                X_dev = torch.tensor(X_dev).to(device, dtype=torch.float)
                y_dev = torch.tensor(y_dev).to(device, dtype=torch.long)

            all_train_data[dataset_name] = {
                'X': X_train,
                'y': y_train
            }
            all_dev_data[dataset_name] = {
                'X': X_dev,
                'y': y_dev
            }
        train_data_loader = PreTrainDataLoader(all_train_data, datasets_list=datasets_list)
        dev_data_loader = PreTrainDataLoader(all_dev_data, datasets_list=datasets_list)
        if all_classes_in_one:
            data = {
                'train': train_data_loader,
                'dev': dev_data_loader,
                'class_to_id': class_to_id
            }
        else:
            data = {
                'train': train_data_loader,
                'dev': dev_data_loader
            }
        with open(save_path['pre_train_data'], 'wb') as f:
            pickle.dump(data, f)
    if all_classes_in_one:
        return train_data_loader, dev_data_loader, class_to_id
    else:
        return train_data_loader, dev_data_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
