import os
import pickle

import sklearn
import numpy as np
import scipy
import mne
from scipy.io import loadmat
import pandas as pd
import scipy.signal as signal

from _braindecode.datasets.bbci import BBCIDataset

MI_CLASSES = ['left_hand', 'right_hand', 'foot', 'tongue', 'rest']


class MIDataset:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def preprocess(self):
        raise NotImplementedError('unimplemented')

    def save(self, path):
        # save dataset use pickle
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        # load dataset use pickle
        return pickle.load(open(path, 'rb'))

    def get_epochs_raw_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels):
        raw = raw.apply_function(lambda a: a * mag_factor)

        raw = raw.apply_function(lowpass_cnt, channel_wise=False)
        if era:
            raw = raw.apply_function(exponential_moving_standardize_long, channel_wise=False)

        epochs = mne.Epochs(raw, events, event_id=events_id, tmin=ival[0], tmax=ival[1],
                            preload=True, picks=selected_channels, baseline=None, proj=False)
        return epochs

    def get_epochs_epoch_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels, low_freq,
                               high_freq, iir_params):
        epochs = mne.Epochs(raw, events, event_id=events_id, tmin=ival[0], tmax=ival[1],
                            preload=True, picks=selected_channels)  # , baseline=None, proj=False

        epochs = epochs.apply_function(lambda a: a * mag_factor)
        epochs = epochs.filter(low_freq, high_freq, iir_params=iir_params, method='iir')

        if era:
            epochs = epochs.apply_function(exponential_moving_standardize, channel_wise=False)
        return epochs

    def collect_x_y(self, xs, ys, epochs, balance):
        # collecting x and y
        if balance is False:
            for i in range(len(epochs)):
                x, y = epochs.next(return_event_id=True)
                xs.append(x)
                ys.append(y)
        else:
            for i in range(len(epochs)):
                x, y = epochs.next(return_event_id=True)
                xs[y].append(x)
                ys[y].append(y)
        return xs, ys

    def concat_x_y(self, xs, ys, balance, sorted_class, sorted_class_and_id=None):
        if balance is False:
            # concat them into an numpy array
            xs = np.stack(xs, axis=0)
            ys = np.asarray(ys)
            if sorted_class:
                ys = convert_class_index(ys, self.classes, sorted_class_and_id)
        else:
            for i in range(len(xs)):
                xs[i] = np.stack(xs[i], axis=0)
                ys[i] = np.asarray(ys[i])
            if sorted_class:
                for i in range(len(xs)):
                    ys[i] = sorted_class_and_id[self.classes[i]]
        return xs, ys

    def collect_x_y_train_test_seperated(self, epochs, xs, ys):
        for i in range(len(epochs)):
            x, y = epochs.next(return_event_id=True)
            x = np.asarray(x, dtype=np.float32)
            xs.append(x)
            ys.append(y)
        return xs, ys


def convert_class_index(y, old_class_list, new_class_and_id):
    new_y = np.zeros(y.shape, dtype=np.int32)
    for class_id, c in enumerate(old_class_list):
        new_class_id = new_class_and_id[c]
        new_y[(y == class_id)] = new_class_id
    return new_y


class BCIC4_2a(MIDataset):
    def __init__(self, data_folder, multi=True, filter=True):
        super(BCIC4_2a, self).__init__(data_folder)
        self.data_folder = data_folder
        self.train_address = []
        self.test_address = []
        self.label_address = []
        self.classes = MI_CLASSES[:4]
        for i in range(1, 10):  # 9 subjects in this dataset
            self.train_address.append(os.path.join(self.data_folder,
                                                   'A' + format(i, '02d') + 'T.gdf'))
            self.test_address.append(os.path.join(self.data_folder,
                                                  'A' + format(i, '02d') + 'E.gdf'))
            self.label_address.append(os.path.join(self.data_folder,
                                                   'A' + format(i, '02d') + 'E.mat'))

    def get_epochs_raw_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels):
        raw = raw.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
        raw = raw.apply_function(remove_nan, channel_wise=False)
        return super().get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)

    def get_epochs_epoch_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels, low_freq,
                               high_freq, iir_params):
        raw = raw.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
        raw = raw.apply_function(remove_nan, channel_wise=False)
        return super().get_epochs_epoch_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                              low_freq, high_freq, iir_params)

    def preprocess(self, balance=False, sorted_class=False, sorted_class_and_id=None, return_subject=False,
                   select_three=True, era=True):
        def preprocess_one(train_address, test_address, test_label_address, balance=False):
            if balance is False:
                xs = []
                ys = []
            else:
                xs = [[] for _ in range(num_class)]
                ys = [[] for _ in range(num_class)]

            # preprocess training data
            raw = mne.io.read_raw_gdf(train_address, stim_channel="auto")
            raw.load_data()

            events_to_class_id = {
                '769': 0,  # Left Hand
                '770': 1,  # Right Hand
                '771': 2,  # Foot
                '772': 3  # Tongue
            }

            events, events_id = mne.events_from_annotations(raw, events_to_class_id)

            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
            xs, ys = self.collect_x_y(xs, ys, epochs, balance)

            # preprocess testing data
            raw = mne.io.read_raw_gdf(test_address, stim_channel="auto", eog=['EOG-left', 'EOG-central', 'EOG-right'])
            raw.load_data()

            label = loadmat(test_label_address)['classlabel']
            events_to_class_id = {
                '783': 0,  # unknown class
            }
            events, _ = mne.events_from_annotations(raw, events_to_class_id)

            # add the class label to test data
            for i in range(events.shape[0]):
                events[i][2] = label[i] - 1

            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
            xs, ys = self.collect_x_y(xs, ys, epochs, balance)

            xs, ys = self.concat_x_y(xs, ys, balance, sorted_class, sorted_class_and_id=sorted_class_and_id)
            return xs, ys

        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]

        if select_three:
            # select only Cz C3 C4
            selected_channels = ['EEG-Cz', 'EEG-C3', 'EEG-C4']
        else:
            selected_channels = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6',
                                 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13',
                                 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']

        mag_factor = 1e6
        num_class = 4

        X_all = []
        y_all = []
        for i, (tr_a, te_a, l_a) in enumerate(zip(self.train_address, self.test_address, self.label_address)):
            X, y = preprocess_one(tr_a, te_a, l_a, balance=balance)
            X_all.append(X)
            if return_subject:
                y_with_subject_id = np.zeros((y.shape[0], 2), dtype=np.int32)
                y_with_subject_id[:, 0] = y
                y_with_subject_id[:, 1] = i
                y_all.append(y_with_subject_id)
            else:
                y_all.append(y)
        return X_all, y_all

    def get_train_test_seperated(self, select_three=False, era=True, sub_band=None):  # use when testing
        def get_one(train_address, test_address, test_label_address):
            xs_train = []
            ys_train = []
            xs_test = []
            ys_test = []
            # preprocess training data
            raw = mne.io.read_raw_gdf(train_address, stim_channel="auto")
            raw.load_data()

            events_to_class_id = {
                '769': 0,  # Left Hand
                '770': 1,  # Right Hand
                '771': 2,  # Foot
                '772': 3  # Tongue
            }

            events, events_id = mne.events_from_annotations(raw, events_to_class_id)

            epochs = self.get_epochs_epoch_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                                 low_freq, high_freq, iir_params)
            xs_train, ys_train = self.collect_x_y_train_test_seperated(epochs, xs_train, ys_train)

            # preprocess testing data
            raw = mne.io.read_raw_gdf(test_address, stim_channel="auto")
            raw.load_data()

            label = loadmat(test_label_address)['classlabel']
            events_to_class_id = {
                '783': 0,  # unknown class
            }
            events, _ = mne.events_from_annotations(raw, events_to_class_id)

            # add the class label to test data
            for i in range(events.shape[0]):
                events[i][2] = label[i] - 1

            epochs = self.get_epochs_epoch_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                                 low_freq, high_freq, iir_params)
            xs_test, ys_test = self.collect_x_y_train_test_seperated(epochs, xs_test, ys_test)

            # concat them into an numpy array
            xs_train = np.stack(xs_train, axis=0)
            ys_train = np.asarray(ys_train, dtype=np.int64)
            xs_test = np.stack(xs_test, axis=0)
            ys_test = np.asarray(ys_test, dtype=np.int64)

            return xs_train, ys_train, xs_test, ys_test

        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]

        if select_three:
            # select only Cz C3 C4
            selected_channels = ['EEG-Cz', 'EEG-C3', 'EEG-C4']
        else:
            # remove FZ will improve accuracy for FBCSP
            selected_channels = ['EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz',
                                 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14',
                                 'EEG-Pz', 'EEG-15', 'EEG-16']
        iir_params = {'order': 5, 'ftype': 'butter'}
        mag_factor = 1e6
        low_freq = 0
        high_freq = 38

        X_train_all = []
        y_train_all = []
        X_test_all = []
        y_test_all = []
        for i, (tr_a, te_a, l_a) in enumerate(zip(self.train_address, self.test_address, self.label_address)):
            X_tr, y_tr, X_te, y_te = get_one(tr_a, te_a, l_a)
            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_test_all.append(X_te)
            y_test_all.append(y_te)
        return X_train_all, y_train_all, X_test_all, y_test_all

    def get_train_test_seperated_shallow_net(self, select_three=False, era=True, sub_band=None):  # use when testing
        def get_one(train_address, test_address, test_label_address):
            xs_train = []
            ys_train = []
            xs_test = []
            ys_test = []

            # preprocess training data
            raw = mne.io.read_raw_gdf(train_address, stim_channel="auto")
            raw.load_data()

            events_to_class_id = {
                '769': 0,  # Left Hand
                '770': 1,  # Right Hand
                '771': 2,  # Foot
                '772': 3  # Tongue
            }

            events, events_id = mne.events_from_annotations(raw, events_to_class_id)

            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
            xs_train, ys_train = self.collect_x_y_train_test_seperated(epochs, xs_train, ys_train)

            # preprocess testing data
            raw = mne.io.read_raw_gdf(test_address, stim_channel="auto")
            raw.load_data()

            label = loadmat(test_label_address)['classlabel']
            events_to_class_id = {
                '783': 0,  # unknown class
            }
            events, _ = mne.events_from_annotations(raw, events_to_class_id)

            # add the class label to test data
            for i in range(events.shape[0]):
                events[i][2] = label[i] - 1

            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
            xs_test, ys_test = self.collect_x_y_train_test_seperated(epochs, xs_test, ys_test)

            # concat them into an numpy array
            xs_train = np.stack(xs_train, axis=0)
            ys_train = np.asarray(ys_train, dtype=np.int64)
            xs_test = np.stack(xs_test, axis=0)
            ys_test = np.asarray(ys_test, dtype=np.int64)

            return xs_train, ys_train, xs_test, ys_test

        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]

        if select_three:
            # select only Cz C3 C4
            selected_channels = ['EEG-Cz', 'EEG-C3', 'EEG-C4']
        else:
            selected_channels = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6',
                                 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13',
                                 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        mag_factor = 1e6

        X_train_all = []
        y_train_all = []
        X_test_all = []
        y_test_all = []
        for i, (tr_a, te_a, l_a) in enumerate(zip(self.train_address, self.test_address, self.label_address)):
            X_tr, y_tr, X_te, y_te = get_one(tr_a, te_a, l_a)
            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_test_all.append(X_te)
            y_test_all.append(y_te)
        return X_train_all, y_train_all, X_test_all, y_test_all

    def get_train_test_seperated_sub_band(self, select_three=False):
        def band_pass(X, band=None):
            sample_rate = 250
            low = (2 * band[0]) / sample_rate
            high = (2 * band[1]) / sample_rate
            b, a = signal.butter(3, [low, high], 'bandpass')
            result = signal.lfilter(b, a, X, axis=1)
            return result

        def get_bandpass_epoch(raw_gdf, events, events_id):
            all_X = []
            for band in bands:
                X = []
                Y = []
                raw_gdf = raw_gdf.apply_function(lambda a: band_pass(a, band=band), channel_wise=False)
                raw_gdf = raw_gdf.apply_function(exponential_moving_standardize, channel_wise=False)

                epochs = mne.Epochs(raw_gdf, events, event_id=events_id, tmin=ival[0], tmax=ival[1],
                                    preload=True, picks=selected_channels, baseline=None, proj=False)

                for i in range(len(epochs)):
                    x, y = epochs.next(return_event_id=True)
                    x = np.asarray(x, dtype=np.float32)
                    X.append(x)
                    Y.append(y)
                X = np.stack(X, axis=0)
                Y = np.asarray(Y, dtype=np.int64)
                all_X.append(X)

            all_X = np.stack(all_X, axis=1)
            return all_X, Y

        def remove_nan(data):
            for i_chan in range(data.shape[0]):
                # first set to nan, than replace nans by nanmean.
                this_chan = data[i_chan]
                data[i_chan] = np.where(
                    this_chan == np.min(this_chan), np.nan, this_chan
                )
                mask = np.isnan(data[i_chan])
                chan_mean = np.nanmean(data[i_chan])
                data[i_chan, mask] = chan_mean
            return data

        def get_one(train_address, test_address, test_label_address):
            xs_train = []
            ys_train = []
            xs_test = []
            ys_test = []
            # preprocess training data
            raw = mne.io.read_raw_gdf(train_address, stim_channel="auto")
            raw.load_data()

            events_to_class_id = {
                '769': 0,  # Left Hand
                '770': 1,  # Right Hand
                '771': 2,  # Foot
                '772': 3  # Tongue
            }

            events, events_id = mne.events_from_annotations(raw, events_to_class_id)

            raw = raw.apply_function(remove_nan, channel_wise=False)
            raw = raw.drop_channels(["EOG-left", "EOG-central", "EOG-right"])

            raw = raw.apply_function(lambda a: a * mag_factor)
            xs_train, ys_train = get_bandpass_epoch(raw, events, events_id)

            # preprocess testing data
            raw = mne.io.read_raw_gdf(test_address, stim_channel="auto")
            raw.load_data()
            raw = raw.apply_function(remove_nan, channel_wise=False)

            label = loadmat(test_label_address)['classlabel']
            events_to_class_id = {
                '783': 0,  # unknown class
            }
            events, _ = mne.events_from_annotations(raw, events_to_class_id)

            # add the class label to test data
            for i in range(events.shape[0]):
                events[i][2] = label[i] - 1

            raw = raw.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
            raw = raw.apply_function(lambda a: a * mag_factor)

            xs_test, ys_test = get_bandpass_epoch(raw, events, events_id)

            return xs_train, ys_train, xs_test, ys_test

        # use criteria to discards some contaminated data
        reject_criteria = dict(mag=4000e-15,  # 4000 fT
                               grad=4000e-13,  # 4000 fT/cm
                               eeg=150e-6,  # 150 µV
                               eog=250e-6)
        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]

        if select_three:
            # select only Cz C3 C4
            selected_channels = ['EEG-Cz', 'EEG-C3', 'EEG-C4']
        else:
            selected_channels = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6',
                                 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13',
                                 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        iir_params = {'order': 3, 'ftype': 'butter'}
        mag_factor = 1e6
        low_freq = 0
        high_freq = 38
        num_class = 4
        bands = [[4, 7], [8, 13], [13, 32]]

        X_train_all = []
        y_train_all = []
        X_test_all = []
        y_test_all = []
        for i, (tr_a, te_a, l_a) in enumerate(zip(self.train_address, self.test_address, self.label_address)):
            X_tr, y_tr, X_te, y_te = get_one(tr_a, te_a, l_a)
            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_test_all.append(X_te)
            y_test_all.append(y_te)
        return X_train_all, y_train_all, X_test_all, y_test_all


class BCIC4_2b(MIDataset):
    def __init__(self, data_folder, multi=True, era=True, filter=True):
        super(BCIC4_2b, self).__init__(data_folder)
        self.data_folder = data_folder
        self.train_address = []
        self.test_address = []
        self.label_address = []
        self.classes = MI_CLASSES[:2]
        for i in range(1, 10):  # 9 subjects in this dataset
            self.train_address.append([os.path.join(self.data_folder, 'B' + format(i, '02d')
                                                    + format(j, '02d') + 'T.gdf') for j in range(1, 4)])
            self.test_address.append([os.path.join(self.data_folder, 'B' + format(i, '02d')
                                                   + format(j, '02d') + 'E.gdf') for j in range(4, 6)])
            self.label_address.append([os.path.join(self.data_folder, 'B' + format(i, '02d')
                                                    + format(j, '02d') + 'E.mat') for j in range(4, 6)])

    def get_epochs_raw_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels):
        raw = raw.apply_function(remove_nan, channel_wise=False)
        return super().get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)

    def get_epochs_epoch_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels, low_freq,
                               high_freq, iir_params):
        raw = raw.apply_function(remove_nan, channel_wise=False)
        return super().get_epochs_epoch_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                              low_freq, high_freq, iir_params)

    def preprocess(self, balance=False, sorted_class=False, sorted_class_and_id=None, select_three=True,
                   return_subject=False, era=True):
        def preprocess_one(train_addresses, test_addresses, test_label_addresses, balance=False):
            if balance is False:
                xs = []
                ys = []
            else:
                xs = [[] for _ in range(num_class)]
                ys = [[] for _ in range(num_class)]

            # preprocess training data
            for train_address in train_addresses:
                raw = mne.io.read_raw_gdf(train_address, stim_channel="auto")
                events_to_class_id = {
                    '769': 0,  # Left
                    '770': 1,  # Right
                }
                raw.load_data()
                events, events_id = mne.events_from_annotations(raw, events_to_class_id)

                epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
                xs, ys = self.collect_x_y(xs, ys, epochs, balance)

            # preprocess testing data
            for test_address, test_label_address in zip(test_addresses, test_label_addresses):
                raw = mne.io.read_raw_gdf(test_address, stim_channel="auto")
                raw.load_data()
                label = loadmat(test_label_address)['classlabel']
                events_to_class_id = {
                    '783': 0,  # unknown class
                }
                events, events_id = mne.events_from_annotations(raw, events_to_class_id)

                # add the class label to test data
                for i in range(events.shape[0]):
                    events[i][2] = label[i] - 1

                # modify events id to fit modified label
                events_id = {
                    '769': 0,  # Left Hand
                    '770': 1,  # Right Hand
                }

                epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
                xs, ys = self.collect_x_y(xs, ys, epochs, balance)

            xs, ys = self.concat_x_y(xs, ys, balance, sorted_class, sorted_class_and_id=sorted_class_and_id)
            return xs, ys

        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]

        # select only Cz C3 C4
        selected_channels = ['EEG:Cz', 'EEG:C3', 'EEG:C4']
        mag_factor = 1e6
        num_class = 2

        X_all = []
        y_all = []
        for i, (tr_as, te_as, l_as) in enumerate(zip(self.train_address, self.test_address, self.label_address)):
            X, y = preprocess_one(tr_as, te_as, l_as, balance=balance)
            X_all.append(X)
            if return_subject:
                y_with_subject_id = np.zeros((y.shape[0], 2), dtype=np.int32)
                y_with_subject_id[:, 0] = y
                y_with_subject_id[:, 1] = i
                y_all.append(y_with_subject_id)
            else:
                y_all.append(y)
        return X_all, y_all

    def get_train_test_seperated(self, select_three=False, era=True):
        def get_one(train_addresses, test_addresses, test_label_addresses):
            xs_train = []
            ys_train = []
            xs_test = []
            ys_test = []
            # preprocess training data
            for train_address in train_addresses:
                raw = mne.io.read_raw_gdf(train_address, stim_channel="auto")
                raw.load_data()
                events_to_class_id = {
                    '769': 0,  # Left
                    '770': 1,  # Right
                }
                events, events_id = mne.events_from_annotations(raw, events_to_class_id)
                epochs = self.get_epochs_epoch_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                                     low_freq, high_freq, iir_params)
                xs_train, ys_train = self.collect_x_y_train_test_seperated(epochs, xs_train, ys_train)

            # preprocess testing data
            for test_address, test_label_address in zip(test_addresses, test_label_addresses):
                raw = mne.io.read_raw_gdf(test_address, stim_channel="auto")
                raw.load_data()
                label = loadmat(test_label_address)['classlabel']
                events_to_class_id = {
                    '783': 0,  # unknown class
                }
                events, events_id = mne.events_from_annotations(raw, events_to_class_id)

                # add the class label to test data
                for i in range(events.shape[0]):
                    events[i][2] = label[i] - 1

                # modify events id to fit modified label
                events_id = {
                    '769': 0,  # Left Hand
                    '770': 1,  # Right Hand
                }
                epochs = self.get_epochs_epoch_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                                     low_freq, high_freq, iir_params)
                xs_test, ys_test = self.collect_x_y_train_test_seperated(epochs, xs_test, ys_test)

            # concat them into an numpy array
            xs_train = np.stack(xs_train, axis=0)
            ys_train = np.asarray(ys_train)
            xs_test = np.stack(xs_test, axis=0)
            ys_test = np.asarray(ys_test)

            return xs_train, ys_train, xs_test, ys_test

        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]

        selected_channels = ['EEG:Cz', 'EEG:C3', 'EEG:C4']
        iir_params = {'order': 5, 'ftype': 'butter'}
        mag_factor = 1e6
        low_freq = 0
        high_freq = 38
        X_train_all = []
        y_train_all = []
        X_test_all = []
        y_test_all = []
        for i, (tr_a, te_a, l_a) in enumerate(zip(self.train_address, self.test_address, self.label_address)):
            X_tr, y_tr, X_te, y_te = get_one(tr_a, te_a, l_a)
            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_test_all.append(X_te)
            y_test_all.append(y_te)
        return X_train_all, y_train_all, X_test_all, y_test_all

    def get_train_test_seperated_shallow_net(self, select_three=False, era=True):
        def get_one(train_addresses, test_addresses, test_label_addresses):
            xs_train = []
            ys_train = []
            xs_test = []
            ys_test = []
            # preprocess training data
            for train_address in train_addresses:
                raw = mne.io.read_raw_gdf(train_address, stim_channel="auto")
                raw.load_data()
                events_to_class_id = {
                    '769': 0,  # Left
                    '770': 1,  # Right
                }
                events, events_id = mne.events_from_annotations(raw, events_to_class_id)

                epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
                xs_train, ys_train = self.collect_x_y_train_test_seperated(epochs, xs_train, ys_train)

            # preprocess testing data
            for test_address, test_label_address in zip(test_addresses, test_label_addresses):
                raw = mne.io.read_raw_gdf(test_address, stim_channel="auto")
                raw.load_data()

                label = loadmat(test_label_address)['classlabel']
                events_to_class_id = {
                    '783': 0,  # unknown class
                }
                events, events_id = mne.events_from_annotations(raw, events_to_class_id)

                # add the class label to test data
                for i in range(events.shape[0]):
                    events[i][2] = label[i] - 1

                # modify events id to fit modified label
                events_id = {
                    '769': 0,  # Left Hand
                    '770': 1,  # Right Hand
                }
                epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
                xs_test, ys_test = self.collect_x_y_train_test_seperated(epochs, xs_test, ys_test)

            # concat them into an numpy array
            xs_train = np.stack(xs_train, axis=0)
            ys_train = np.asarray(ys_train)
            xs_test = np.stack(xs_test, axis=0)
            ys_test = np.asarray(ys_test)

            return xs_train, ys_train, xs_test, ys_test

        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]

        selected_channels = ['EEG:Cz', 'EEG:C3', 'EEG:C4']
        mag_factor = 1e6
        X_train_all = []
        y_train_all = []
        X_test_all = []
        y_test_all = []
        for i, (tr_a, te_a, l_a) in enumerate(zip(self.train_address, self.test_address, self.label_address)):
            X_tr, y_tr, X_te, y_te = get_one(tr_a, te_a, l_a)
            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_test_all.append(X_te)
            y_test_all.append(y_te)
        return X_train_all, y_train_all, X_test_all, y_test_all


class HighGammaD(MIDataset):
    def __init__(self, data_folder, multi=True, era=True, filter=True):
        super(HighGammaD, self).__init__(data_folder)
        self.data_folder = data_folder
        self.train_address = []
        self.test_address = []
        self.classes = [MI_CLASSES[1], MI_CLASSES[0], MI_CLASSES[4], MI_CLASSES[2]]
        for i in range(1, 15):  # 14 subjects in this dataset
            self.train_address.append(os.path.join(data_folder, 'train/{:d}.mat'.format(i)))
            self.test_address.append(os.path.join(data_folder, 'test/{:d}.mat'.format(i)))

    def get_epochs_raw_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels,
                             re_reference_channel, down_sample_rate):
        raw = raw.set_eeg_reference(ref_channels=re_reference_channel)
        raw = raw.pick_channels(selected_channels, ordered=False)
        epochs = super().get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
        epochs = epochs.resample(down_sample_rate)
        return epochs

    def preprocess(self, balance=False, sorted_class=False, sorted_class_and_id=None, return_subject=False, era=True, select_three=False):
        def preprocess_one(train_address, test_address, balance=False):
            if balance is False:
                xs = []
                ys = []
            else:
                xs = [[] for _ in range(num_class)]
                ys = [[] for _ in range(num_class)]

            loader = BBCIDataset(train_address, load_sensor_names=load_sensor_names)
            data, events, info = loader.load()

            raw = mne.io.RawArray(data, info)
            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                               re_reference_channel, down_sample_rate)
            xs, ys = self.collect_x_y(xs, ys, epochs, balance)

            # collecting test set
            loader = BBCIDataset(test_address, load_sensor_names=load_sensor_names)
            data, events, info = loader.load()

            raw = mne.io.RawArray(data, info)
            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                               re_reference_channel, down_sample_rate)
            xs, ys = self.collect_x_y(xs, ys, epochs, balance)

            xs, ys = self.concat_x_y(xs, ys, balance, sorted_class, sorted_class_and_id=sorted_class_and_id)
            return xs, ys

        load_sensor_names = None
        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]
        down_sample_rate = 250
        # select only Cz C3 C4'
        selected_channels = ['Cz', 'C3', 'C4']
        re_reference_channel = ['M1']
        iir_params = {'order': 8, 'ftype': 'butter'}
        mag_factor = 1
        low_freq = 0
        high_freq = 38
        events_id = {'right': 0,
                     'left': 1,
                     'rest': 2,
                     'feet': 3}
        num_class = 4

        X_all = []
        y_all = []
        for i, (tr_a, te_a) in enumerate(zip(self.train_address, self.test_address)):
            X, y = preprocess_one(tr_a, te_a, balance=balance)
            X_all.append(X)
            if return_subject:
                y_with_subject_id = np.zeros((y.shape[0], 2), dtype=np.int32)
                y_with_subject_id[:, 0] = y
                y_with_subject_id[:, 1] = i
                y_all.append(y_with_subject_id)
            else:
                y_all.append(y)
        return X_all, y_all

    def get_train_test_seperated_shallow_net(self, select_three=False, era=True, sub_band=None):
        def get_one(train_address, test_address):
            xs_train = []
            ys_train = []
            xs_test = []
            ys_test = []
            loader = BBCIDataset(train_address, load_sensor_names=load_sensor_names)
            data, events, info = loader.load()

            raw = mne.io.RawArray(data, info)
            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                               re_reference_channel, down_sample_rate)
            xs_train, ys_train = self.collect_x_y_train_test_seperated(epochs, xs_train, ys_train)

            # collecting test set
            loader = BBCIDataset(test_address, load_sensor_names=load_sensor_names)
            data, events, info = loader.load()

            raw = mne.io.RawArray(data, info)
            epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                               re_reference_channel, down_sample_rate)
            xs_test, ys_test = self.collect_x_y_train_test_seperated(epochs, xs_test, ys_test)
            # concat them into an numpy array
            xs_train = np.stack(xs_train, axis=0)
            ys_train = np.asarray(ys_train, dtype=np.int64)
            xs_test = np.stack(xs_test, axis=0)
            ys_test = np.asarray(ys_test, dtype=np.int64)

            return xs_train, ys_train, xs_test, ys_test

        load_sensor_names = None
        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]
        down_sample_rate = 250
        # select only Cz C3 C4'
        selected_channels = ['Cz', 'C3', 'C4']
        re_reference_channel = ['M1']
        iir_params = {'order': 8, 'ftype': 'butter'}
        mag_factor = 1
        low_freq = 0
        high_freq = 38
        events_id = {'right': 0,
                     'left': 1,
                     'rest': 2,
                     'feet': 3}
        num_class = 4

        X_train_all = []
        y_train_all = []
        X_test_all = []
        y_test_all = []
        for i, (tr_a, te_a) in enumerate(zip(self.train_address, self.test_address)):
            X_tr, y_tr, X_te, y_te = get_one(tr_a, te_a)
            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_test_all.append(X_te)
            y_test_all.append(y_te)
        return X_train_all, y_train_all, X_test_all, y_test_all


class MINE(MIDataset):
    def __init__(self, data_folder, multi=True, filter=True):
        super(MINE, self).__init__(data_folder)
        self.data_folder = data_folder
        self.addresses = []
        self.classes = [MI_CLASSES[0], MI_CLASSES[1]]
        for path in os.listdir(self.data_folder):
            self.addresses.append([])
            for day in ['day1', 'day2', 'day3']:  #
                # self.addresses[len(self.addresses)-1].append(os.path.join(self.data_folder, path, day, '4.cnt'))
                for graz in ['graz1', 'graz2']:  #
                    for cnt in ['1.cnt', '2.cnt', '3.cnt']:
                        self.addresses[len(self.addresses) - 1].append(
                            os.path.join(self.data_folder, path, day, graz, cnt))

    def get_epochs_raw_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels):
        return super().get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)

    def get_epochs_epoch_based(self, raw, mag_factor, era, events, events_id, ival, selected_channels, low_freq,
                               high_freq, iir_params):
        return super().get_epochs_epoch_based(raw, mag_factor, era, events, events_id, ival, selected_channels,
                                              low_freq, high_freq, iir_params)

    def preprocess(self, balance=False, sorted_class=False, sorted_class_and_id=None, return_subject=False,
                   select_three=True, era=True):
        def preprocess_one(address, balance=False):
            xs = []
            ys = []

            # preprocess training data
            for a in address:
                raw = mne.io.read_raw_cnt(a)
                raw.load_data()
                raw.resample(down_sample_rate)
                raw = raw.set_eeg_reference(ref_channels=re_reference_channel)

                events, events_id = mne.events_from_annotations(raw)
                era = False
                epochs = self.get_epochs_raw_based(raw, mag_factor, era, events, events_id, ival, selected_channels)
                xs, ys = self.collect_x_y(xs, ys, epochs, balance)

            xs, ys = self.concat_x_y(xs, ys, balance, sorted_class, sorted_class_and_id=sorted_class_and_id)
            ys = ys - 1
            if balance:
                new_xs = [[] for _ in range(num_class)]
                new_ys = [[] for _ in range(num_class)]

                for class_id in range(num_class):
                    new_xs[class_id] = xs[ys == class_id]
                    new_ys[class_id] = ys[ys == class_id]
                xs = new_xs
                ys = new_ys
            return xs, ys

        # select 0.5 before stimulate and 4 after stimulate
        ival = [-0.5, 4]
        if select_three:
            # select only Cz C3 C4
            selected_channels = ['CZ', 'C3', 'C4']
        else:
            selected_channels = ['CZ', 'C3', 'C4']
        re_reference_channel = ['M1']
        mag_factor = 1e6
        num_class = 2
        down_sample_rate = 250

        X_all = []
        y_all = []
        for i, a in enumerate(self.addresses):
            X, y = preprocess_one(a, balance=balance)
            X_all.append(X)
            if return_subject:
                y_with_subject_id = np.zeros((y.shape[0], 2), dtype=np.int32)
                y_with_subject_id[:, 0] = y
                y_with_subject_id[:, 1] = i
                y_all.append(y_with_subject_id)
            else:
                y_all.append(y)
        return X_all, y_all

    def get_train_test_seperated_shallow_net(self, select_three=True, era=True):
        X_trains, X_tests, y_trains, y_tests = [], [], [], []
        X_all, y_all = self.preprocess(select_three=select_three, era=era)
        for X, y in zip(X_all, y_all):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_train)
            y_tests.append(y_test)
        return X_trains, y_trains, X_tests, y_tests

    def get_train_test_seperated(self, select_three=True, era=True):
        X_trains, X_tests, y_trains, y_tests = [], [], [], []
        X_all, y_all = self.preprocess(select_three=select_three, era=era)
        for X, y in zip(X_all, y_all):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_train)
            y_tests.append(y_test)
        return X_trains, y_trains, X_tests, y_tests

    def collect_x_y(self, xs, ys, epochs, balance):
        # collecting x and y
        for i in range(len(epochs)):
            x, y = epochs.next(return_event_id=True)
            xs.append(x)
            ys.append(y)
        return xs, ys

    def concat_x_y(self, xs, ys, balance, sorted_class, sorted_class_and_id=None):
        # concat them into an numpy array
        xs = np.stack(xs, axis=0)
        ys = np.asarray(ys)
        '''if sorted_class:
            ys = convert_class_index(ys, self.classes, sorted_class_and_id)'''
        return xs, ys


def get_events_from_single_time(event, label):
    new_event = np.zeros((event.shape[0], 3), dtype=np.int32)
    new_event[:, 0] = event
    new_event[:, 2] = label
    return new_event


def get_selected_events_from_single_time(event, label, selected_event_id):
    new_event = []
    for e, l in zip(event, label):
        if l in selected_event_id:
            new_event.append([e, 0, l])
    new_event = np.asarray(new_event, dtype=np.int32)
    new_event[:, 2] = new_event[:, 2] - min(selected_event_id)  # we assume selected_event_id is continuous
    return new_event


def exponential_moving_standardize(data, factor_new=0.001, init_block_size=1000, eps=1e-4):
    r"""Perform exponential moving standardization.
    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.
    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.
    Parameters
    ----------
    data_slice: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.
    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    import pandas as pd
    new_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        data_slice = data[i]
        data_slice = data_slice.T
        df = pd.DataFrame(data_slice)
        meaned = df.ewm(alpha=factor_new).mean()
        demeaned = df - meaned
        squared = demeaned * demeaned
        square_ewmed = squared.ewm(alpha=factor_new).mean()
        standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
        standardized = np.array(standardized)
        if init_block_size is not None:
            i_time_axis = 0
            init_mean = np.mean(
                data_slice[0:init_block_size], axis=i_time_axis, keepdims=True
            )
            init_std = np.std(
                data_slice[0:init_block_size], axis=i_time_axis, keepdims=True
            )
            init_block_standardized = (
                                              data_slice[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
            standardized[0:init_block_size] = init_block_standardized
        new_data[i] = standardized.T
    return new_data

def exponential_moving_standardize_long(
        input_data, factor_new=0.001, init_block_size=1000, eps=1e-4
):
    r"""Perform exponential moving standardization.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.


    Parameters
    ----------
    data_slice: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    data = input_data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
                                          data[0:init_block_size] - init_mean
                                  ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T


def lowpass_cnt(data, filt_order=3, axis=1):
    """
    Lowpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        channels x Time
    high_cut_hz: float
    fs: float
    filt_order: int

    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    high_cut_hz = 38
    fs = 250.0
    b, a = scipy.signal.butter(
        filt_order, high_cut_hz / (fs / 2.0), btype="lowpass"
    )
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed


def remove_nan(data):
    for i_chan in range(data.shape[0]):
        # first set to nan, than replace nans by nanmean.
        this_chan = data[i_chan]
        data[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(data[i_chan])
        chan_mean = np.nanmean(data[i_chan])
        data[i_chan, mask] = chan_mean
    return data


if __name__ == '__main__':
    '''data_folder = "downloaded_data/BCIC4_2a"
    a = BCIC4_2a(data_folder)
    X, Y = a.preprocess()
    X_mean = np.mean(np.concatenate(np.absolute(X), axis=0))
    print(X_mean)'''

    '''data_folder = "downloaded_data/BCIC4_2b"

    b = BCIC4_2b(data_folder)
    X, Y = b.preprocess()
    X_mean = np.mean(np.concatenate(np.absolute(X), axis=0))
    print(X_mean)'''

    '''data_folder = "downloaded_data/HighGammaD"
    c = HighGammaD(data_folder)
    X, Y = c.preprocess()
    X_mean = np.mean(np.concatenate(np.absolute(X), axis=0))
    print(X_mean)'''
