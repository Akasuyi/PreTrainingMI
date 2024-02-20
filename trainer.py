import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, backbone, model, pre_train_optimizer, fine_tune_optimizer, lr_scheduler, constrain,
                 loss_func, classify_func, save_path, num_epochs,
                 evaluate_epoch, base_folder, device):
        self.backbone = backbone
        self.model = model
        self.pre_train_optimizer = pre_train_optimizer
        self.fine_tune_optimizer = fine_tune_optimizer
        self.lr_scheduler = lr_scheduler
        self.constrain = constrain
        self.loss_func = loss_func
        self.classify_func = classify_func
        self.save_path = save_path  # should be a dict, contain paths to save model and optimizer
        self.num_epochs = num_epochs
        self.evaluate_epoch = evaluate_epoch
        self.base_folder = base_folder
        self.device = device

    @staticmethod
    def true_or_false(pre_y, y, num_class=None):
        pre_y = torch.argmax(pre_y, dim=-1)
        true_num = torch.sum(pre_y == y)
        if num_class is not None:
            true_num_per_class = np.zeros(num_class)
            for i in range(num_class):
                mask = (y == i)
                true_num_per_class[i] = torch.sum(pre_y[mask] == y[mask])
            return true_num.item(), true_num_per_class
        else:
            return true_num.item()

    def pre_train(self, train_data, dev_data, same_accuracy):
        train_losses_list = []

        stay_the_same = 0
        max_accuracy = 0
        
        for e in tqdm(range(self.num_epochs)):
            losses = 0
            true_num = 0
            num_data = 0
            for i, batch in enumerate(train_data):
                (x, y) = batch

                tmp = self.backbone.classify(x)
                loss = self.loss_func(tmp, y)
                num_data += y.size(0)

                result = self.classify_func(tmp).detach()
                true_num += self.true_or_false(result, y)
                losses += loss.item()
                
                self.constrain.apply(self.backbone)
                self.pre_train_optimizer.zero_grad()
                loss.backward()
                self.pre_train_optimizer.step()

            average_loss = losses / num_data
            train_losses_list.append(average_loss)
            accuracy = (true_num / num_data) * 100

            if e % self.evaluate_epoch == 0:
                # evaluate result
                self.backbone.eval()
                with torch.no_grad():
                    dev_losses = 0
                    true_num = 0
                    num_data = 0
                    for i, batch in enumerate(dev_data):
                        (x, y) = batch
                        result = self.backbone.classify(x)
                        loss = self.loss_func(result, y)
                        result = self.classify_func(result)
                        # TODO: RENAME IT
                        num_data += y.size(0)
                        true_num += self.true_or_false(result, y)
                        dev_losses += loss.item()
                    average_loss = dev_losses / num_data
                    accuracy = (true_num / num_data) * 100
                self.backbone.train()
                if max_accuracy < accuracy:
                    max_accuracy = accuracy
                    max_epoch = e
                    stay_the_same = 0
                    self.save_backbone()
                else:
                    stay_the_same += 1

            if stay_the_same >= same_accuracy:
                break

        return max_accuracy, max_epoch

    def train_with_self_validation(self, train_data, dev_data, test_data, train_dev_data, same_accuracy, subject=None):
        training_dataset = train_data
        max_accuracy = 0
        stay_the_same = 0
        loaded_flag = False
        saved_epoch = 9999
        saved_train_loss = 0
        losses_record_train = []
        losses_record_dev = []
        
        for e in tqdm(range(self.num_epochs)):
            losses = 0
            num_data = 0
            for i, batch in enumerate(training_dataset):
                (x, y) = batch
                result = self.model(x)
                loss = self.loss_func(result, y)
                num_data += y.size(0)
            
                self.fine_tune_optimizer.zero_grad()
                loss.backward()
                self.fine_tune_optimizer.step()
                self.constrain.apply(self.model)
                
                losses += loss.item()

            average_loss = losses / num_data
            losses_record_train.append(average_loss)
            train_loss = average_loss

            self.model.eval()
            with torch.no_grad():
                losses = 0
                true_num = 0
                num_data = 0
                for i, batch in enumerate(dev_data):
                    (x, y) = batch
                    result = self.model(x)
                    num_data += y.size(0)
                    loss = self.loss_func(result, y)
                    result = self.classify_func(result)
                    # TODO: RENAME IT
                    true_num += self.true_or_false(result, y)
                    losses += loss.item()
                average_loss = losses / num_data
                losses_record_dev.append(average_loss)
                accuracy = (true_num / num_data) * 100
                dev_loss = average_loss
                if max_accuracy + 1e-6 < accuracy or e == 0:
                    max_accuracy = accuracy
                    stay_the_same = 0
                    self.save_whole_model()
                else:
                    stay_the_same += 1

            self.model.train()
            
            if stay_the_same == same_accuracy and loaded_flag is False:
                training_dataset = train_dev_data
                loaded_flag = True
                saved_train_loss = train_loss
                self.load_whole_model()
                saved_epoch = e
            
            if (saved_train_loss >= dev_loss and e > 50) or e >= min(self.num_epochs - 2, saved_epoch*2):
                self.model.eval()
                with torch.no_grad():
                    true_num = 0
                    num_data = 0
                    for i, batch in enumerate(test_data):
                        (x, y) = batch
                        result = self.model(x)
                        num_data += y.size(0)
                        result = self.classify_func(result)
                        # TODO: RENAME IT
                        true_num += self.true_or_false(result, y)
                    accuracy = (true_num / num_data) * 100

                self.model.train()
                return accuracy
    
    def save_backbone(self):
        torch.save(self.backbone.state_dict(), self.save_path['backbone'])
        # torch.save(self.pre_train_optimizer.state_dict(), self.save_path['pre_train_optimizer'])

    def load_backbone(self):
        self.backbone.load_state_dict(torch.load(self.save_path['backbone']))
        # self.pre_train_optimizer.load_state_dict(torch.load(self.save_path['pre_train_optimizer']))

    def re_load_for_fine_tune(self):
        self.load_backbone()
        self.model.re_initialize()

    def save_whole_model(self):
        torch.save(self.model.state_dict(), self.save_path['whole_model'])
        torch.save(self.fine_tune_optimizer.state_dict(), self.save_path['fine_tune_optimizer'])
    
    def load_whole_model(self):
        self.model.load_state_dict(torch.load(self.save_path['whole_model']))
        self.fine_tune_optimizer.load_state_dict(torch.load(self.save_path['fine_tune_optimizer']))
