import attr
from pathlib import Path
from data_management import DirManagement, DataPreparation
from train import train_and_eval
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import torchvision
from torchvision import models
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@attr.s(auto_attribs=True)
class ECGClassifier:
    config_path: str
    device: str = attr.ib(default="cuda", init=False)
    model: torchvision.models = attr.ib(default=None, init=False)
    optimizer: torch.optim = attr.ib(default=None, init=False)
    exp_lr_scheduler: torch.optim.lr_scheduler = attr.ib(default=None, init=False)
    dataloaders: dict = attr.ib(default=None, init=False)
    datasets_sizes: dict = attr.ib(default=None, init=False)
    class_names: list = attr.ib(default=None, init=False)

    @property
    def configurations(self):
        with open(self.config_path) as json_file:
            data = json.load(json_file)
        return data

    def _prepare_data(self):
        if not self.configurations["dirs_already_prepared"]:
            dir_prep = DirManagement(Path(self.configurations["data_dir"]), self.configurations["labels"])
            train, val, test = dir_prep.create_datasets(self.configurations["test_fraction"],
                                                        self.configurations["val_fraction"])
            dir_prep.write_data(train, val, test)
            data_prep = DataPreparation(dir_prep.data_dir)
        else:
            data_prep = DataPreparation(Path(self.configurations["data_dir"]) / "figures")

        self.device = data_prep.device
        self.dataloaders, self.datasets_sizes, self.class_names = data_prep.create_dataloaders(
            self.configurations["batch_size"],
            self.configurations["shuffle_data"],
            self.configurations["number_workers"])

    def _define_model(self):
        model = None
        if self.configurations["model_name"] == "resnet50":
            model = models.resnet50(pretrained=True)
        elif self.configurations["model_name"] == "resnet18":
            model = models.resnet18(pretrained=True)
        n_feat = model.fc.in_features
        class_names = list(self.configurations["labels"].keys())
        model.fc = nn.Linear(n_feat, len(class_names))
        self.model = model.to(self.device)

    def get_class_balance(self):
        normal, abnormal = 0, 0
        data_dir = Path(self.configurations["data_dir"]) / "raw_figures"
        for folder in data_dir.iterdir():
            for signal in folder.glob("*.txt"):
                labels = np.loadtxt(signal.as_posix(), dtype=np.object)[1:, 1]
                for label in labels:
                    if label in self.configurations["labels"]['normal']:
                        normal += 1
                    elif label in self.configurations["labels"]['abnormal']:
                        abnormal += 1
        return {"normal": normal, "abnormal": abnormal}

    def _loss(self):
        weights = self.get_class_balance()
        total = weights["normal"] + weights["abnormal"]
        weights = torch.FloatTensor([weights["normal"] / total, weights["abnormal"] / total]).to(self.device)
        return nn.CrossEntropyLoss(weight=weights)

    def _define_learning(self):
        """
        Add differential learning rate
        :return:
        """
        if self.configurations["diff_learn"]:
            learning_rate_diff = [
                {'params': self.model.layer1.parameters(), 'lr': 10e-6},
                {'params': self.model.layer2.parameters(), 'lr': 10e-4},
                {'params': self.model.layer3.parameters(), 'lr': 10e-4},
                {'params': self.model.layer4.parameters(), 'lr': 10e-2},
            ]
            self.optimizer = optim.SGD(learning_rate_diff,
                                       weight_decay=self.configurations["weight_decay"],
                                       momentum=self.configurations["optimizer_momentum"])
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       weight_decay=self.configurations["weight_decay"],
                                       momentum=self.configurations["optimizer_momentum"],
                                       lr=self.configurations["initial_learning_rate"])


        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.configurations["decay_step"],
                                                    gamma=self.configurations["lr_scheduler_gamma"])

    def _save_model(self, model, metrics, epoch):
        now = datetime.now()
        path = Path.cwd() / "models"
        name = f"{self.configurations['model_name']}_{now.strftime('d_%d_t_%H:%M')}"
        trained_model_filepath = path / f"{name}.pth"
        model_config_filepath = path / f"{name}.json"
        torch.save(model, trained_model_filepath.as_posix())

        with open(self.config_path) as json_file:
            data = json.load(json_file)

        data["last_trained_model"] = trained_model_filepath.as_posix()
        data["epochs"] = epoch

        with open(self.config_path, 'w') as outfile:
            json.dump(data, outfile)
        data["metrics"] = metrics
        with open(model_config_filepath, 'w') as new_file:
            json.dump(data, new_file)

    def train_and_eval(self):
        self._prepare_data()
        print("Folders created and data prepared")
        self._define_model()
        self._define_learning()
        loss = self._loss()
        model, metrics, epoch = train_and_eval(self.model, loss, self.optimizer, self.exp_lr_scheduler, self.device,
                                               self.dataloaders,
                                               self.datasets_sizes, self.configurations["epochs"])
        self._save_model(self.model, metrics, epoch)
        for metric in metrics:
            self.plot(metrics[metric], metric, f"{metric}_per_epoch")

    def plot(self, plottable, ylabel='', name=''):
        now = datetime.now()
        plt.clf()
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.plot(plottable)
        plt.savefig(f'plots/{name}_{now.strftime("d_%d_t_%H:%M")}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    configure_seed(42)
    test_only = False
    model_init = ECGClassifier("config.json")
    model_init.train_and_eval()
