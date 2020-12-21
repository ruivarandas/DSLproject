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
import time


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
        dir_prep = DirManagement(Path(self.configurations["proj_dir"]), self.configurations["labels"])
        train, val, test = dir_prep.create_datasets(self.configurations["test_fraction"], self.configurations["val_fraction"])
        dir_prep.write_data(train, val, test)
        data_prep = DataPreparation(dir_prep)
        self.device = data_prep.device
        self.dataloaders, self.datasets_sizes, self.class_names = data_prep.create_dataloaders(self.configurations["batch_size"],
                                                                                               self.configurations["shuffle_data"],
                                                                                               self.configurations["number_workers"])

    def _define_model(self):
        if self.configurations["model_name"] == "resnet50":
            model = models.resnet50(pretrained=True)
            n_feat = model.fc.in_features
            class_names = list(self.configurations["labels"].keys())
            model.fc = nn.Linear(n_feat, len(class_names))
            self.model = model.to(self.device)
        else:
            print("model not configured")
            raise ValueError

    @staticmethod
    def _loss():
        return nn.CrossEntropyLoss()

    def _define_learning(self):
        """
        Add differential learning rate
        :return:
        """
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.configurations["initial_learning_rate"],
                                   momentum=self.configurations["optimizer_momentum"])
        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.configurations["decay_step"],
                                                    gamma=self.configurations["lr_scheduler_gamma"])

    def _save_model(self, model):
        path = Path.cwd() / "models"
        name = f"{self.configurations['model_name']}_{time.time()}"
        trained_model_filepath = path / f"{name}.pth"
        model_config_filepath = path / f"{name}.json"
        torch.save(model.state_dict(), trained_model_filepath.as_posix())

        with open(self.config_path) as json_file:
            data = json.load(json_file)

        data["last_trained_model"] = trained_model_filepath.as_posix()

        with open(self.config_path, 'w') as outfile:
            json.dump(data, outfile)

        with open(model_config_filepath, 'w') as new_file:
            json.dump(data, new_file)

    def train_and_eval(self):
        self._prepare_data()
        self._define_model()
        self._define_learning()
        loss = self._loss()
        model = train_and_eval(self.model, loss, self.optimizer, self.exp_lr_scheduler, self.device, self.dataloaders,
                               self.datasets_sizes, self.configurations["epochs"])
        self._save_model(model)


if __name__ == '__main__':
    model_init = ECGClassifier("config.json")
    model_init.train_and_eval()

