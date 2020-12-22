from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import attr
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


@attr.s(auto_attribs=True)
class DirManagement:
    project_dir: str
    labels_dict: dict 
    
    @property
    def labels_list(self):
        return list(self.labels_dict.keys())
    
    @property
    def data_dir(self):
        return Path(self.project_dir) / "figures"
    
    @property
    def raw_data_dir(self):
        return Path(self.project_dir) / "raw_figures"
    
    @property
    def all_filenames(self):
        all_filenames = []
        for folder in self.raw_data_dir.iterdir():
            for image in folder.glob("*.png"):
                all_filenames.append(image)
        return all_filenames

    @property
    def all_labels(self):
        all_labels = {}
        for folder in self.raw_data_dir.iterdir():
            for txt in folder.glob("*.txt"):
                all_labels[folder.name] = np.loadtxt(Path(folder) / txt.name, dtype=np.object)[1:, 1]
        return all_labels
                
    def create_datasets(self, test_size, val_size):
        """
        split all filenames in train, validation and test datasets
        """
        _, test_filenames = train_test_split(self.all_filenames, test_size=test_size, random_state=42, shuffle=True)
        train_filenames, val_filenames = train_test_split(_, test_size=val_size/(1-test_size), random_state=42, shuffle=True)
        return train_filenames, val_filenames, test_filenames
    
    def _create_new_dirs(self):
        """
        create new organized directories
        """
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        Path.mkdir(self.data_dir)
        datasets = ["train", "val", "test"]
        for dataset in datasets:
            dataset_dir = self.data_dir / dataset
            Path.mkdir(dataset_dir)
            for label in self.labels_list:
                Path.mkdir(dataset_dir / label)
    
    def write_data(self, train_filenames, val_filenames, test_filenames):
        """

        copy the images from raw dir to the new directory
        """
        self._create_new_dirs()
        all_labels = self.all_labels
        for dataset in [("train", train_filenames), ("val", val_filenames), ("test", test_filenames)]:
            print(dataset[0])
            for i, filename in enumerate(dataset[1]):
                print(f"{i+1}/{len(dataset[1])}", end='\r')
                signal, segment = str(filename).split('\\')[2:]
                segment = segment.split('_')[0]
                if all_labels[signal][int(segment)] == self.labels_dict["normal"]:
                    shutil.copy(filename, self.data_dir / dataset[0] / "normal" / f"{filename.stem}.png")
                else:
                    shutil.copy(filename, self.data_dir / dataset[0] / "abnormal" / f"{filename.stem}.png")
    
    
@attr.s(auto_attribs=True)
class DataPreparation:
    data_dir: Path
    device: str = attr.ib(default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), init=False)  

    @staticmethod
    def data_transformations():
        data_transforms = {
            'train': transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
                transforms.Resize((224,224)),
        #         transforms.CenterCrop((800, 200)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
        #         transforms.CenterCrop(800),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
        #         transforms.CenterCrop(800),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        return data_transforms
    
    def create_dataloaders(self, batch_size, shuffle, num_workers):
        data_transforms = self.data_transformations()
        image_datasets = {x: datasets.ImageFolder((self.data_dir / x).as_posix(), data_transforms[x]) for x in ['train', 'val', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes
        return dataloaders, dataset_sizes, class_names
    
    @staticmethod
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    dirmanag = DirManagement(project_dir='.\Figures', labels_dict={"abnormal": [], "normal": "N"})
    dirmanag.write_data(dirmanag.all_filenames, dirmanag.all_filenames, dirmanag.all_filenames)
