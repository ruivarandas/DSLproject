import torch
import time
import copy
import numpy as np
from sklearn import metrics


def train_and_eval(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1score = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{ num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            y_pred = []
            y_true = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                for j in range(len(preds)):
                    y_pred.append(int(preds[j].item()))
                    y_true.append(int(labels.data[j].item()))

                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = metrics.accuracy_score(y_pred, y_true)
            f1_score = metrics.f1_score(y_pred, y_true)
            print(f'{phase} Loss: {round(epoch_loss,4)} Acc: {round(epoch_acc,4)} F1Score: {round(f1_score, 4)}')

            # deep copy the model
            # CHECK BELLOW
            if phase == 'val' and epoch_acc > best_acc and f1_score > best_f1score:
                best_acc = epoch_acc
                best_f1score = f1_score
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {round(time_elapsed // 60,0)}m {round(time_elapsed % 60, 0)}s')
    print(f'Best val Acc: {round(best_acc,4)}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
