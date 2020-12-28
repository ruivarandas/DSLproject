import torch
import time
import copy
import numpy as np
from sklearn import metrics as sk_metrics


def train_and_eval(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs, early_stop):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1score = 0.0

    losses = []
    accs = []
    f1_scores = []
    stop = False

    val_losses, val_f1_scores = [], []
    current_epoch = num_epochs
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = sk_metrics.accuracy_score(y_pred, y_true)
            f1_score = sk_metrics.f1_score(y_pred, y_true)
            print(f'{phase} Loss: {round(epoch_loss,4)} Acc: {round(epoch_acc,4)} F1Score: {round(f1_score, 4)}')

            if phase == 'train':
                losses.append(epoch_loss)
                scheduler.step()

            if phase == 'val':
                f1_scores.append(f1_score)
                accs.append(epoch_acc)

                val_losses.append(epoch_loss)
                val_f1_scores.append(f1_score)

                if epoch >= 4 and early_stop:
                    if val_losses[-1] >= np.mean(val_losses[-4:-1]) or val_f1_scores[-1] <= np.mean(val_f1_scores[-4:-1]):
                        stop = True

            # deep copy the model
            # CHECK BELLOW
            if phase == 'val' and epoch_acc > best_acc and f1_score > best_f1score:
                best_acc = epoch_acc
                best_f1score = f1_score
                best_model_wts = copy.deepcopy(model.state_dict())

        if stop:
            print(f"Stopped at epoch {epoch}.")
            current_epoch = epoch
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {round(time_elapsed // 60,0)}m {round(time_elapsed % 60, 0)}s')
    print(f'Best val Acc: {round(best_acc,4)}')
    print(f'Best val f1score: {round(best_f1score, 4)}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = {
        "loss train": losses,
        "f1_score val": f1_scores,
        "acc val": accs,
        "best val acc": round(best_acc, 4),
        "best val f1": round(best_f1score, 4)
    }
    return model, metrics, current_epoch
