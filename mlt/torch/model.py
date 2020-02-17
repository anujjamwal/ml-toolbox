import torch
import numpy as np
from torch import nn, optim
from torch.utils import data


class TrainingResult:
    def __init__(self):
        self._recorder = Recorder()

    def recorder(self):
        return self._recorder


class Recorder(object):
    def __init__(self):
        self.checkpoints = list()

    def record(self, data):
        self.checkpoints.append(data)

    def __len__(self):
        return len(self.checkpoints)

    def __getitem__(self, item):
        return self.checkpoints[item]


def train(model: nn.Module, trainset: data.DataLoader,
          loss_fn=nn.CrossEntropyLoss(), optim=optim.Adam, epochs=5, logger=print):
    """
    The function runs a simple training on model. The training executes for the provided epochs.
    :param model: PyTorch Model to train
    :param trainset: torch.utils.data.DataLoader. Represents the training dataset.
    :param loss_fn: Loss Function.
    :param optim: Optimizer Function.
    :param epochs: Number of Epochs to run the training for.
    :param logger: function to log progress.
    :return: TrainingResult
    """
    res = TrainingResult()

    model.train()
    optimizer = optim(model.parameters())
    for epoch in range(epochs):
        train_loss = 0.0
        for data, labels in trainset:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, labels)
            train_loss += loss.item() * data.size(0)

            loss.backward()
            optimizer.step()

        state = dict(epoch=epoch, train_loss=train_loss / len(trainset), state_dict=model.state_dict())
        res.recorder().record(state)
        logger(f'EPOCH: {epoch + 1}/{epochs} | Training Loss: {state["train_loss"]}')

    return res


def train_with_validation(model: nn.Module, trainset: data.DataLoader, valset: data.DataLoader,
                          loss_fn=nn.CrossEntropyLoss(), optim=optim.Adam, epochs=5,
                          logger=print):
    """
    The function runs a simple training on model. The training executes for the provided epochs.
    :param model: PyTorch Model to train
    :param trainset: torch.utils.data.DataLoader. Represents the training dataset.
    :param valset: torch.utils.data.DataLoader. Represents the validation dataset.
    :param loss_fn: Loss Function.
    :param optim: Optimizer Function.
    :param epochs: Number of Epochs to run the training for.
    :param logger: function to log progress.
    :return: TrainingResult
    """
    res = TrainingResult()

    optimizer = optim(model.parameters())
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        val_correct = 0
        for data, labels in trainset:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, labels)
            train_loss += loss.item() * data.size(0)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for data, labels in valset:
                output = model(data)
                val_loss += loss_fn(output, labels).item() * data.size(0)
                val_correct += torch.max(output, dim=1)[1].view(labels.size()).eq(labels).sum().item()

        state = dict(epoch=epoch,
                     train_loss=train_loss / len(trainset),
                     validation_loss=val_loss / len(valset),
                     validation_accuracy=val_correct / len(valset),
                     state_dict=model.state_dict())

        res.recorder().record(state)
        logger(f'EPOCH: {epoch + 1}/{epochs} | Training Loss: {state["train_loss"]} | '
               f'Validation Loss: {state["validation_loss"]} | Validation Accuracy: {state["validation_accuracy"]}')

    return res


class TestResult:
    def __init__(self, classes):
        self.classes = classes
        self.loss = 0
        self.class_correct = [0] * classes
        self.class_total = [0] * classes
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def total_loss(self):
        return self.loss / np.sum(self.class_total)

    def accuracy(self):
        return 100. * np.sum(self.class_correct) / np.sum(self.class_total)

    def class_accuracy(self):
        return [(i, 100 * self.class_correct[i] / self.class_total[i]) for i in range(self.classes)]

    def record(self, data, labels, output, loss):
        if self.frozen:
            raise PermissionError("Cannot modify frozen result")

        self.loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)

        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

        for i in range(len(labels)):
            label = labels.data[i]
            self.class_correct[label] += correct[i].item()
            self.class_total[label] += 1

    def __str__(self):
        return f'Test Loss: {self.total_loss()} \n' \
            f'Test Accuracy: {self.accuracy()}\n' \
            f'Class Accuracy: {self.class_accuracy()}'


def test(model: nn.Module, classes: int, dataloader: data.DataLoader, loss_fn):
    model.eval()

    result = TestResult(classes)

    with torch.no_grad():
        for data, labels in dataloader:
            output = model(data)
            loss = loss_fn(output, labels)

            result.record(data, labels, output, loss)

    result.freeze()

    return result
