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


def train_with_recorder(model: nn.Module, trainset: data.DataLoader,
                        loss_fn=nn.CrossEntropyLoss(), optim=optim.Adam, epochs=5, logger=print,
                        device=torch.device("cpu")):
    """
    The function runs a simple training on model. The training executes for the provided epochs.
    :param model: PyTorch Model to train
    :param trainset: torch.utils.data.DataLoader. Represents the training dataset.
    :param loss_fn: Loss Function.
    :param optim: Optimizer Function.
    :param epochs: Number of Epochs to run the training for.
    :param logger: function to log progress.
    :param device: torch device to run on
    :return: TrainingResult
    """
    res = TrainingResult()

    model.to(device)

    model.train()
    optimizer = optim(model.parameters())
    for epoch in range(epochs):
        train_loss = 0.0
        for data, labels in trainset:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, labels)
            train_loss += loss.detach().item() * data.size(0)

            loss.backward()
            optimizer.step()

        state = dict(epoch=epoch, train_loss=train_loss / len(trainset), state_dict=model.state_dict())
        res.recorder().record(state)
        logger(f'EPOCH: {epoch + 1}/{epochs} | Training Loss: {state["train_loss"]}')

    return res


def train_validate_with_recorder(model: nn.Module, trainset: data.DataLoader, valset: data.DataLoader,
                                 loss_fn=nn.CrossEntropyLoss(), optim=optim.Adam, epochs=5,
                                 logger=print, device=torch.device("cpu")):
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

    model.to(device)

    optimizer = optim(model.parameters())
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, labels in trainset:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, labels)
            train_loss += loss.detach().item() * data.size(0)

            loss.backward()
            optimizer.step()

        testResult = test(model, valset, loss_fn=loss_fn, device=device)

        state = dict(epoch=epoch,
                     train_loss=train_loss / len(trainset),
                     validation_loss=testResult.total_loss(),
                     validation_accuracy=testResult.accuracy(),
                     validation_result=testResult,
                     state_dict=model.state_dict())

        res.recorder().record(state)
        logger(f'EPOCH: {epoch + 1}/{epochs} | Training Loss: {state["train_loss"]} | '
               f'Validation Loss: {state["validation_loss"]} | Validation Accuracy: {state["validation_accuracy"]}')

    return res


class TestResult:
    def __init__(self):
        self.loss = 0
        self.frozen = False
        self.class_correct = list()
        self.class_count = list()

    def freeze(self):
        self.frozen = True

    def total_loss(self):
        return self.loss / np.sum(self.class_count)

    def accuracy(self):
        return 100. * np.sum(self.class_correct) / np.sum(self.class_count)

    def class_accuracy(self):
        return [(idx, 100. * correct / total) for idx, (correct, total) in
                enumerate(zip(self.class_correct, self.class_count))]

    def record(self, data, labels, output, loss):
        if self.frozen:
            raise PermissionError("Cannot modify frozen result")

        self.loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)

        correct_pred = np.squeeze(pred.eq(labels.data.view_as(pred)))

        for idx in labels:
            if len(self.class_correct) < idx + 1:
                delta = idx - len(self.class_correct) + 1
                self.class_correct += [0] * delta
                self.class_count += [0] * delta

            self.class_correct[idx] += correct_pred[idx].item()
            self.class_count[idx] += 1

    def __str__(self):
        nl = '\n'
        return f'Test Size: {np.sum(self.class_count)} \n' \
            f'Test Loss: {self.total_loss()} \n' \
            f'Test Accuracy: {self.accuracy()} \n' \
            f'Class Accuracy: \n' \
            f'{nl.join([f"{kls}: {acc}" for kls, acc in self.class_accuracy()])}'


def test_with_recorder(model: nn.Module, dataloader: data.DataLoader, loss_fn, device=torch.device("cpu")):
    model.to(device)
    model.eval()

    result = TestResult()

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            loss = loss_fn(output, labels).detach()

            result.record(data, labels, output.detach(), loss)

    result.freeze()

    return result


def train_validate(model: nn.Module, trainset: data.DataLoader, valset: data.DataLoader,
                   loss_fn=nn.CrossEntropyLoss(), optim=optim.Adam, epochs=5,
                   logger=print, device=torch.device("cpu"), path="model",
                   on_epoch_start=lambda x: x, on_epoch_end=lambda x: x):
    """
    The function runs a simple training on model. The training executes for the provided epochs.
    :param on_epoch_end:
    :param on_epoch_start:
    :param path:
    :param device:
    :param model: PyTorch Model to train
    :param trainset: torch.utils.data.DataLoader. Represents the training dataset.
    :param valset: torch.utils.data.DataLoader. Represents the validation dataset.
    :param loss_fn: Loss Function.
    :param optim: Optimizer Function.
    :param epochs: Number of Epochs to run the training for.
    :param logger: function to log progress.
    :return: TrainingResult
    """
    acc = AccuracyRecorder()

    best_loss = np.Inf
    model.to(device)
    optimizer = optim(model.parameters())

    for epoch in range(epochs):
        on_epoch_start(epoch)

        model.train()
        train_loss = 0.0
        count = 0
        for data, labels in trainset:
            count += 1
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, labels)
            train_loss += loss.detach().item()

            loss.backward()
            optimizer.step()

        model.eval()
        test_count = 0
        test_loss = 0
        with torch.no_grad():
            test_count += 1
            for data, labels in valset:
                output = model(data.to(device))
                loss = loss_fn(output, labels.to(device)).item()
                test_loss += loss

                acc.record(labels.cpu(), output.detach().cpu())

        state = dict(
            epoch=epoch,
            train_loss=train_loss / count,
            validation_loss=test_loss / test_count,
            validation_accuracy=acc.accuracy(),
            validation_result=acc.class_accuracy(),
        )

        if best_loss > state["validation_loss"]:
            best_loss = state["validation_loss"]
            logger(f"Saving model for loss {best_loss} accuracy {acc.accuracy()}")
            torch.save(model, f'{path}/model-{acc.accuracy()}.pt')

        on_epoch_end(state)
        logger(f'EPOCH: {epoch + 1}/{epochs} | Training Loss: {state["train_loss"]} | '
               f'Validation Loss: {state["validation_loss"]} | Validation Accuracy: {state["validation_accuracy"]} | '
               f'Validation Class Accuracy: {state["validation_result"]}')


class AccuracyRecorder(object):

    def __init__(self):
        self.class_correct = []
        self.class_count = []

    def accuracy(self):
        return 100. * np.sum(self.class_correct) / np.sum(self.class_count)

    def class_accuracy(self):
        return [(idx, 100. * correct / total if total > 0 else -correct) for idx, (correct, total) in
                enumerate(zip(self.class_correct, self.class_count))]

    def record(self, labels, output):
        _, pred = torch.max(output, 1)

        correct_pred = np.squeeze(pred.eq(labels.data.view_as(pred)))

        for idx in labels:
            if len(self.class_correct) < idx + 1:
                delta = idx - len(self.class_correct) + 1
                self.class_correct += [0] * delta
                self.class_count += [0] * delta

            self.class_correct[idx] += correct_pred[idx].item()
            self.class_count[idx] += 1
