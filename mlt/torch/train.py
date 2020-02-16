from torch import nn, optim


class TrainingResult:
    def __init__(self):
        self._recorder = Recorder()

    def recorder(self):
        return self._recorder


class Recorder(object):
    def __init__(self):
        self.checkpoints = []

    def record(self, epoch, data):
        self.checkpoints[epoch] = data

    def __len__(self):
        return len(self.checkpoints)

    def __getitem__(self, item):
        return self.checkpoints[item]


def train(model, trainset, loss_fn=nn.CrossEntropyLoss(), optim=optim.Adam, epochs=5, logger=print):
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

        state = dict(train_loss=train_loss / len(trainset), state_dict=model.state_dict())
        res.recorder().record(epoch, state)
        logger(f'EPOCH: {epoch + 1}/{epochs} | Training Loss: {state["train_loss"]}')

    return res
