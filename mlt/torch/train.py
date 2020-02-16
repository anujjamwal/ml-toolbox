from torch import nn, optim


def train(model, trainset, loss_fn=nn.CrossEntropyLoss(), optim=optim.Adam, epochs=5, logger=print):
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

        logger(f'EPOCH: {epoch+1}/{epochs} | Training Loss: {train_loss / len(trainset)}')
