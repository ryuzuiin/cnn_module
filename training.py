import torch
import time

class Trainer:
    def __init__(self, model, train_data, learning_rate=0.001, epochs=100):
        self.model = model
        self.train_data = train_data
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.train_losses = []

    def train(self):
        self.model.train()
        start_time = time.time()

        for epoch in range(self.epochs):
            for seq, label in self.train_data:
                self.optimizer.zero_grad()
                y_pred = self.model(seq.reshape(1, 1, -1))
                loss = self.criterion(y_pred, label)
                loss.backward()
                self.optimizer.step()

            self.train_losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch+1:03d}/{self.epochs:03d} | Loss: {loss.item():10.8f}')

        print(f'\nDuration: {time.time() - start_time: .0f} s')

    def get_train_losses(self):
        return self.train_losses
