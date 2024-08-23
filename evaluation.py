import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math

class Evaluator:
    def __init__(self, model, data_processor, test_norm, window_size):
        self.model = model
        self.data_processor = data_processor
        self.test_norm = test_norm
        self.window_size = window_size

    def predict(self):
        self.model.eval()
        preds = self.test_norm[:self.window_size].tolist()

        with torch.no_grad():
            for i in range(len(self.test_norm) - self.window_size):
                seq = torch.FloatTensor(preds[-self.window_size:])
                y_pred = self.model(seq.reshape(1, 1, -1))
                preds.append(y_pred.item())

        true_predictions = self.data_processor.inverse_transform(preds[self.window_size:])
        return true_predictions

    def calculate_metrics(self, true_values, predictions):
        mse = mean_squared_error(true_values, predictions)
        rmse = math.sqrt(mse)
        mape = mean_absolute_percentage_error(true_values, predictions) * 100
        return mse, rmse, mape

    def plot_results(self, true_values, predictions, train_losses):
        plt.figure(figsize=(12, 6))

        # Loss plot
        plt.subplot(3, 1, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')

        # Prediction vs Actual plot
        plt.subplot(3, 1, 2)
        plt.plot(true_values.index, true_values.values, label='Actual')
        plt.plot(true_values.index, predictions, label='Prediction')
        plt.legend()

        # Accuracy plot
        plt.subplot(3, 1, 3)
        plt.plot(true_values.index, np.abs(true_values.values - predictions), label='Absolute Error')
        plt.legend()

        plt.tight_layout()
        plt.show()
