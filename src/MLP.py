import pandas as pd
import matplotlib.pyplot as plt
from src.Layer import Layer
import numpy as np
from src.helper_functions import softmax, cross_entropy

class MLP:

    def __init__(self, nin: int, nouts: list, activation: str = 'relu'):
        sz = [nin] + nouts
        self.nn = [Layer(sz[i], sz[i+1], activation=activation) for i in range(len(nouts))]
        self.hidden_activation = activation

    def __call__(self, x):
        for layer in self.nn:
            x = layer(x)
        return x

    def __repr__(self):
        return f'NN: {[x for x in self.nn]}'

    def __parameters(self):
        return [p for x in self.nn for p in x.parameters()]

    def __backprop(self, output: np.ndarray, pred: int):
        # Compute gradient for softmax + cross-entropy
        # The gradient is: predicted_probs - one_hot_true_label
        grad = output.copy()
        grad[pred] -= 1  # This is correct IF output contains the softmax probabilities

        # Backprop through output layer
        last_layer = self.nn[-1]
        for i, neuron in enumerate(last_layer.neurons):
            delta = grad[i]
            neuron.gradb = delta
            neuron.grads = delta * neuron.inputs
            neuron.delta = delta

        # Backprop through hidden layers
        for layer_idx in range(len(self.nn) - 2, -1, -1):
            layer = self.nn[layer_idx]
            layer_n = self.nn[layer_idx + 1]
            for i, neuron in enumerate(layer.neurons):
                downstream = sum(n.weights[i] * n.delta for n in layer_n.neurons)
                neuron.delta = downstream * neuron.activation_derivative()
                neuron.grads = neuron.delta * neuron.inputs
                neuron.gradb = neuron.delta

        # Gradient ascent
        step = 0.01
        for neuron in self.__parameters():
            neuron.weights -= step * neuron.grads  # Now we subtract (gradient descent)
            neuron.bias -= step * neuron.gradb

        # Reset gradients
        for neuron in self.__parameters():
            neuron.grads = np.zeros_like(neuron.grads)
            neuron.gradb = 0.0

    def train_model(self, x_train:np.ndarray, y_train: np.ndarray, epochs: int = 100, print_report: bool = True, print_graph: bool = False):
        losses = []
        accuracies = []
        prediction_labels = pd.unique(y_train)

        for neuron in self.__parameters():
            nin = len(neuron.weights)
            limit = np.sqrt(1.0 / nin)
            neuron.weights = np.random.uniform(-limit, limit, nin).astype(np.float64)
            neuron.bias = 0.0

        for epoch in range(epochs):
            total_loss = 0
            correct = 0

            for x, y in zip(x_train, y_train):
                out = self(x)
                out = softmax(out)
                loss = cross_entropy(out[np.where(prediction_labels == y)[0][0]])
                total_loss += float(loss)

                if np.argmax(out) == np.where(prediction_labels == y):
                    correct += 1

                self.__backprop(out, int(np.where(prediction_labels == y)[0][0]))

            avg_loss = total_loss / len(x_train)
            accuracy = correct / len(x_train)
            losses.append(avg_loss)
            accuracies.append(accuracy)

            if print_report and epoch % 10 == 0:  # Print every 10 epochs
                print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        if print_graph:
            self.__draw_loss_graph(list(range(epochs)), losses, accuracies)

    @staticmethod
    def __draw_loss_graph(epochs:list, losses:list, accuracies: list):
        plt.plot(epochs, losses, label='loss')
        plt.plot(epochs, accuracies, label='accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Loss and Accuracy over Epochs during Training')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_model(self, x_test, y_test, print_report: bool = True):
        prediction_labels = pd.unique(y_test)
        predictions = []
        correct = 0
        test_loss = 0

        for x,y in zip(x_test, y_test):
            # Forward pass
            out = self(x)
            out = softmax(out)

            pred_idx = np.argmax(out)
            pred_label = prediction_labels[pred_idx]
            predictions.append(pred_label)

            if pred_label == y:
                correct += 1

            true_idx = np.where(prediction_labels == y)[0][0]
            test_loss += cross_entropy(out[true_idx])

        accuracy = correct / len(y_test)
        avg_loss = test_loss / len(y_test)

        if print_report:
            print(f'TEST SET RESULTS')
            print(f'{'='*50}')
            print(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(x_test)})")
            print(f"Test Loss: {avg_loss:.4f}")

            from collections import Counter
            true_counts = Counter(y_test)
            pred_counts = Counter(predictions)

            print("True distribution:")
            for label, count in true_counts.items():
                print(f"  {label}: {count}")

            print("\nPredicted distribution:")
            for label, count in pred_counts.items():
                print(f"  {label}: {count}")

            # Calculate per-class accuracy
            print("\nPer-class accuracy:")
            for class_name in prediction_labels:
                class_correct = sum(1 for true, pred in zip(y_test, predictions)
                                    if true == class_name and pred == class_name)
                class_total = sum(1 for true in y_test if true == class_name)
                if class_total > 0:
                    class_acc = class_correct / class_total
                    print(f"  {class_name}: {class_acc:.4f} ({class_correct}/{class_total})")
