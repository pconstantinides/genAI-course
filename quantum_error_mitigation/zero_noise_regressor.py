from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np



class ZeroNoiseRegressor(nn.Module):
    """
    Neural network regressor with 4 input neurons and 1 output neuron.
    Uses sequential hidden layers for learning complex mappings.
    """
    
    def __init__(self, hidden_sizes=[16, 8], activation=nn.LeakyReLU):
        """
        Initialize the neural network.
        
        Args:
            hidden_sizes (list): List of hidden layer sizes. Default is [16, 8].
        """
        super(ZeroNoiseRegressor, self).__init__()
        
        # Build sequential layers
        layers = []
        input_size = 5
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation())
            input_size = hidden_size

        # Output layer (single neuron)
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 5)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = self.signed_log_transform(x)
        # x = self.sqrt_transform(x)
        return self.network(x)
    
    def signed_log_transform(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    def sqrt_transform(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8)

    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: File path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_architecture': {
                'hidden_sizes': [layer.out_features for layer in self.network if isinstance(layer, nn.Linear)][:-1]
            }
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a model from disk.
        
        Args:
            path: File path to load the model from
        
        Returns:
            Loaded ZeroNoiseRegressor model
        """
        checkpoint = torch.load(path)
        hidden_sizes = checkpoint['model_architecture']['hidden_sizes']
        
        model = ZeroNoiseRegressor(hidden_sizes=hidden_sizes)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        
        return model


def train_model(
    model: ZeroNoiseRegressor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    patience: int = 10,
    min_delta: float = 1e-4,
    min_epochs: int = 5,
    max_epochs: int | None = None,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> dict:
    """
    Train the ZeroNoiseRegressor model.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        patience: Number of consecutive epochs without sufficient validation
            improvement before stopping.
        min_delta: Minimum decrease in validation loss to qualify as improvement.
        min_epochs: Minimum number of epochs before convergence checks can stop.
        max_epochs: Optional safety cap on epochs. If None, train until convergence.
        learning_rate: Learning rate for the optimizer
        verbose: Whether to print training progress
    
    Returns:
        Dictionary containing training history ('train_loss' and 'val_loss')
    """
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    def _train(epoch):
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            # X, y = X.to(device), y.to(device)
            y = y.view(-1, 1)
            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            history['train_loss'].append(loss.item())

            # if verbose and batch % 10 == 0:
                # print(f"Epoch {epoch+1}/{epochs}, Batch {batch}, Loss: {loss.item():.4f}")

    def _validate(epoch):
        model.eval()
        test_loss = 0.
        num_batches = len(val_loader)

        with torch.no_grad():
            for X, y in val_loader:
                # X, y = X.to(device), y.to(device)
                y = y.view(-1, 1)
                pred = model(X)
                test_loss += criterion(pred, y).item()

        test_loss /= num_batches
        history['val_loss'].append(test_loss)
        if verbose:
            print(f"Epoch {epoch+1}, Validation Loss: {test_loss:.4f}")
        return test_loss

    epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Convergence-based training: stop when validation loss no longer improves.
    while True:
        _train(epoch)
        current_val_loss = _validate(epoch)

        if best_val_loss - current_val_loss > min_delta:
            best_val_loss = current_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch += 1

        reached_convergence = (
            epoch >= min_epochs and epochs_without_improvement >= patience
        )
        reached_max_epochs = max_epochs is not None and epoch >= max_epochs

        if reached_convergence or reached_max_epochs:
            if verbose and reached_convergence:
                print(
                    f"Stopping at epoch {epoch}: validation loss converged "
                    f"(patience={patience}, min_delta={min_delta})."
                )
            if verbose and reached_max_epochs:
                print(f"Stopping at epoch {epoch}: reached max_epochs={max_epochs}.")
            break

    return history

def evaluate_model(
    model: ZeroNoiseRegressor,
    test_loader: torch.utils.data.DataLoader
) -> dict:
    """
    Evaluate the model on test data.
    
    Args:
        model: The trained neural network model
        test_loader: DataLoader for test data
    
    Returns:
        Dictionary containing evaluation metrics:
            - bias: bias for each evaluation case
            - variance: variance of absolute error across all evaluation cases
    """
    model.eval()
    
    metrics = {
        'bias': [],
        'abs_error': []
    }
    with torch.no_grad():
        for X, y in test_loader:
            # X, y = X.to(device), y.to(device)
            y = y.view(-1, 1)
            pred = model(X)
            bias = torch.abs(pred) - torch.abs(y)
            abs_error = torch.abs(pred - y)

            metrics['bias'].append(bias)
            metrics['abs_error'].append(abs_error)

    if len(metrics['bias']) == 0:
        raise ValueError("test_loader is empty; cannot evaluate model metrics.")

    bias_tensor = torch.cat(metrics['bias'], dim=0)
    abs_error_tensor = torch.cat(metrics['abs_error'], dim=0)

    return {
        'bias': bias_tensor,
        'variance': torch.var(abs_error_tensor, unbiased=False)
    }


def predict(model: ZeroNoiseRegressor, X: torch.Tensor) -> torch.Tensor:
    """
    Make predictions using the trained model.
    
    Args:
        model: The trained neural network model
        X: Input data of shape (n_samples, 4)
    
    Returns:
        Predictions of shape (n_samples, 1)
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    return predictions





if __name__ == "__main__":
    pass