"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    @staticmethod
    def forward(logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:

        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return torch.nn.functional.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        # Flatten layer: input size is 3 * h * w (3 channels of size h x w)
        self.flatten = nn.Flatten()

        # Linear layer: input size is 3 * h * w, output size is num_classes
        self.fc = nn.Linear(3 * h * w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)

        # Pass through the fully connected layer
        x = self.fc(x)

        return x


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        self.flatten = nn.Flatten()

        # Set hidden size to a fixed number (e.g., 256 neurons in the hidden layer)
        hidden_size = 256  # You can adjust this value as needed

        # Hidden layer: input size is 3 * h * w, output size is hidden_size
        self.hidden = nn.Linear(3 * h * w, hidden_size)

        # Output layer: maps hidden_size to num_classes
        self.output = nn.Linear(hidden_size, num_classes)

        # Activation function for the hidden layer
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)

        # Pass through the hidden layer and apply ReLU activation
        x = self.hidden(x)
        x = self.relu(x)

        # Pass through the output layer to get logits
        x = self.output(x)
        return x

class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        self.flatten = nn.Flatten()

        # Fixed hidden layer size and number of hidden layers
        hidden_dim = 256  # Size of each hidden layer
        num_layers = 3  # Fixed number of hidden layers

        # List to hold all layers
        layers = []

        # Input size to the first hidden layer
        input_dim = 3 * h * w

        # Create hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())  # Apply ReLU activation after each hidden layer
            input_dim = hidden_dim  # After each hidden layer, the input size for the next is hidden_dim

        # Output layer: maps hidden_dim to num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)
        x = self.model(x)
        return x


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        # Flatten the input image (3 * h * w)
        self.flatten = nn.Flatten()

        # Fixed hidden layer size and number of hidden layers
        hidden_dim = 256  # Size of each hidden layer
        num_layers = 3  # Fixed number of hidden layers

        # List to hold all layers
        layers = []

        # Input size to the first hidden layer
        input_dim = 3 * h * w

        # Create residual layers
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())  # Apply ReLU activation after each layer
            input_dim = hidden_dim  # After each layer, the input size for the next is hidden_dim

        # Output layer: maps hidden_dim to num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input and pass it through the layers
        x = self.flatten(x)

        # Apply the layers sequentially and include residual connections
        residual = x
        for layer in self.model:
            x = layer(x)
            # Add the residual connection (skip connection)
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Linear):
                x = x + residual
                residual = x

        return x


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="gpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
