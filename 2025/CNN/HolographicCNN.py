import torch
import torch.nn as nn
import torch.fft

class HolographicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(HolographicConv, self).__init__()
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.cfloat))

    def forward(self, x):
        # Apply Fourier Transform to the input (holographic encoding)
        x_freq = torch.fft.fft2(x)

        # Perform element-wise complex multiplication (Holographic Convolution)
        conv_freq = x_freq * self.weights

        # Apply Inverse Fourier Transform to return to spatial domain
        output = torch.fft.ifft2(conv_freq)
        return output.real  # Extract real part as the final output

# Example Usage
x = torch.randn(1, 1, 64, 64, dtype=torch.cfloat)  # Simulated complex input
holographic_layer = HolographicConv(1, 8, 3)  # 1 input channel, 8 output channels
output = holographic_layer(x)
print(output.shape)  # Output shape should match standard CNN outputs

import torch
import torch.nn as nn
import torch.fft

class HolographicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(HolographicConv, self).__init__()
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.cfloat))

    def forward(self, x):
        # Apply Fourier Transform to the input (holographic encoding)
        x_freq = torch.fft.fft2(x)

        # Perform element-wise complex multiplication (Holographic Convolution)
        conv_freq = x_freq * self.weights

        # Apply Inverse Fourier Transform to return to spatial domain
        output = torch.fft.ifft2(conv_freq)
        return output.real  # Extract real part as the final output

# Example Usage
x = torch.randn(1, 1, 64, 64, dtype=torch.cfloat)  # Simulated complex input
holographic_layer = HolographicConv(1, 8, 3)  # 1 input channel, 8 output channels
output = holographic_layer(x)
print(output.shape)  # Output shape should match standard CNN outputs

class HolographicCNN(nn.Module):
    def __init__(self):
        super(HolographicCNN, self).__init__()
        self.conv1 = HolographicConv(1, 8, 3)
        self.conv2 = HolographicConv(8, 16, 3)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (for example, like MNIST)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Initialize and test the model
model = HolographicCNN()
sample_input = torch.randn(1, 1, 28, 28, dtype=torch.cfloat)  # Simulated holographic image
output = model(sample_input)
print(output.shape)  # Expect (1, 10) for classification
