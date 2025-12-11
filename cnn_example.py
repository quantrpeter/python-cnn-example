"""
Comprehensive CNN Example - Demonstrating Key CNN Concepts
============================================================
This script demonstrates all major components of a Convolutional Neural Network:
1. Convolutional Layers
2. Pooling Layers
3. Activation Functions
4. Batch Normalization
5. Dropout
6. Fully Connected Layers
7. Different Filter Sizes
8. Feature Map Visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class ComprehensiveCNN(nn.Module):
    """
    A comprehensive CNN that demonstrates all key concepts.
    
    Architecture:
    - Conv Layer 1: Extracts low-level features (edges, textures)
    - Conv Layer 2: Extracts mid-level features
    - Conv Layer 3: Extracts high-level features
    - Pooling: Reduces spatial dimensions
    - Batch Normalization: Normalizes activations
    - Dropout: Prevents overfitting
    - Fully Connected: Final classification
    """
    
    def __init__(self, num_classes=10):
        super(ComprehensiveCNN, self).__init__()
        
        # ============================================================
        # KEY CONCEPT 1: CONVOLUTIONAL LAYERS
        # ============================================================
        # Conv layers extract features using learnable filters/kernels
        # Parameters: in_channels, out_channels, kernel_size, stride, padding
        
        # First conv layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # Second conv layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Third conv layer with different kernel size to capture different receptive fields
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        
        # ============================================================
        # KEY CONCEPT 2: BATCH NORMALIZATION
        # ============================================================
        # Normalizes the inputs of each layer, improving training stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # ============================================================
        # KEY CONCEPT 3: POOLING LAYERS
        # ============================================================
        # Max pooling reduces spatial dimensions while keeping important features
        # 2x2 pooling with stride 2 reduces size by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Alternative: Average pooling (computes average instead of max)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ============================================================
        # KEY CONCEPT 4: DROPOUT
        # ============================================================
        # Randomly drops neurons during training to prevent overfitting
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        
        # ============================================================
        # KEY CONCEPT 5: FULLY CONNECTED (DENSE) LAYERS
        # ============================================================
        # After feature extraction, FC layers perform classification
        # Input size calculation: 128 channels * 3 * 3 (after 3 pooling operations on 28x28)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Store intermediate outputs for visualization
        self.feature_maps = {}
    
    def forward(self, x):
        """
        Forward pass demonstrating all CNN operations
        """
        # ============================================================
        # CONVOLUTIONAL BLOCK 1
        # ============================================================
        # Input: (batch_size, 1, 28, 28)
        x = self.conv1(x)  # Apply convolution
        x = self.bn1(x)    # Normalize
        
        # KEY CONCEPT 6: ACTIVATION FUNCTIONS (ReLU)
        # ReLU introduces non-linearity: f(x) = max(0, x)
        x = F.relu(x)      # Apply activation
        
        # Store feature maps for visualization
        self.feature_maps['conv1'] = x.detach()
        
        x = self.pool(x)   # Reduce spatial size
        # Output: (batch_size, 32, 14, 14)
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 2
        # ============================================================
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        self.feature_maps['conv2'] = x.detach()
        x = self.pool(x)
        x = self.dropout1(x)  # Apply dropout
        # Output: (batch_size, 64, 7, 7)
        
        # ============================================================
        # CONVOLUTIONAL BLOCK 3
        # ============================================================
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        self.feature_maps['conv3'] = x.detach()
        x = self.pool(x)
        # Output: (batch_size, 128, 3, 3)
        
        # ============================================================
        # FLATTEN & FULLY CONNECTED LAYERS
        # ============================================================
        # Flatten the 3D feature maps into 1D vector
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 128*3*3)
        
        # First FC layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Second FC layer
        x = self.fc2(x)
        x = F.relu(x)
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x


def visualize_feature_maps(model, image, layer_name='conv1'):
    """
    KEY CONCEPT 7: FEATURE MAP VISUALIZATION
    Visualize what the CNN learns at different layers
    """
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
        feature_maps = model.feature_maps[layer_name].squeeze(0)
    
    # Plot first 16 feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < feature_maps.shape[0]:
            ax.imshow(feature_maps[idx].cpu().numpy(), cmap='viridis')
            ax.set_title(f'Filter {idx+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'feature_maps_{layer_name}.png')
    print(f"Feature maps saved as 'feature_maps_{layer_name}.png'")


def visualize_filters(model):
    """
    KEY CONCEPT 8: FILTER/KERNEL VISUALIZATION
    Visualize the learned filters in the first conv layer
    """
    # Get weights from first conv layer
    filters = model.conv1.weight.data.cpu().numpy()
    
    # Normalize filters for visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot first 32 filters
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Learned Filters (Conv Layer 1)', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < filters.shape[0]:
            ax.imshow(filters[idx, 0], cmap='gray')
            ax.set_title(f'Filter {idx+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('learned_filters.png')
    print("Filters saved as 'learned_filters.png'")


def demonstrate_convolution_operation():
    """
    KEY CONCEPT 9: CONVOLUTION OPERATION
    Demonstrate how convolution works with a simple example
    """
    print("\n" + "="*60)
    print("DEMONSTRATING CONVOLUTION OPERATION")
    print("="*60)
    
    # Create a simple 5x5 input
    input_data = torch.tensor([[
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5]
    ]], dtype=torch.float32).unsqueeze(0)  # Add batch and channel dims
    
    # Define different kernels to show different features
    edge_detector = torch.tensor([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32).view(1, 1, 3, 3)
    
    vertical_edge = torch.tensor([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Apply convolutions
    conv_layer = nn.Conv2d(1, 1, 3, padding=0, bias=False)
    
    # Edge detection
    conv_layer.weight.data = edge_detector
    output_edge = conv_layer(input_data)
    
    # Vertical edge detection
    conv_layer.weight.data = vertical_edge
    output_vertical = conv_layer(input_data)
    
    print("\nOriginal Input:")
    print(input_data.squeeze().numpy())
    print("\nEdge Detection Output:")
    print(output_edge.squeeze().detach().numpy())
    print("\nVertical Edge Detection Output:")
    print(output_vertical.squeeze().detach().numpy())
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_data.squeeze().numpy(), cmap='gray')
    axes[0].set_title('Original Input')
    axes[1].imshow(output_edge.squeeze().detach().numpy(), cmap='gray')
    axes[1].set_title('Edge Detection')
    axes[2].imshow(output_vertical.squeeze().detach().numpy(), cmap='gray')
    axes[2].set_title('Vertical Edge Detection')
    plt.tight_layout()
    plt.savefig('convolution_demo.png')
    print("\nConvolution demonstration saved as 'convolution_demo.png'")
    exit(1)

def demonstrate_pooling():
    """
    KEY CONCEPT 10: POOLING OPERATIONS
    Demonstrate max pooling vs average pooling
    """
    print("\n" + "="*60)
    print("DEMONSTRATING POOLING OPERATIONS")
    print("="*60)
    
    # Create a simple input
    input_data = torch.tensor([[
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 8, 3, 1],
        [2, 1, 4, 5]
    ]], dtype=torch.float32).unsqueeze(0)
    
    # Max pooling
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    max_output = max_pool(input_data)
    
    # Average pooling
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_output = avg_pool(input_data)
    
    print("\nOriginal Input (4x4):")
    print(input_data.squeeze().numpy())
    print("\nMax Pooling Output (2x2):")
    print(max_output.squeeze().numpy())
    print("\nAverage Pooling Output (2x2):")
    print(avg_output.squeeze().numpy())
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(input_data.squeeze().numpy(), cmap='viridis')
    axes[0].set_title('Original Input (4x4)')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(max_output.squeeze().numpy(), cmap='viridis')
    axes[1].set_title('Max Pooling (2x2)')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(avg_output.squeeze().numpy(), cmap='viridis')
    axes[2].set_title('Average Pooling (2x2)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('pooling_demo.png')
    print("\nPooling demonstration saved as 'pooling_demo.png'")


def train_model(model, train_loader, test_loader, epochs=5):
    """
    KEY CONCEPT 11: TRAINING PROCESS
    Demonstrates the complete training loop
    """
    print("\n" + "="*60)
    print("TRAINING THE CNN MODEL")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = model.to(device)
    
    # KEY CONCEPT 12: LOSS FUNCTION
    # CrossEntropyLoss combines LogSoftmax and NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # KEY CONCEPT 13: OPTIMIZER
    # Adam optimizer adapts learning rate for each parameter
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # KEY CONCEPT 14: LEARNING RATE SCHEDULER
    # Reduces learning rate when learning plateaus
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass (compute gradients)
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Testing phase
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'\nEpoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n')
    
    # Plot training curves
    plot_training_curves(train_losses, test_accuracies)
    
    return model


def evaluate_model(model, test_loader, device):
    """
    KEY CONCEPT 15: MODEL EVALUATION
    Evaluate model performance on test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def plot_training_curves(train_losses, test_accuracies):
    """
    Visualize training progress
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.grid(True)
    
    ax2.plot(test_accuracies, marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy over Epochs')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved as 'training_curves.png'")


def visualize_predictions(model, test_loader, device, num_images=10):
    """
    KEY CONCEPT 16: PREDICTION VISUALIZATION
    Show model predictions on test images
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images], labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = outputs.max(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Model Predictions on Test Images', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx].squeeze().cpu().numpy(), cmap='gray')
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Predictions saved as 'predictions.png'")


def main():
    """
    Main function to run the complete CNN demonstration
    """
    print("="*60)
    print("COMPREHENSIVE CNN EXAMPLE")
    print("Demonstrating All Key Concepts in Convolutional Neural Networks")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # ============================================================
    # KEY CONCEPT 17: DATA PREPROCESSING & AUGMENTATION
    # ============================================================
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std
    ])
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # KEY CONCEPT 18: BATCH PROCESSING
    # DataLoader handles batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Demonstrate basic CNN concepts
    demonstrate_convolution_operation()
    demonstrate_pooling()
    
    # Create and display model architecture
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model = ComprehensiveCNN(num_classes=10)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the model
    trained_model = train_model(model, train_loader, test_loader, epochs=5)
    
    # Visualize learned features
    print("\n" + "="*60)
    print("VISUALIZING LEARNED FEATURES")
    print("="*60)
    
    # Get a sample image
    sample_image, _ = train_dataset[0]
    
    # Visualize feature maps at different layers
    visualize_feature_maps(trained_model, sample_image, 'conv1')
    visualize_feature_maps(trained_model, sample_image, 'conv2')
    visualize_feature_maps(trained_model, sample_image, 'conv3')
    
    # Visualize learned filters
    visualize_filters(trained_model)
    
    # Visualize predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_predictions(trained_model, test_loader, device)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'cnn_model.pth')
    print("\n" + "="*60)
    print("Model saved as 'cnn_model.pth'")
    print("All visualizations have been saved as PNG files")
    print("="*60)
    
    print("\n✓ CNN demonstration complete!")
    print("\nKey concepts demonstrated:")
    print("  1. Convolutional Layers")
    print("  2. Batch Normalization")
    print("  3. Pooling Layers (Max & Average)")
    print("  4. Dropout Regularization")
    print("  5. Fully Connected Layers")
    print("  6. Activation Functions (ReLU)")
    print("  7. Feature Map Visualization")
    print("  8. Filter/Kernel Visualization")
    print("  9. Convolution Operation")
    print(" 10. Pooling Operations")
    print(" 11. Training Process")
    print(" 12. Loss Functions")
    print(" 13. Optimizers")
    print(" 14. Learning Rate Scheduling")
    print(" 15. Model Evaluation")
    print(" 16. Prediction Visualization")
    print(" 17. Data Preprocessing & Normalization")
    print(" 18. Batch Processing")


if __name__ == "__main__":
    main()
