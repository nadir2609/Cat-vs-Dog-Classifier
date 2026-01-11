"""
Training script for Cat vs Dog CNN Classifier
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

from model import CatDogCNN


def get_data_loaders(batch_size=32):
    """
    Create data loaders for training and testing.
    
    Args:
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
    test_dataset = datasets.ImageFolder("data/test", transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train the CNN model.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        epochs (int): Number of training epochs
    
    Returns:
        model: Trained model
    """
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Train Acc: {train_acc:.2f}%")

    return model


def visualize_predictions(model, test_dataset, train_dataset, device, num_samples=10):
    """
    Visualize model predictions on random test samples.
    
    Args:
        model: Trained PyTorch model
        test_dataset: Test dataset
        train_dataset: Train dataset (for class names)
        device: Device (cuda/cpu)
        num_samples (int): Number of samples to visualize
    """
    indices = random.sample(range(len(test_dataset)), num_samples)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    model.eval()
    with torch.no_grad():
        for idx, test_idx in enumerate(indices):
            image, true_label = test_dataset[test_idx]
            
            # Get prediction
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            _, pred_label = torch.max(output, 1)
            
            # Denormalize image for display
            img_display = image.permute(1, 2, 0).cpu().numpy()
            img_display = img_display * 0.5 + 0.5
            img_display = img_display.clip(0, 1)
            
            # Get class names
            true_class = train_dataset.classes[true_label]
            pred_class = train_dataset.classes[pred_label.item()]
            
            # Plot
            axes[idx].imshow(img_display)
            axes[idx].axis('off')
            color = 'green' if true_label == pred_label.item() else 'red'
            axes[idx].set_title(f'True: {true_class}\nPred: {pred_class}', color=color)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Predictions saved to predictions.png")
    plt.show()


def main():
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(batch_size=32)
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("\nStarting training...")
    model = train_model(model, train_loader, criterion, optimizer, device, epochs=10)
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), "cat_dog_cnn.pth")
    print("Model saved to cat_dog_cnn.pth")
    
    # Visualize predictions
    print("\nGenerating predictions visualization...")
    visualize_predictions(model, test_dataset, train_dataset, device)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
