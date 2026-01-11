"""
Prediction script for Cat vs Dog CNN Classifier
"""
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import CatDogCNN


def load_model(model_path, device):
    """
    Load trained model from file.
    
    Args:
        model_path (str): Path to saved model weights
        device: Device to load model on
    
    Returns:
        model: Loaded PyTorch model
    """
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_image(model, image_path, device):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained PyTorch model
        image_path (str): Path to image file
        device: Device to run inference on
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    classes = ['Cat', 'Dog']
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score


def visualize_prediction(image_path, predicted_class, confidence):
    """
    Display the image with prediction.
    
    Args:
        image_path (str): Path to image
        predicted_class (str): Predicted class name
        confidence (float): Confidence score
    """
    image = Image.open(image_path)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict cat or dog from image')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='cat_dog_cnn.pth',
                        help='Path to trained model (default: cat_dog_cnn.pth)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Do not display the image')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    
    # Make prediction
    print(f"Predicting image: {args.image}")
    predicted_class, confidence = predict_image(model, args.image, device)
    
    # Display results
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Visualize if requested
    if not args.no_visualize:
        visualize_prediction(args.image, predicted_class, confidence)


if __name__ == "__main__":
    main()
