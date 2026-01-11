# Cat vs Dog CNN Classifier

A deep learning project that uses a Convolutional Neural Network (CNN) to classify images of cats and dogs using PyTorch.

## Project Overview

This project implements a CNN model trained on cat and dog images to perform binary classification. The model achieves classification of images into two categories: Cat and Dog.

## Model Architecture

The CNN architecture consists of:
- **Feature Extraction Layers:**
  - 3 convolutional blocks with ReLU activation and max pooling
  - Progressively increasing filters: 32 → 64 → 128
- **Classification Layers:**
  - Fully connected layer with 256 neurons
  - Dropout layer (0.5) for regularization
  - Output layer with 2 classes

## Dataset Structure

```
data/
├── train/
│   ├── Cat/
│   └── Dog/
└── test/
    ├── Cat/
    └── Dog/
```

Place your cat images in the `Cat` folders and dog images in the `Dog` folders.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nadir2609/Cat-vs-Dog-Classifier.gi
cd cat-dog-cnn
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

**Option 1: Using Jupyter Notebook**
```bash
jupyter notebook cat_vs_dog.ipynb
```

**Option 2: Using Python Script**
```bash
python train.py
```

### Making Predictions

```bash
python predict.py --image path/to/your/image.jpg
```

## Results

The model trains for 5 epochs and outputs:
- Training loss per epoch
- Training accuracy per epoch
- Visual predictions on test samples

## Files Description

- `cat_vs_dog.ipynb` - Jupyter notebook with complete training pipeline
- `train.py` - Standalone training script
- `model.py` - CNN model architecture definition
- `predict.py` - Script for making predictions on new images
- `cat_dog_cnn.pth` - Saved model weights
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- PIL

## Training Parameters

- **Image Size:** 64x64
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 10

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- PyTorch team for the deep learning framework
- torchvision for image processing utilities
