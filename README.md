# Pneumonia Classifier

A machine learning-based pneumonia classifier using deep learning and PyTorch. This project trains a convolutional neural network (CNN) to classify chest X-ray images as either showing pneumonia or being healthy.

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model](#model)
- [Training](#training)
- [Web Application](#web-application)
- [License](#license)

## Live Demo
[View Demo](https://aryansachan12-pneumonia-classifier-app-cytodc.streamlit.app/)

## Installation

Follow these steps to get the project up and running:

1. Clone this repository:
    ```bash
    git clone https://github.com/AryanSachan12/pneumonia_classifier.git
    cd pneumonia_classifier
    ```

2. Set up a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Requirements

Ensure you have the following packages installed:

- Python 3.x
- PyTorch
- Streamlit
- Pillow
- Matplotlib
- torchvision

You can install all dependencies with:

```bash
pip install -r requirements.txt
````

Here’s the list of libraries in `requirements.txt`:

```
torch
torchvision
streamlit
pillow
matplotlib
```

## Usage

1. **Running the Web Application:**

   After installing dependencies, run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

   This will start a local server where you can upload images and see the model's predictions.

2. **Predicting with the Model:**

   You can use the trained model for predictions by running the following script in Python:

   ```python
   import torch
   from model import PneumoniaClassifier  # Your model's class

   model = PneumoniaClassifier()
   model.load_state_dict(torch.load('pneumonia_classifier.pth'))
   model.eval()

   # Example image preprocessing and prediction
   from PIL import Image
   from torchvision import transforms

   image = Image.open('test_image.jpg')
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])

   image = transform(image).unsqueeze(0)  # Add batch dimension
   with torch.no_grad():
       output = model(image)
   prediction = torch.argmax(output, dim=1)
   print(f'Predicted class: {prediction.item()}')
   ```

## Model

This project uses a Convolutional Neural Network (CNN) for classification. The model is trained on chest X-ray images to detect whether the image shows signs of pneumonia or if the patient’s lungs are healthy.

The model is saved as `pneumonia_classifier.pth` and can be loaded into any PyTorch-compatible environment for inference.

## Training

To train the model, use the script `model_training.py`. Ensure you have the necessary datasets in the appropriate directories and that you’ve set up your environment correctly. Here is the basic usage to train the model:

1. Prepare your dataset:

   * Put the training and validation data in respective folders (`train` and `valid`) with subfolders `pneumonia` and `normal` for classifying images.

2. Run the training script:

   ```bash
   python model_training.py
   ```

3. The model will be saved as `pneumonia_classifier.pth`.

## Web Application

The Streamlit app (`app.py`) provides a user-friendly interface where you can upload chest X-ray images and get predictions from the trained model. The app uses the model saved in the `pneumonia_classifier.pth` file.

### Steps:

* Run the Streamlit app using `streamlit run app.py`.
* Upload an image and see if it’s classified as healthy or pneumonia-infected.
