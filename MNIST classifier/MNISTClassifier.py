import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier

# Interface for all classification models
class DigitClassificationInterface:
    def predict(self, image: np.ndarray) -> int:
        """
        Predict the class (digit) for the given image.
        :param image: Input image as a NumPy array (28x28x1).
        :return: Predicted class as an integer.
        """
        raise NotImplementedError("The predict method is not implemented.")

    def train(self, *args, **kwargs):
        """
        Train the model with given data (Not implemented).
        """
        raise NotImplementedError("The train method is not implemented.")

# CNN model 
class CNNModel(DigitClassificationInterface):
    def __init__(self):
        # Define a simple CNN architecture
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Define transformations
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image: np.ndarray) -> int:
        # Convert to grayscale if color
        if img.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = numpy.expand_dims(image, axis=-1)
            
        # Apply transformations
        image_tensor = self.transforms(image.squeeze(-1)).unsqueeze(0).to(self.device)  # Convert to tensor, add batch dimension, and move to device

        # Perform prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return int(predicted.item())

# Random Forest model
class RFModel(DigitClassificationInterface):
    def __init__(self):
        # Initialize Random Forest classifier
        self.rf_classifier = RandomForestClassifier(n_estimators=100)
        self.trained = False

    #def train(self, X_train, y_train):
    #    # Train the Random Forest model
    #    self.rf_classifier.fit(X_train, y_train)
    #    self.trained = True

    def predict(self, image: np.ndarray) -> int:
        # Convert to grayscale if color
        if img.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = numpy.expand_dims(image, axis=-1)
            
        # Resize the image to 28x28
        resized_image = cv2.resize(image.squeeze(-1), (28, 28))
        
        # Flatten the image
        flattened_image = image.flatten().reshape(1, -1)  # Reshape for single prediction

        # Perform prediction
        prediction = self.rf_classifier.predict(flattened_image)
        return int(prediction[0])

# Random model 
class RandomModel(DigitClassificationInterface):
    def __init__(self):
        self.rand_classifier = None
        # Center crop transform
        self.center_crop = transforms.CenterCrop(10)

    def predict(self, image: np.ndarray) -> int:
        image_pil = Image.fromarray(image)
        # Center crop
        cropped_image = self.center_crop(image_pil)
        cropped_image_np = np.array(cropped_image)
        
        return int(np.random.randint(0, 10))  # Simulating a prediction

# Main classifier class
class DigitClassifier:
    def __init__(self, algorithm: str):
        """
        Initialize the classifier with the specified algorithm.
        :param algorithm: The algorithm to use ('cnn', 'rf', 'rand').
        """
        if algorithm == 'cnn':
            self.model = CNNModel()
        elif algorithm == 'rf':
            self.model = RFModel()
        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError("Unsupported algorithm. Choose 'cnn', 'rf', or 'rand'.")

    def predict(self, image: np.ndarray) -> int:
        """
        Predict the class (digit) for the given image using the selected model.
        :param image: Input image as a NumPy array (28x28x1).
        :return: Predicted class as an integer.
        """
        return self.model.predict(image)

# Example usage
if __name__ == "__main__":
    # Example input image (28x28x1)
    example_image = np.random.randint(0, 255, (28, 28, 1), dtype=np.uint8)

    # Initialize the classifier with the CNN model
    classifier = DigitClassifier(algorithm='cnn')
    prediction = classifier.predict(example_image)
    print(f"Predicted digit using CNN: {prediction}")

    # Initialize the classifier with the RF model
    classifier = DigitClassifier(algorithm='rf')
    prediction = classifier.predict(example_image)
    print(f"Predicted digit using RF: {prediction}")

    # Initialize the classifier with the Random model
    classifier = DigitClassifier(algorithm='rand')
    prediction = classifier.predict(example_image)
    print(f"Predicted digit using Random model: {prediction}")
