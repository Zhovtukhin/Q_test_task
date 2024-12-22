### Data Science Engineering Tasks
This repository contains solutions for the three tasks from Quantum team. Each task is implemented in its dedicated folder

#### Repository Structure
```
├── Counting islands/           # Task 1: Counting Islands
│   ├── count_islands.py     # Python script for solving the problem
│
├──Regression on the tabular data/         # Task 2: Regression on Tabular Data
│   ├── train.py                # Script for training the model
│   ├── predict.py              # Script for making predictions
│   ├── EDA.ipynb               # Jupyter Notebook for exploratory data analysis
│   ├── correlation matrix.png               # correlation matrix (not display on notebook)
│   ├── requirements.txt        # Dependencies required to run the scripts
│   ├── train.csv               # Training dataset (example)
│   ├── hidden_test.csv         # Test dataset (example)
│   └── hidden_test_target.csv         # Predictions on the test dataset
│
├── MNIST classifier/           # Task 3: MNIST Classifier
│   ├── MNISTClassifier.py     # Iimplementation of the DigitClassifier
│
└── README.md                   # General repository information
```

### Task Descriptions
##### 1. Counting Islands
count_islands.py contains function count_islands that take numpe array as input and return numver of islands. Component analysis from opencv used for this task. If you execute file from the terminal output will show you test cases.
##### 2. Regression on Tabular Data
While researching the data, I found that two features (6 and 8) were correlated, and one of them (feature 8) contained only 0 and 1, which is not representative of the target value. Therefore, I dropped the 8th column during model training. The remaining features appeared to be fairly random and showed no significant correlation with the target.
For preprocessing, I applied a MinMax scaler to normalize the data. The target value was divided by 100 during training, so the model's predictions should be multiplied by 100 during inference.
I used a multi-layer perceptron (MLP) as the model, configured with 500 hidden units. The root mean square error (RMSE) achieved was 0.0085 on the training data and 0.0087 on the test data.
##### 3. MNIST Classifier
MNISTClassifier.py contains all requred classes. If you execute file from the terminal, script will call all 3 models with random array. Requires PyTorch