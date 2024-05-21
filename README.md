# GasHisClassifier: Gastric Cancer Diagnosis with patch images from Whole Slide Images

This repository contains code and resources for detecting gastric cancer from Whole Slide Images (WSI) (processed into patch images) using deep learning models.

## Features

- **Deep Learning Models**: Implementations of VGG16, ResNet50, and EfficientNetB0 for image classification.
- **Data Processing**: Scripts for preprocessing and augmenting WSI data into patch images - smaller images.
- **Training and Evaluation**: Training models and evaluating their performance.
- **Web Application**: A simple web interface for uploading and classifying new images.

## Repository Structure

- `Data/`: Directory for storing datasets (can be excluded if too large) & script for image processing.
- `TrainModel/`: Scripts for training models.
- `Web/`: Flask web application for image classification.
- `requirements.txt`: Required Python packages.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Harito97/GasHisClassifier.git
cd GasHisClassifier
```
If you can not download, then zip the repository and download the zip file of all the repository.

### Step 2: Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

### Step 3: Prepare Data

If the `Data/` directory is too large, you can exclude it during the download. The code assumes you have a dataset ready in the `Data/` directory. You can organize your dataset as follows:

```
data_version_xx/
└── train/
    ├── class1/
    ├── class2/
    └── ...
└── valid/
    ├── class1/
    ├── class2/
    └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### Step 4: Data Processing and Model

1. **Data Exploration and Clustering**: 

   Include some techniques to explore the data, use clustering algorithms to try classification of images and preprocess the data to make data version 2.

2. **Training the Model**:  

   Run Python files with the format `Train_*.py` in `TrainModel/`, where `*` can be `VGGxx`, `ResNetxx`, `EfficientNetxx`.

### Step 5: Run the Web Application

1. **Start the Web Server**:

   ```bash
   python Web/app.py
   ```

2. **Access the Web Application**:

   Open your browser and navigate to `http://localhost:5000`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License.