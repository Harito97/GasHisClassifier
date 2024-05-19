# GasHisClassifier: Gastric Cancer Diagnosis from Whole Slide Images

This repository contains code and resources for detecting gastric cancer from Whole Slide Images (WSI) using deep learning models.

## Features

- **Deep Learning Models**: Implementations of VGG16, ResNet50, and EfficientNetB0 for image classification.
- **Data Processing**: Scripts for preprocessing and augmenting the WSI data.
- **Training and Evaluation**: Jupyter notebooks for training models and evaluating their performance.
- **Web Application**: A simple web interface for uploading and classifying new images.

## Repository Structure

- `data/`: Directory for storing dataset (can be excluded if too large).
- `models/`: Pre-trained model weights.
- `notebooks/`: Jupyter notebooks for data exploration, training, and clustering.
- `scripts/`: Python scripts for data processing and utility functions.
- `Web/`: Flask web application for image classification.
- `requirements.txt`: Required Python packages.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Harito97/GasHisClassifier.git
cd GasHisClassifier
```

### Step 2: Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

### Step 3: Prepare Data

If the `data/` directory is too large, you can exclude it during the download. The code assumes you have a dataset ready in the `data/` directory. You can organize your dataset as follows:

```
data/
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

### Step 4: Run Jupyter Notebooks

1. **Training the Model**: Use the following notebook to train your model.

   ```bash
   jupyter notebook notebooks/train_model.ipynb
   ```

2. **Data Exploration and Clustering**: Use the following notebook to explore your data and perform clustering.

   ```bash
   jupyter notebook notebooks/data_exploration_clustering.ipynb
   ```

### Step 5: Run the Web Application

1. **Start the Web Server**:

   ```bash
   python Web/app.py
   ```

2. **Access the Web Application**:

   Open your browser and navigate to `http://localhost:5000`.

## Usage

- **Train Model**: Follow the steps in `notebooks/train_model.ipynb` to train your classification model.
- **Explore Data and Cluster Segments**: Use `notebooks/data_exploration_clustering.ipynb` for data visualization and clustering analysis.
- **Classify New Images**: Use the web interface to upload and classify new WSI images.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License.
