# Simple Activity Recognition Project

This project focuses on detecting and recognizing basic activities (e.g., waving, nodding, walking) using computer vision and machine learning. It includes scripts for capturing labeled data, preparing the dataset, training a machine learning model, and making real-time predictions.

## Files

- **capture_and_label.py**: Captures video from a webcam, detects motion, and saves labeled frames for training.
- **prepare_dataset.py**: Prepares the captured data for model training by organizing and processing the labeled images.
- **train_model.py**: Trains a machine learning model using the prepared dataset.
- **predict_activity.py**: Uses the trained model to predict activities in real-time.

## Usage

1. **Capture and Label Data**:
    - Run `capture_and_label.py` to capture and label video data for different activities.
    - The labeled images will be saved in the `labeled_data` directory.

2. **Prepare the Dataset**:
    - Run `prepare_dataset.py` to process the labeled images into a dataset ready for model training.

3. **Train the Model**:
    - Run `train_model.py` to train a machine learning model using the prepared dataset.

4. **Predict Activities**:
    - Run `predict_activity.py` to predict activities in real-time using your webcam.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
