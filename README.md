# activity-recognition

## Overview

This project captures video data, labels activities (e.g., waving, nodding, walking), and trains a machine learning model to recognize these activities.

## Files

- **capture_and_label.py**: Captures and labels video data.
- **train_model.py**: Trains a machine learning model using labeled data.
- **requirements.txt**: List of required Python packages.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Capture and Label Data**:
   ```bash
   python capture_and_label.py
   ```
   Labeled frames are saved in the `labeled_data/` directory.

3. **Train the Model**:
   ```bash
   python train_model.py
   ```
   The trained model is saved as `activity_model.pkl`.

## License

This project is licensed under the MIT License.

