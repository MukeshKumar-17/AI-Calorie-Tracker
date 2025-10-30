# Food Classifier CNN

A lightweight CNN model for classifying 7 food categories with nutrition information.

## Setup

1. **Install dependencies:**
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

2. **Prepare your dataset:**
   - Place the `FOOD_DATA` folder in the same directory as `train_food_classifier.py`
   - Ensure folder structure:
     ```
     FOOD_DATA/
       chicken_curry/
       chicken_wings/
       french_fries/
       grilled_cheese_sandwich/
       omelette/
       pizza/
       samosa/
     ```

## Usage

### Train the model:
```bash
python train_food_classifier.py
```

This will:
- Train the model for 10 epochs
- Save training graphs (`training_history.png` and `confusion_matrix.png`)
- Save the trained model as `food_classifier.h5`

### Make predictions on new images:
```python
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

# Load the saved model
model = load_model('food_classifier.h5')

# Define the prediction function (copy from train_food_classifier.py)
def predict_food(img_path):
    # ... (use the function from the script)
    pass

# Predict
result = predict_food('path/to/your/test_image.jpg')
```

## Model Details
- **Input size:** 128x128 RGB images
- **Architecture:** 3 Conv blocks + Dense layer
- **Classes:** 7 food categories
- **Training time:** ~5-10 minutes on GPU

## Output
- Final validation accuracy
- Training/validation accuracy and loss plots
- Confusion matrix
- Trained model file (`food_classifier.h5`)
