import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ===============================
# IMAGE DATA PREPROCESSING
# ===============================

# Define a function to prepare and load image datasets for training, validation, and testing
def load_image_data(dataset_path):

    # Construct the full file path for the training directory
    train_path = os.path.join(dataset_path, "train")
    # Construct the full file path for the validation directory
    valid_path = os.path.join(dataset_path, "valid")
    # Construct the full file path for the testing directory
    test_path = os.path.join(dataset_path, "test")

    # Define a generator that scales pixel values (0-255 to 0-1) and applies random transformations to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values to be between 0 and 1
        rotation_range=20, # Randomly rotate images by up to 20 degrees
        zoom_range=0.2, # Randomly zoom in or out by 20%
        horizontal_flip=True # Randomly flip images horizontally
    )

    # Define a generator for validation/testing that only scales pixels without adding random variations
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create a data stream for training images from the directory, resizing them and grouping them into batches
    train_generator = train_datagen.flow_from_directory(
        train_path, # Directory where training folders/classes are located
        target_size=(224,224), # Resize all images to 224x224 pixels
        batch_size=32, # Process 32 images at a time
        class_mode='categorical' # Use one-hot encoding for the multiple image classes
    )

    # Create a data stream for validation images using the same settings as training (no augmentation)
    valid_generator = test_datagen.flow_from_directory(
        valid_path, # Directory where validation folders are located
        target_size=(224,224), # Resize images to match the model input
        batch_size=32, # Process 32 images at a time
        class_mode='categorical' # Use one-hot encoding for labels
    )

    # Create a data stream for testing images to evaluate final model performance
    test_generator = test_datagen.flow_from_directory(
        test_path, # Directory where test folders are located
        target_size=(224,224), # Resize images to match the model input
        batch_size=32, # Process 32 images at a time
        class_mode='categorical' # Use one-hot encoding for labels
    )

    # Return the three generators to be used in the training script
    return train_generator, valid_generator, test_generator


# ===============================
# CSV DATA PREPROCESSING
# ===============================

# Define a function to load and clean tabular (CSV) data for machine learning
def load_csv_data(csv_path):
    
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle missing values
    df = df.dropna()
    
    # Convert GENDER to numeric
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    
    # Convert LUNG_CANCER to binary
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    
    # Feature engineering
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 50, 60, 70, 100], labels=[0, 1, 2, 3])
    df['RISK_SCORE'] = df[['SMOKING', 'ALCOHOL CONSUMING', 'CHRONIC DISEASE']].sum(axis=1)
    df['SYMPTOM_SEVERITY'] = df[['COUGHING', 'SHORTNESS OF BREATH', 'CHEST PAIN', 'WHEEZING']].sum(axis=1)
    
    # Convert categorical to numeric
    le = LabelEncoder()
    for col in df.select_dtypes(include=['category', 'object']).columns:
        if col != 'LUNG_CANCER':
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y, scaler

def balance_dataset(X, y):
    """Apply SMOTE to balance the dataset"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
