import os
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV2, EfficientNetB0

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


MODEL_DIR = "models"


# -----------------------------
# BUILD CNN MODEL
# -----------------------------
def build_cnn_model():
    """High-performance CNN using DenseNet121 for 80%+ accuracy"""
    
    # Use DenseNet121 - excellent for medical imaging
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )
    
    # Strategic fine-tuning
    base_model.trainable = False
    # Unfreeze last 40 layers for fine-tuning
    for layer in base_model.layers[-40:]:
        layer.trainable = True
    
    # Build model on top of DenseNet121
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Efficient classifier head
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer
    output = tf.keras.layers.Dense(4, activation="softmax")(x)
    
    # Create model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    
    # Optimized optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    return model


# -----------------------------
# TRAIN CNN MODEL
# -----------------------------
def train_cnn(image_dataset_path):

    print("\nTraining CNN model on CT images...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Simple but effective data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        shear_range=0.15,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(image_dataset_path,"train"),
        target_size=(224,224),
        batch_size=16,  # Optimal batch size
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(image_dataset_path,"valid"),
        target_size=(224,224),
        batch_size=16,  # Consistent batch size
        class_mode="categorical",
        shuffle=False
    )

    # Print detected class labels
    print("Class indices:", train_generator.class_indices)

    model = build_cnn_model()

    # Optimized callbacks
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        mode="max",
        min_delta=0.003
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        mode="max",
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR,"best_cnn_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    )

    # Optimized class weights for higher accuracy
    class_counts = Counter(train_generator.classes)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Enhanced class weighting with better balance
    class_weights = {}
    for class_id, count in class_counts.items():
        # Inverse frequency with optimization
        weight = total_samples / (num_classes * count)
        # Apply smoothing to prevent extreme weights
        class_weights[class_id] = weight ** 0.9
    
    # Normalize weights to prevent dominance
    max_weight = max(class_weights.values())
    class_weights = {k: v/max_weight for k, v in class_weights.items()}
    
    print("Class distribution:", dict(class_counts))
    print("Optimized class weights:", class_weights)
    print(f"Total training samples: {total_samples}")
    print(f"Batch size: 16, Steps per epoch: {len(train_generator)}")

    # Optimized training configuration
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=60,  # Balanced number of epochs
        callbacks=[early_stop, checkpoint, reduce_lr],
        class_weight=class_weights,
        verbose=1,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator)
    )

    model.save(os.path.join(MODEL_DIR,"cnn_model.keras"))

    print("CNN model saved successfully.")

    return model, history


# -----------------------------
# TRAIN ML MODELS
# -----------------------------
def train_ml_models(csv_dataset_path):
    
    print("\nTraining ML models on symptom dataset...")
    
    # Import the improved preprocessing function
    from src.data_preprocessing import load_csv_data, balance_dataset
    
    X, y, scaler = load_csv_data(csv_dataset_path)
    
    # Balance the dataset
    X_balanced, y_balanced = balance_dataset(X, y)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # Optimized models with hyperparameters
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        "svm": SVC(
            probability=True,
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto'
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        "naive_bayes": GaussianNB(),
        "ann": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss'
        ),
        "logistic_regression": LogisticRegression(
            C=10,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
    }
    
    ensemble_estimators = [
        ('rf', models['random_forest']),
        ('svm', models['svm']),
        ('xgb', models['xgboost']),
        ('lr', models['logistic_regression'])
    ]
    
    models['ensemble'] = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft'
    )
    
    trained_models = {}
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Train each model with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Save the model
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        trained_models[name] = model
        
        # Cross-validation score
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if name == 'ensemble':
                # Special handling for ensemble model
                model_copy = VotingClassifier(
                    estimators=ensemble_estimators,
                    voting='soft'
                )
            else:
                model_copy = type(model)(**model.get_params())
            model_copy.fit(X_cv_train, y_cv_train)
            cv_scores.append(model_copy.score(X_cv_val, y_cv_val))
        
        print(f"{name} CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Save the scaler for future use
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    print("\nAll ML models saved successfully.")
    
    return trained_models, X_test, y_test