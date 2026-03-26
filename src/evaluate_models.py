# Import os to manage file directories and paths
import os
# Import joblib to load the saved traditional machine learning models (.pkl files)
import joblib
# Import numpy for numerical array manipulations
import numpy as np
# Import matplotlib's pyplot for creating data visualizations and charts
import matplotlib.pyplot as plt
# Import seaborn for advanced statistical data visualization (like heatmaps)
import seaborn as sns
# Import tensorflow to load and evaluate the deep learning CNN model
import tensorflow as tf

# Import scikit-learn metrics to calculate accuracy, confusion matrices, and detailed reports
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Import the custom image loader from your preprocessing script
from src.data_preprocessing import load_image_data


# -----------------------------
# CNN EVALUATION
# -----------------------------

# Define a function to test the performance of the trained CNN model
def evaluate_cnn(dataset_path):
    
    _, _, test_gen = load_image_data(dataset_path)
    
    model = tf.keras.models.load_model("models/cnn_model.keras")
    
    loss, accuracy, auc, precision, recall = model.evaluate(test_gen)
    
    print(f"\nCNN Test Accuracy: {accuracy:.4f}")
    print(f"CNN Test AUC: {auc:.4f}")
    print(f"CNN Test Precision: {precision:.4f}")
    print(f"CNN Test Recall: {recall:.4f}")
    
    # Create a line plot showing CNN metrics
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall']
    values = [accuracy, auc, precision, recall]
    
    plt.figure(figsize=(10,6))
    plt.plot(metrics, values, marker='o', linewidth=3, markersize=10, color='darkblue', markerfacecolor='red')
    plt.title('CNN Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    
    # Add value labels on each point
    for i, value in enumerate(values):
        plt.annotate(f'{value:.3f}', (i, value), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('results/cnn_metrics_line.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy


# -----------------------------
# ML MODELS EVALUATION
# -----------------------------

# Define a function to evaluate all traditional ML models and generate performance visuals
def evaluate_ml_models(X_test, y_test):

    model_files = [
        "random_forest.pkl",
        "svm.pkl", 
        "knn.pkl",
        "decision_tree.pkl",
        "naive_bayes.pkl",
        "ann.pkl",
        "gradient_boost.pkl",
        "xgboost.pkl",
        "logistic_regression.pkl",
        "ensemble.pkl"
    ]

    # Initialize a dictionary to store accuracy scores for each model
    results = {}

    # Create a 'results' folder if it doesn't exist to store the generated graphs
    os.makedirs("results", exist_ok=True)

    # Iterate through each model file to load, predict, and evaluate
    for model_file in model_files:

        # Create the full relative path to the specific model file
        model_path = os.path.join("models", model_file)

        # Load the trained model into memory
        model = joblib.load(model_path)

        # Use the model to predict outcomes based on the test feature set
        y_pred = model.predict(X_test)

        # Calculate the accuracy by comparing predictions to the actual labels
        acc = accuracy_score(y_test, y_pred)

        # Store the accuracy score in our results dictionary
        results[model_file] = acc

        # Print the model name and its accuracy to the console
        print(f"\nModel: {model_file}")
        print("Accuracy:", acc)

        # Print a detailed report (precision, recall, f1-score) for the current model
        print(classification_report(y_test, y_pred))

        # Generate a confusion matrix to see where the model made correct/incorrect guesses
        cm = confusion_matrix(y_test, y_pred)

        # Initialize a new figure for the heatmap visualization
        plt.figure(figsize=(6,5))
        # Create a heatmap using seaborn with annotations (numbers) and a blue color scheme
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        # Set the title and axis labels for the confusion matrix plot
        plt.title(f"Confusion Matrix - {model_file}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Save the plot as a PNG image in the results directory
        plt.savefig(f"results/{model_file}_confusion_matrix.png")

        # Close the plot to free up system memory
        plt.close()

    # Return the dictionary containing all model accuracies
    return results


# -----------------------------
# MODEL COMPARISON GRAPH
# -----------------------------

# Define a function to create a line chart comparing the accuracy of all models
def plot_model_comparison(results):

    # Extract model names for the horizontal (X) axis
    names = list(results.keys())
    # Extract accuracy scores for the vertical (Y) axis
    scores = list(results.values())
    
    # Clean model names for better display
    clean_names = [name.replace('.pkl', '').replace('_', ' ').title() for name in names]
    
    # Initialize the figure size for the comparison chart
    plt.figure(figsize=(12,6))
    
    # Create a line plot with markers
    plt.plot(clean_names, scores, marker='o', linewidth=2, markersize=8, color='blue', markerfacecolor='red', markeredgewidth=2)
    
    # Set the chart title and the label for the Y-axis
    plt.title("Model Accuracy Comparison - Line Plot", fontsize=16, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Models", fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate the model names on the X-axis by 45 degrees for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits to better show differences
    plt.ylim(0.7, 1.0)
    
    # Add value labels on each point
    for i, score in enumerate(scores):
        plt.annotate(f'{score:.3f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Automatically adjust layout to prevent labels from being cut off
    plt.tight_layout()
    
    # Save the final comparison chart as an image
    plt.savefig("results/model_accuracy_comparison_line.png", dpi=300, bbox_inches='tight')
    
    # Display the chart on the screen
    plt.show()
