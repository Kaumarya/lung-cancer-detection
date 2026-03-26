from src.train_models import train_cnn, train_ml_models
from src.evaluate_models import evaluate_cnn, evaluate_ml_models, plot_model_comparison
from src.data_preprocessing import load_csv_data, balance_dataset
import os

def main():
    
    image_dataset_path = "dataset/lung_images"
    csv_dataset_path = "dataset/lung_cancer.csv"
    
    print("\n==============================")
    print("LUNG CANCER AI PIPELINE STARTED")
    print("==============================")
    
    # Check if image dataset exists
    if not os.path.exists(image_dataset_path):
        print(f"Warning: Image dataset path {image_dataset_path} not found.")
        print("Skipping CNN training. Only ML models will be trained.")
        train_cnn_model = False
    else:
        train_cnn_model = True
    
    # Train CNN (image dataset)
    if train_cnn_model:
        print("\nTraining CNN model on CT images...")
        try:
            cnn_model, history = train_cnn(image_dataset_path)
            print("\nEvaluating CNN model...")
            cnn_accuracy = evaluate_cnn(image_dataset_path)
        except Exception as e:
            print(f"CNN training failed: {e}")
            cnn_accuracy = 0
    else:
        cnn_accuracy = 0
    
    # Train ML models (symptom dataset)
    print("\nTraining ML models on symptom dataset...")
    try:
        trained_models, X_test, y_test = train_ml_models(csv_dataset_path)
        
        # Evaluate ML models
        print("\nEvaluating ML models...")
        results = evaluate_ml_models(X_test, y_test)
        
        # Plot comparison
        plot_model_comparison(results)
        
        # Find best model
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]
        
        print("\n==============================")
        print("PIPELINE FINISHED SUCCESSFULLY")
        print("==============================")
        print(f"\nBest ML Model: {best_model_name}")
        print(f"Best ML Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        if train_cnn_model:
            print(f"CNN Accuracy: {cnn_accuracy:.4f} ({cnn_accuracy*100:.2f}%)")
        
        # Check if target accuracy achieved
        target_accuracy = 0.85
        if best_accuracy >= target_accuracy:
            print(f"\n TARGET ACHIEVED! Accuracy >= {target_accuracy*100}%")
        else:
            print(f"\n Target not reached. Best accuracy: {best_accuracy*100:.2f}% < {target_accuracy*100}%")
        
    except Exception as e:
        print(f"Error in ML pipeline: {e}")
        return


# Standard Python check to ensure the script only runs if executed directly
if __name__ == "__main__":
    main()