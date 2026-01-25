import os
import json
import time
from datetime import datetime
import sys
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.models import (
    train_linear_svc,
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest,
    train_torch_ffnn,
    TORCH_AVAILABLE
)
from scripts.data_prep import load_and_preprocess_data
from src.reporting import generate_best_model_report
from src.predict_utils import default_length_bins, predict_with_model

def train_all_models(X_train, X_test, y_train, y_test, use_torch=True):
    """Train and evaluate all models"""
    print("\n=== Model Training and Evaluation ===")
    models_config = {
        'LinearSVC': train_linear_svc,
        'Logistic_Regression': train_logistic_regression,
        'Naive_Bayes': train_naive_bayes,
        'Random_Forest': train_random_forest
    }
    if use_torch and TORCH_AVAILABLE:
        # Use smaller hidden_dim & batch for memory safety
        models_config['Torch_FFNN'] = lambda X, Y: train_torch_ffnn(
            X, Y, epochs=15, batch_size=64, hidden_dim=256, predict_batch_size=512, use_half=False
        )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = root_dir / 'saved_models' / 'model_runs' / timestamp
    print(f"Artifacts will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # Save metadata about label mapping
    meta = {
        'label_meaning': {'0':'Real','1':'Fake'},
        'label_flip_applied': True,
        'timestamp': timestamp
    }
    with open(os.path.join(output_dir,'run_metadata.json'),'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved run_metadata.json with label mapping.')
    results_summary = {}
    
    for model_name, train_function in models_config.items():
        print(f"\n{'='*50}")
        print(f"Processing {model_name}")
        print(f"{'='*50}")
        
        try:
            # Train model
            start_time = time.time()
            model = train_function(X_train, y_train)
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate model
            accuracy, confusion, report, prediction_time = evaluate_model(
                model, X_test, y_test, model_name
            )
            
            # Save model and results
            model_filename, results_filename = save_model_and_results(
                model, model_name, accuracy, confusion, report, prediction_time, timestamp, output_dir
            )
            
            # Store results for summary
            results_summary[model_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'model_filename': model_filename,
                'results_filename': results_filename,
                'run_dir': output_dir,
                'timestamp': timestamp,
                'confusion_matrix': confusion.tolist(),
                'classification_report': report
            }
            # Hyperparameters snapshot
            try:
                if hasattr(model, 'is_torch_model') and model.is_torch_model:
                    results_summary[model_name]['hyperparams'] = {
                        'model_type': 'Torch_FFNN',
                        'epochs': 15,
                        'batch_size': 128,
                        'hidden_dim': 512,
                        'lr': 1e-3,
                        'val_split': 0.1,
                        'early_stop_patience': 3,
                        'weight_decay': 1e-4
                    }
                elif hasattr(model, 'get_params'):
                    params = model.get_params()
                    # Keep only a small subset for brevity
                    keep_keys = [k for k in params.keys() if k in {'C','penalty','dual','max_iter','n_estimators','random_state','alpha'}]
                    results_summary[model_name]['hyperparams'] = {k: params[k] for k in keep_keys}
            except Exception:
                pass
            # If torch model, attach history for later plotting
            if hasattr(model, 'history'):
                results_summary[model_name]['history'] = model.history
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    return results_summary

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics (with label names)."""
    print(f"\n=== Evaluating {model_name} ===")
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred, labels=[0,1])
    report = classification_report(y_test, y_pred, labels=[0,1], target_names=['Real','Fake'])
    print(f'Model: {model_name}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Prediction Time: {prediction_time:.4f} seconds')
    print('Confusion Matrix (rows=true, cols=pred) order [Real,Fake]:')
    print(confusion)
    print(f'Classification Report:\n{report}')
    return accuracy, confusion, report, prediction_time

def save_model_and_results(model, model_name, accuracy, confusion, report, prediction_time, timestamp, output_dir):
    """Save model and results to files inside output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, f"{model_name}_{timestamp}.pkl")
    dump(model, model_filename)
    print(f"Model saved as: {model_filename}")
    results_filename = os.path.join(output_dir, f"{model_name}_{timestamp}.txt")
    with open(results_filename, "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Prediction Time: {prediction_time:.4f} seconds\n\n")
        file.write("Confusion Matrix:\n")
        file.write(str(confusion) + "\n\n")
        file.write("Classification Report:\n")
        file.write(report)
    print(f"Results saved as: {results_filename}")
    return model_filename, results_filename

def print_summary(results_summary):
    """Print summary of all model results"""
    print("\n" + "="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'Training Time':<15} {'Prediction Time':<15}")
    print("-" * 80)
    
    for model_name, results in results_summary.items():
        print(f"{model_name:<20} {results['accuracy']:<10.4f} {results['training_time']:<15.2f} {results['prediction_time']:<15.4f}")
    
    # Find best model
    if results_summary:
        best_model = max(results_summary.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
    
    print("="*80)

def interactive_prediction(results_summary):
    """Interactive prediction using trained models (enhanced)."""
    print("\n=== Interactive Prediction ===")
    if not results_summary:
        print("No trained models available for prediction.")
        return
    names = list(results_summary.keys())
    for i,n in enumerate(names,1):
        print(f"{i}. {n}")
    raw = input('\nSelect model (number/name, q=quit): ').strip()
    if raw.lower()=='q':
        return
    if raw.isdigit():
        idx = int(raw)-1
        if not (0<=idx<len(names)):
            print('Invalid model selection.')
            return
        selected = names[idx]
    else:
        nm_lower = {n.lower():n for n in names}
        if raw.lower() not in nm_lower:
            print('Model name not found.')
            return
        selected = nm_lower[raw.lower()]
    model_filename = results_summary[selected]['model_filename']
    # Advanced settings
    try:
        use_chunk = input('Use chunking? (y/N): ').strip().lower()=='y'
        margin_in = input('Uncertainty margin (default 0.02): ').strip()
        margin = float(margin_in) if margin_in else 0.02
        adaptive = input('Use adaptive length thresholds? (y/N): ').strip().lower()=='y'
        lbins = default_length_bins() if adaptive else None
    except Exception:
        use_chunk = False
        margin = 0.02
        lbins = None
    while True:
        txt = input('\nEnter text (/back to exit): ').strip()
        if txt.lower() in {'/back','/q','/quit'}:
            break
        if not txt:
            continue
        label, p_fake = predict_with_model(
            model_filename, txt, return_score=True, margin=margin,
            use_chunking=use_chunk, length_bin_thresholds=lbins
        )
        print(f"Final: {label} (P(Fake)={p_fake:.4f})")

# === Main execution block added ===
if __name__ == "__main__":
    print("Starting multi-algorithm training pipeline...")
    X_train, X_test, y_train, y_test, prep_stats = load_and_preprocess_data()
    results = train_all_models(X_train, X_test, y_train, y_test, use_torch=True)
    print_summary(results)
    generate_best_model_report(results, X_train, X_test, y_train, y_test, prep_stats)
    try:
        interactive_prediction(results)
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")