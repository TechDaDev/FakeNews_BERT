import os, platform, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

# Root directory
root_dir = Path(__file__).resolve().parent.parent

def generate_best_model_report(results_summary, X_train, X_test, y_train, y_test, prep_stats):
    if not results_summary:
        return None
    best_model_name, best_info = max(results_summary.items(), key=lambda x: x[1]['accuracy'])
    run_dir = best_info['run_dir']
    vectorizer_path = root_dir / "saved_models" / "tfidf_vectorizer.pkl"

    vectorizer = load(vectorizer_path)
    vocab_size = len(vectorizer.vocabulary_)
    env_info = {
        'Platform': platform.platform(),
        'Python Version': platform.python_version(),
        'Processor': platform.processor(),
    }
    gpu_info = None
    if best_model_name == 'Torch_FFNN':
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
        except Exception:
            pass
    cm = best_info.get('confusion_matrix')
    cm_heatmap_file = None
    metrics_chart_file = None
    class_metrics = {}
    if cm is not None:
        try:
            cm_arr = np.array(cm)
            fig, ax = plt.subplots(figsize=(6,6)) # Increased size
            im = ax.imshow(cm_arr, cmap='Blues')
            ax.set_title('Confusion Matrix Heatmap (Standardized)', fontsize=14, pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xticks([0,1]); ax.set_xticklabels(['Real','Fake'])
            ax.set_yticks([0,1]); ax.set_yticklabels(['Real','Fake'])
            # Add text annotations with better formatting
            for i in range(cm_arr.shape[0]):
                for j in range(cm_arr.shape[1]):
                    ax.text(j, i, f"{cm_arr[i, j]:,}", ha='center', va='center', 
                            color='white' if cm_arr[i,j] > cm_arr.max()/2 else 'black',
                            fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cm_heatmap_file = os.path.join(run_dir, f"{best_model_name}_confusion_matrix_heatmap.png")
            plt.tight_layout(); plt.savefig(cm_heatmap_file, dpi=300); plt.close(fig) # High DPI
        except Exception as e:
            print('Heatmap error:', e)
    cr_text = best_info.get('classification_report')
    if cr_text:
        try:
            for line in cr_text.strip().splitlines():
                line_clean = line.strip()
                if line_clean.startswith('Real') or line_clean.startswith('Fake'):
                    parts = line_clean.split()
                    if len(parts) >=5:
                        label = parts[0]
                        precision, recall, f1, support = parts[1:5]
                        class_metrics[label] = {
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1': float(f1),
                            'support': int(support)
                        }
            if class_metrics:
                metrics_chart_file = os.path.join(run_dir, f"{best_model_name}_class_metrics.png")
                labels = list(class_metrics.keys())
                metrics_names = ['precision','recall','f1']
                x = np.arange(len(labels)); width=0.2
                fig, ax = plt.subplots(figsize=(8,5))
                colors = ['#4e79a7', '#f28e2c', '#e15759'] # Professional palette
                for idx, m in enumerate(metrics_names):
                    ax.bar(x+(idx-1)*width, [class_metrics[l][m] for l in labels], width, label=m.capitalize(), color=colors[idx])
                ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1.1)
                ax.set_ylabel('Score', fontsize=12); ax.set_title('Classification Performance by Class', fontsize=14, pad=15); ax.legend(loc='lower right')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                for idx, m in enumerate(metrics_names):
                    for xi,l in enumerate(labels):
                        val = class_metrics[l][m]
                        ax.text(xi+(idx-1)*width, val+0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
                plt.tight_layout(); plt.savefig(metrics_chart_file, dpi=300); plt.close(fig) # High DPI
        except Exception as e:
            print('Metrics chart error:', e)
    md_lines = []
    md_lines.append(f"# Best Model Report: {best_model_name}\n")
    md_lines.append(f"Generated on: {best_info['timestamp']}\n")
    md_lines.append("## 1. Dataset & Preprocessing\n")
    md_lines.append("Source file: combined_TF_data.csv")
    md_lines.append("Features: Title + Text (Combined)")

    md_lines.append("Preprocessing: drop NaN (text/label), remove empty texts, remove duplicate texts (first kept).\n")
    md_lines.append("### Statistics")
    md_lines.append(f"- Raw rows: {prep_stats['raw_rows']}")
    md_lines.append(f"- Final rows: {prep_stats['final_rows']}")
    md_lines.append(f"- Dropped NaN: {prep_stats['dropped_nan']}")
    md_lines.append(f"- Empties removed: {prep_stats['empties_removed']}")
    md_lines.append(f"- Duplicates removed: {prep_stats['duplicates_removed']}")
    md_lines.append("- Label distribution (0=Real,1=Fake): " + str(prep_stats['label_distribution']))
    md_lines.append("\n## 2. Split\n")
    md_lines.append(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} (test_fraction={prep_stats['test_size']}, stratified={prep_stats['stratify']})\n")
    md_lines.append("## 3. TF-IDF\n")
    md_lines.append("Params: stop_words='english', max_df=0.7")
    md_lines.append(f"Vocab size: {vocab_size}\n")
    md_lines.append("## 4. Model & Hyperparameters\n")
    md_lines.append(f"Model: {best_model_name}")
    if 'hyperparams' in best_info:
        for k,v in best_info['hyperparams'].items():
            md_lines.append(f"- {k}: {v}")
    else:
        md_lines.append("(No hyperparameter snapshot)")
    if best_model_name != 'Torch_FFNN':
        md_lines.append("- epochs: N/A (non-neural)")
    md_lines.append("\n## 5. Training\n")
    md_lines.append(f"Training time: {best_info['training_time']:.2f} s")
    if best_model_name == 'Torch_FFNN' and 'history' in best_info:
        hist = best_info['history']; epochs_run = len(hist.get('loss', []))
        md_lines.append(f"Epochs run: {epochs_run}")
        md_lines.append("### Neural Network Method")
        md_lines.append("Feed-forward TF-IDF network: Linear->ReLU->Dropout -> Linear->ReLU->Dropout -> Linear->Logit; Adam optimizer (lr=1e-3, weight_decay=1e-4), BCEWithLogitsLoss, early stopping, grad clipping (5.0).")
        if 'loss' in hist:
            md_lines.append("Train loss: " + ', '.join(f"{x:.4f}" for x in hist['loss']))
        if 'val_loss' in hist:
            md_lines.append("Val loss: " + ', '.join(f"{x:.4f}" for x in hist['val_loss']))
    md_lines.append("\n## 6. Evaluation\n")
    md_lines.append(f"Accuracy: {best_info['accuracy']:.4f}")
    md_lines.append("Confusion Matrix (rows=true [Real, Fake], cols=pred):")
    if cm is not None:
        md_lines.append(f"``\n{cm}\n``")
    if cm_heatmap_file:
        md_lines.append(f"![Confusion Matrix Heatmap]({os.path.basename(cm_heatmap_file)})")
    if metrics_chart_file:
        md_lines.append(f"![Classification Metrics]({os.path.basename(metrics_chart_file)})")
    if cr_text:
        md_lines.append("Classification Report:\n```\n" + cr_text + "```")
    md_lines.append("\n## 7. Environment\n")
    for k,v in env_info.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append(f"- GPU: {gpu_info if gpu_info else 'None or not used'}")
    md_lines.append("\n## 8. Reproducibility\n")
    md_lines.append("1. Run training script (random_state=42).\n2. Keep same TF-IDF params & preprocessing.\n3. Check run_metadata.json label mapping.")
    report_path = os.path.join(run_dir, f"{best_model_name}_best_model_report.md")
    with open(report_path,'w') as f:
        f.write('\n'.join(md_lines))
    print('Best model report written:', report_path)
    return report_path
