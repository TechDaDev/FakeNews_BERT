import time
from joblib import dump
from typing import Optional

# Sklearn models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Optional PyTorch model
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

__all__ = [
    'train_linear_svc',
    'train_logistic_regression',
    'train_naive_bayes',
    'train_random_forest',
    'train_torch_ffnn',
    'TORCH_AVAILABLE'
]

def train_linear_svc(X_train, y_train):
    start = time.time()
    model = LinearSVC(dual=False, random_state=42)
    model.fit(X_train, y_train)
    model._fit_time = time.time() - start
    return model

def train_logistic_regression(X_train, y_train):
    start = time.time()
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    model._fit_time = time.time() - start
    return model

def train_naive_bayes(X_train, y_train):
    start = time.time()
    model = MultinomialNB()
    model.fit(X_train, y_train)
    model._fit_time = time.time() - start
    return model

def train_random_forest(X_train, y_train):
    start = time.time()
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    model._fit_time = time.time() - start
    return model

# ----------------- PyTorch FeedForward Network -----------------
class TorchNewsClassifierWrapper:
    """Wraps a PyTorch model to provide a sklearn-like predict interface."""
    def __init__(self, model, device, vectorizer_dim, batch_size_predict: int = 1024):
        self.model = model
        self.device = device
        self.vectorizer_dim = vectorizer_dim
        self.is_torch_model = True  # flag used by saving logic
        self.batch_size_predict = batch_size_predict

    def predict(self, X):
        import numpy as np
        import torch
        from scipy import sparse
        self.model.eval()
        preds = []
        if hasattr(X, 'toarray') and hasattr(X, 'shape') and sparse.issparse(X):
            # Chunked prediction to avoid densifying whole matrix
            n = X.shape[0]
            for start in range(0, n, self.batch_size_predict):
                end = min(start + self.batch_size_predict, n)
                batch = X[start:end].toarray().astype('float32')
                with torch.no_grad():
                    tensor = torch.from_numpy(batch).to(self.device)
                    logits = self.model(tensor).squeeze(-1)
                    batch_preds = (logits.sigmoid() > 0.5).long().cpu().numpy().ravel()
                preds.append(batch_preds)
            return np.concatenate(preds)
        else:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            with torch.no_grad():
                tensor = torch.from_numpy(X).float().to(self.device)
                logits = self.model(tensor)
                preds = (logits.sigmoid() > 0.5).long().cpu().numpy().ravel()
            return preds

    def save(self, path):
        import torch
        torch.save({
            'state_dict': self.model.state_dict(),
            'vectorizer_dim': self.vectorizer_dim
        }, path)

if TORCH_AVAILABLE:
    class FFNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim//2, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

def train_torch_ffnn(
    X_train,
    y_train,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    require_gpu: bool = False,
    predict_batch_size: int = 512,
    use_half: bool = False,
    verbose: bool = True,
    val_split: float = 0.1,
    early_stop_patience: int = 3,
    weight_decay: float = 1e-4,
    pos_weight_auto: bool = True,
    max_grad_norm: float = 5.0
):
    """Train a feed-forward net on (sparse) TF-IDF without densifying the whole matrix.

    Adds validation split, early stopping, weight decay, optional class imbalance handling,
    gradient clipping, and tracking of validation loss.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError('PyTorch not installed. Cannot train torch model.')

    import numpy as np
    import torch
    import torch.nn as nn
    from scipy import sparse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if require_gpu and device.type != 'cuda':
        raise RuntimeError('GPU requested (require_gpu=True) but CUDA device not available.')
    if verbose:
        print(f"[Torch FFNN] Using device: {device}")
        if device.type == 'cuda':
            print(f"[Torch FFNN] GPU name: {torch.cuda.get_device_name(0)}")

    # Prepare labels
    y_np = y_train.values.astype('float32') if hasattr(y_train, 'values') else y_train.astype('float32')
    n_samples, n_features = X_train.shape
    if verbose:
        print(f"[Torch FFNN] Training samples: {n_samples}, features: {n_features}")

    model = FFNN(n_features, hidden_dim).to(device)
    if use_half and device.type == 'cuda':
        model.half()

    # Class imbalance handling via pos_weight for BCEWithLogitsLoss
    if pos_weight_auto:
        pos_count = (y_np == 1).sum()
        neg_count = (y_np == 0).sum()
        if pos_count > 0:
            imbalance_ratio = neg_count / max(pos_count, 1)
            pos_weight_tensor = torch.tensor([imbalance_ratio], device=device)
        else:
            pos_weight_tensor = None
    else:
        pos_weight_tensor = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) if pos_weight_tensor is not None else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    index_order = np.arange(n_samples)

    # Validation split indices
    val_size = int(n_samples * val_split)
    if val_size > 0:
        np.random.shuffle(index_order)
        val_idx = index_order[:val_size]
        train_idx_full = index_order[val_size:]
    else:
        val_idx = None
        train_idx_full = index_order

    epoch_losses = []
    val_losses = []
    best_val = float('inf')
    patience_ctr = 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        np.random.shuffle(train_idx_full)
        epoch_loss = 0.0
        model.train()
        for start in range(0, len(train_idx_full), batch_size):
            end = min(start + batch_size, len(train_idx_full))
            batch_idx = train_idx_full[start:end]
            if sparse.issparse(X_train):
                batch_x_dense = X_train[batch_idx].toarray().astype('float16' if (use_half and device.type=='cuda') else 'float32')
            else:
                batch_x_dense = X_train[batch_idx].astype('float16' if (use_half and device.type=='cuda') else 'float32')
            batch_y = y_np[batch_idx]

            xb = torch.from_numpy(batch_x_dense).to(device)
            yb = torch.from_numpy(batch_y).to(device)
            if use_half and device.type == 'cuda':
                xb = xb.half()
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * (end - start)
        avg_loss = epoch_loss / len(train_idx_full)
        epoch_losses.append(avg_loss)

        # Validation
        if val_idx is not None:
            model.eval()
            v_loss_accum = 0.0
            with torch.no_grad():
                for s in range(0, len(val_idx), batch_size):
                    vi = val_idx[s:s+batch_size]
                    if sparse.issparse(X_train):
                        vx = X_train[vi].toarray().astype('float16' if (use_half and device.type=='cuda') else 'float32')
                    else:
                        vx = X_train[vi].astype('float16' if (use_half and device.type=='cuda') else 'float32')
                    vy = y_np[vi]
                    vx_t = torch.from_numpy(vx).to(device)
                    if use_half and device.type=='cuda':
                        vx_t = vx_t.half()
                    vy_t = torch.from_numpy(vy).to(device)
                    v_logits = model(vx_t).squeeze(-1)
                    v_loss = criterion(v_logits, vy_t)
                    v_loss_accum += v_loss.item() * len(vi)
            val_loss = v_loss_accum / len(val_idx)
            val_losses.append(val_loss)
            improved = val_loss < (best_val - 1e-6)
            if improved:
                best_val = val_loss
                patience_ctr = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
            if verbose:
                print(f"[Torch FFNN] Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - val_loss: {val_loss:.4f}{' *' if improved else ''}")
            if patience_ctr >= early_stop_patience:
                if verbose:
                    print(f"[Torch FFNN] Early stopping triggered (patience={early_stop_patience}). Best val_loss={best_val:.4f}")
                break
        else:
            if verbose:
                print(f"[Torch FFNN] Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    # Restore best weights if validation used
    if val_idx is not None and best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - start_time
    wrapper = TorchNewsClassifierWrapper(model, device, n_features, batch_size_predict=predict_batch_size)
    wrapper._fit_time = total_time
    wrapper.history = {'loss': epoch_losses}
    if val_idx is not None:
        wrapper.history['val_loss'] = val_losses
    if verbose:
        print(f"[Torch FFNN] Training finished in {total_time:.2f}s. Best val_loss: {best_val:.4f}" if val_idx is not None else f"[Torch FFNN] Training finished in {total_time:.2f}s.")
    return wrapper
