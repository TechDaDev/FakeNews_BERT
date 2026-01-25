import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os
from pathlib import Path

# Add root directory calculation
root_dir = Path(__file__).resolve().parent.parent.parent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU setup
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(physical_gpus)} GPU(s): {[g.name for g in physical_gpus]}")
    except RuntimeError as e:
        print("GPU setup error:", e)
else:
    print("No GPU found. Running on CPU.")

# Load the CSV file into a DataFrame
data_path = root_dir / 'data' / 'combined_TF_data.csv'
data = pd.read_csv(data_path)

# 1. Data Preprocessing
X = data["text"]
y = data["label"]

# Label handling (dataset spec: 0 = Fake, 1 = Real). Support string variants too.
if y.dtype == 'O':
    y_norm = y.astype(str).str.lower().str.strip()
    y_bin = y_norm.isin(['1', 'real', 'true']).astype(int)
else:
    y_bin = y.astype(int)
print('Label distribution (after mapping, 0=Fake,1=Real):')
print(y_bin.value_counts().sort_index())

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.25, random_state=42, stratify=y_bin)

# 2. Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token='[OOV]')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_len = 500
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# 3. Build the LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

# 4. Train the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=30, batch_size=64, validation_split=0.2)

# Saving the trained model
save_dir = root_dir / 'saved_models'
os.makedirs(save_dir, exist_ok=True)
model.save(save_dir / "lstm_model1.h5")

# Saving the tokenizer
joblib.dump(tokenizer, save_dir / "tokenizerDL1.pkl")

# Plotting training history
plt.figure(figsize=(12, 4))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
