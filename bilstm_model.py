import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("dataset/KDDTrain+.txt", header=None)

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty_level"
]

df.columns = columns
df.drop("difficulty_level", axis=1, inplace=True)

# Attack mapping
attack_mapping = {
    'normal': 'normal',
    'neptune': 'dos', 'back': 'dos', 'land': 'dos',
    'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
    'satan': 'probe', 'ipsweep': 'probe',
    'nmap': 'probe', 'portsweep': 'probe',
    'guess_passwd': 'r2l', 'ftp_write': 'r2l',
    'imap': 'r2l', 'phf': 'r2l',
    'buffer_overflow': 'u2r', 'rootkit': 'u2r'
}

df['attack_type'] = df['label'].map(lambda x: attack_mapping.get(x, 'other'))
df.drop('label', axis=1, inplace=True)

# Encode categorical
encoders = {}
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['attack_type'])

# Features
X = df.drop('attack_type', axis=1).values

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to categorical AFTER split
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Model
model = Sequential([
    Bidirectional(LSTM(64), input_shape=(1, X_train.shape[2])),
    Dense(32, activation='relu'),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train_cat, epochs=5, batch_size=64)

# ✅ Evaluate Accuracy
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\n✅ BiLSTM Accuracy: {accuracy * 100:.2f}%\n")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Reports
print("📊 Classification Report:")
print(classification_report(y_test, y_pred_classes))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# Save model
model.save("bilstm_model.h5")

# Save preprocessors (only if not already saved)
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("\n💾 BiLSTM model trained and saved!")