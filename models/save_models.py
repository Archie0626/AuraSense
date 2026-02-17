"""
Script to train and save models for HAR system
"""
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import HARPreprocessor, FeatureEngineer
from utils.feature_engineering import AdvancedFeatureEngineering

def create_models_dir():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')
    print("✓ Models directory created/verified")

def train_and_save_ml_models(X_train, y_train, X_test, y_test):
    """Train and save traditional ML models"""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Save model
        joblib.dump(model, f'models/{name}_model.pkl')
        print(f"  ✓ {name} saved (Accuracy: {accuracy:.4f})")
    
    # Save label encoder and scaler
    return results

def build_cnn_model(input_shape, num_classes):
    """Build CNN model"""
    model = keras.Sequential([
        keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),

        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),

        keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling1D(),

        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_lstm_model(input_shape, num_classes):
    """Build LSTM model"""
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_dl_models(X_train_seq, y_train_seq, X_test_seq, y_test_seq, num_classes):
    """Train and save deep learning models"""
    input_shape_cnn = (X_train_seq.shape[1], X_train_seq.shape[2])
    input_shape_lstm = (X_train_seq.shape[1], X_train_seq.shape[2])
    
    results = {}
    
    # Train and save CNN
    print("Training CNN model...")
    cnn_model = build_cnn_model(input_shape_cnn, num_classes)
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = cnn_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_seq, y_test_seq, verbose=0)
    results['cnn'] = cnn_accuracy
    
    # Save model
    cnn_model.save('models/cnn_model.h5')
    print(f"  ✓ CNN saved (Accuracy: {cnn_accuracy:.4f})")
    
    # Train and save LSTM
    print("Training LSTM model...")
    lstm_model = build_lstm_model(input_shape_lstm, num_classes)
    
    history = lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate
    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
    results['lstm'] = lstm_accuracy
    
    # Save model
    lstm_model.save('models/lstm_model.h5')
    print(f"  ✓ LSTM saved (Accuracy: {lstm_accuracy:.4f})")
    
    return results

def create_synthetic_data_for_demo():
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    n_samples = 10000
    n_features = 50  # Reduced for demo
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 6, n_samples)
    
    return X, y

def main():
    print("=" * 50)
    print("HAR Model Training Script")
    print("=" * 50)
    
    # Create models directory
    create_models_dir()
    
    # Initialize preprocessor
    preprocessor = HARPreprocessor()
    feature_engineer = FeatureEngineer()
    advanced_fe = AdvancedFeatureEngineering()
    
    # Load data
    print("\n📊 Loading data...")
    X, y, subjects, activity_labels = preprocessor.load_uci_har_data()
    
    if X is None:
        print("⚠️  Using synthetic data for demonstration...")
        X, y = create_synthetic_data_for_demo()
        activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                          'SITTING', 'STANDING', 'LAYING']
    
    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Activities: {len(np.unique(y))} classes")
    
    # Preprocess data
    print("\n🔄 Preprocessing data...")
    X_scaled = preprocessor.normalize_data(X)
    y_encoded = preprocessor.encode_labels(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Train and save ML models
    print("\n🤖 Training ML models...")
    ml_results = train_and_save_ml_models(X_train, y_train, X_test, y_test)
    
    # Create sequences for deep learning
    print("\n🔧 Creating sequences for deep learning...")
    seq_length = 128
    
    # Reshape data for sequences (simulate time series)
    n_sequences = len(X_train) // seq_length
    X_train_seq = X_train[:n_sequences*seq_length].reshape(n_sequences, seq_length, -1)
    y_train_seq = y_train[:n_sequences*seq_length:seq_length]
    
    n_sequences_test = len(X_test) // seq_length
    X_test_seq = X_test[:n_sequences_test*seq_length].reshape(n_sequences_test, seq_length, -1)
    y_test_seq = y_test[:n_sequences_test*seq_length:seq_length]
    
    print(f"✓ Sequences created: {X_train_seq.shape}")
    
    # Train and save DL models
    print("\n🧠 Training Deep Learning models...")
    num_classes = len(np.unique(y))
    dl_results = train_and_save_dl_models(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, num_classes
    )
    
    # Save scaler and label encoder
    print("\n💾 Saving preprocessors...")
    joblib.dump(preprocessor.scaler, 'models/scaler.pkl')
    joblib.dump(preprocessor.label_encoder, 'models/label_encoder.pkl')
    print("✓ Scaler saved")
    print("✓ Label encoder saved")
    
    # Save activity labels
    with open('models/activity_labels.txt', 'w') as f:
        for label in activity_labels:
            f.write(f"{label}\n")
    print("✓ Activity labels saved")
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 Training Summary")
    print("=" * 50)
    print("\nML Models:")
    for name, accuracy in ml_results.items():
        print(f"  {name}: {accuracy:.4f}")
    
    print("\nDL Models:")
    for name, accuracy in dl_results.items():
        print(f"  {name}: {accuracy:.4f}")
    
    print("\n" + "=" * 50)
    print("✅ All models saved successfully in 'models/' directory!")
    print("=" * 50)

if __name__ == "__main__":
    main()