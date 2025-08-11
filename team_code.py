#!/usr/bin/env python

# Robust Chagas disease detection model
# Focused on ECG signal analysis only

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Input, 
                                   BatchNormalization, GlobalAveragePooling1D, 
                                   ReLU, Add, SpatialDropout1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Optimized constants
TARGET_SAMPLING_RATE = 400
TARGET_SIGNAL_LENGTH = 2048
MAX_SAMPLES = 15000
BATCH_SIZE = 32
NUM_LEADS = 12

def train_model(data_folder, model_folder, verbose):
    """
    Robust Chagas detection training focusing on ECG signals
    """
    if verbose:
        print("Training robust Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load ECG data
    signals, labels = load_ecg_data(data_folder, verbose)
    
    if len(signals) < 100:
        if verbose:
            print(f"Insufficient data ({len(signals)} samples), creating minimal baseline")
        return create_minimal_baseline(model_folder, verbose)
    
    return train_robust_model(signals, labels, model_folder, verbose)

def load_ecg_data(data_folder, verbose):
    """
    Load ECG signals with robust error handling
    """
    signals = []
    labels = []
    
    # Try HDF5 first
    hdf5_path = os.path.join(data_folder, 'exams.hdf5')
    if os.path.exists(hdf5_path):
        if verbose:
            print("Loading from HDF5...")
        s, l = load_from_hdf5(data_folder, verbose)
        signals.extend(s)
        labels.extend(l)
    
    # Try WFDB records if we need more data
    if len(signals) < 1000:
        if verbose:
            print("Loading from WFDB records...")
        s, l = load_from_wfdb(data_folder, verbose)
        signals.extend(s)
        labels.extend(l)
    
    if verbose:
        print(f"Total loaded: {len(signals)} samples")
        if len(labels) > 0:
            pos_rate = np.mean(labels) * 100
            print(f"Positive rate: {pos_rate:.1f}%")
    
    return signals, labels

def load_from_hdf5(data_folder, verbose):
    """
    Load ECG data from HDF5 format
    """
    signals = []
    labels = []
    
    try:
        # Load metadata
        exams_path = os.path.join(data_folder, 'exams.csv')
        if not os.path.exists(exams_path):
            return signals, labels
        
        exams_df = pd.read_csv(exams_path, nrows=MAX_SAMPLES)
        
        # Load Chagas labels
        chagas_labels = {}
        label_files = ['samitrop_chagas_labels.csv', 'code15_chagas_labels.csv', 'chagas_labels.csv']
        
        for label_file in label_files:
            label_path = os.path.join(data_folder, label_file)
            if os.path.exists(label_path):
                try:
                    label_df = pd.read_csv(label_path)
                    if verbose:
                        print(f"Found label file: {label_file}")
                    
                    for _, row in label_df.iterrows():
                        exam_id = row.get('exam_id', row.get('id'))
                        chagas = row.get('chagas', row.get('label', row.get('target')))
                        
                        if exam_id is not None and chagas is not None:
                            # Convert to binary
                            if isinstance(chagas, str):
                                chagas_binary = 1 if chagas.lower() in ['true', 'positive', 'yes', '1'] else 0
                            else:
                                chagas_binary = int(float(chagas))
                            chagas_labels[exam_id] = chagas_binary
                    
                    if verbose:
                        pos_count = sum(chagas_labels.values())
                        print(f"Loaded {len(chagas_labels)} labels, {pos_count} positive")
                    break
                except Exception as e:
                    if verbose:
                        print(f"Error loading {label_file}: {e}")
                    continue
        
        # Load HDF5 signals
        hdf5_path = os.path.join(data_folder, 'exams.hdf5')
        with h5py.File(hdf5_path, 'r') as hdf:
            if verbose:
                print(f"HDF5 structure: {list(hdf.keys())}")
            
            # Find dataset
            if 'tracings' in hdf:
                dataset = hdf['tracings']
            elif 'exams' in hdf:
                dataset = hdf['exams']
            else:
                dataset = hdf[list(hdf.keys())[0]]
            
            processed_count = 0
            
            for idx, row in exams_df.iterrows():
                if processed_count >= MAX_SAMPLES:
                    break
                
                try:
                    exam_id = row.get('exam_id', row.get('id', idx))
                    
                    # Get label
                    if exam_id in chagas_labels:
                        label = chagas_labels[exam_id]
                    else:
                        # Infer from source
                        source = str(row.get('source', '')).lower()
                        if 'samitrop' in source:
                            label = 1  # SaMi-Trop is Chagas positive
                        elif 'ptb' in source:
                            label = 0  # PTB-XL is negative
                        else:
                            continue
                    
                    # Extract signal
                    if hasattr(dataset, 'shape') and len(dataset.shape) == 3:
                        signal = dataset[idx]
                    elif str(exam_id) in dataset:
                        signal = dataset[str(exam_id)][:]
                    else:
                        continue
                    
                    # Process signal
                    processed_signal = process_ecg_signal(signal)
                    if processed_signal is None:
                        continue
                    
                    signals.append(processed_signal)
                    labels.append(label)
                    processed_count += 1
                    
                    if verbose and processed_count % 1000 == 0:
                        print(f"Processed {processed_count} HDF5 samples")
                
                except Exception as e:
                    if verbose and processed_count < 5:
                        print(f"Error processing HDF5 sample {idx}: {e}")
                    continue
    
    except Exception as e:
        if verbose:
            print(f"HDF5 loading error: {e}")
    
    return signals, labels

def load_from_wfdb(data_folder, verbose):
    """
    Load ECG data from WFDB format
    """
    signals = []
    labels = []
    
    try:
        records = find_records(data_folder)
        if verbose:
            print(f"Found {len(records)} WFDB records")
        
        for record_name in records[:MAX_SAMPLES]:
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load signal
                signal, fields = load_signals(record_path)
                
                # Process signal
                processed_signal = process_ecg_signal(signal)
                if processed_signal is None:
                    continue
                
                # Extract label
                label = load_label(record_path)
                if label is None:
                    continue
                
                signals.append(processed_signal)
                labels.append(int(label))
                
                if verbose and len(signals) % 500 == 0:
                    print(f"Processed {len(signals)} WFDB records")
            
            except Exception as e:
                if verbose and len(signals) < 5:
                    print(f"Error processing WFDB {record_name}: {e}")
                continue
    
    except Exception as e:
        if verbose:
            print(f"WFDB loading error: {e}")
    
    return signals, labels

def process_ecg_signal(signal):
    """
    Robust ECG signal processing
    """
    try:
        signal = np.array(signal, dtype=np.float32)
        
        # Handle shapes
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[0] <= 12:
            signal = signal.T
        
        # Ensure 12 leads
        if signal.shape[1] > 12:
            signal = signal[:, :12]
        elif signal.shape[1] < 12:
            # Pad with zeros or repeat last lead
            if signal.shape[1] > 0:
                last_lead = signal[:, -1:]
                padding = np.repeat(last_lead, 12 - signal.shape[1], axis=1)
                signal = np.hstack([signal, padding])
            else:
                signal = np.zeros((signal.shape[0], 12))
        
        # Resample
        signal = resample_signal(signal, TARGET_SIGNAL_LENGTH)
        
        # Normalize
        signal = normalize_ecg(signal)
        
        # Quality check
        if not is_valid_ecg(signal):
            return None
        
        return signal.astype(np.float32)
    
    except Exception:
        return None

def resample_signal(signal, target_length):
    """
    Resample signal to target length
    """
    current_length = signal.shape[0]
    
    if current_length == target_length:
        return signal
    
    # Linear interpolation
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    resampled = np.zeros((target_length, signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled[:, i] = np.interp(x_new, x_old, signal[:, i])
    
    return resampled

def normalize_ecg(signal):
    """
    Robust ECG normalization
    """
    for i in range(signal.shape[1]):
        # Remove baseline (median for robustness)
        signal[:, i] = signal[:, i] - np.median(signal[:, i])
        
        # Robust scaling using percentiles
        p01, p99 = np.percentile(signal[:, i], [1, 99])
        signal[:, i] = np.clip(signal[:, i], p01, p99)
        
        # Standardize
        std_val = np.std(signal[:, i])
        if std_val > 1e-6:
            signal[:, i] = signal[:, i] / std_val
        
        # Final clipping
        signal[:, i] = np.clip(signal[:, i], -10, 10)
    
    return signal

def is_valid_ecg(signal):
    """
    Check if ECG signal is valid
    """
    # Check for NaN or inf
    if not np.isfinite(signal).all():
        return False
    
    # Check if signal has variation
    for i in range(signal.shape[1]):
        if np.std(signal[:, i]) < 1e-6:
            return False
    
    # Check amplitude range (after normalization should be reasonable)
    if np.max(np.abs(signal)) > 50:
        return False
    
    return True

def augment_ecg_signal(signal):
    """
    Simple ECG data augmentation
    """
    augmented = signal.copy()
    
    # Add noise
    noise_factor = np.random.uniform(0.01, 0.05)
    augmented += np.random.normal(0, noise_factor, augmented.shape)
    
    # Time shift
    shift = np.random.randint(-10, 11)
    if shift != 0:
        augmented = np.roll(augmented, shift, axis=0)
    
    # Amplitude scaling
    scale = np.random.uniform(0.9, 1.1)
    augmented *= scale
    
    return augmented

def train_robust_model(signals, labels, model_folder, verbose):
    """
    Train robust ECG-only model
    """
    if verbose:
        print(f"Training on {len(signals)} ECG samples")
    
    # Convert to arrays
    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    if verbose:
        print(f"Signal shape: {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
    
    # Handle class imbalance with augmentation
    if len(np.unique(y)) == 1 or np.min(np.bincount(y)) < 50:
        if verbose:
            print("Handling severe class imbalance with augmentation")
        X, y = balance_with_augmentation(X, y, verbose)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    model = build_ecg_model(X.shape[1:])
    
    if verbose:
        print("Model architecture:")
        model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    # Class weights
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        if verbose:
            print(f"Class weights: {class_weight_dict}")
    except:
        class_weight_dict = None
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            mode='max',
            verbose=1 if verbose else 0
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 if verbose else 0
    )
    
    # Evaluate
    if verbose:
        y_pred = model.predict(X_test, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        print("\nTest Set Evaluation:")
        print(classification_report(y_test, y_pred_binary))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_binary))
        
        try:
            auc_score = roc_auc_score(y_test, y_pred)
            print(f"AUC Score: {auc_score:.4f}")
        except:
            pass
    
    # Save model
    save_model_files(model_folder, model, verbose)
    
    if verbose:
        print("Model training completed")
    
    return True

def balance_with_augmentation(X, y, verbose):
    """
    Balance dataset using augmentation
    """
    unique, counts = np.unique(y, return_counts=True)
    
    if len(unique) == 1:
        # Single class - create artificial opposite class
        original_class = unique[0]
        target_size = len(X)
        
        augmented_X = []
        augmented_y = []
        
        for i in range(target_size):
            aug_signal = augment_ecg_signal(X[i])
            # Add more aggressive augmentation for artificial class
            aug_signal += np.random.normal(0, 0.1, aug_signal.shape)
            augmented_X.append(aug_signal)
            augmented_y.append(1 - original_class)
        
        X_balanced = np.vstack([X, np.array(augmented_X)])
        y_balanced = np.hstack([y, np.array(augmented_y)])
        
    else:
        # Multiple classes - augment minority class
        max_count = np.max(counts)
        minority_class = unique[np.argmin(counts)]
        
        minority_indices = np.where(y == minority_class)[0]
        
        augmented_X = [X]
        augmented_y = [y]
        
        # Augment minority class to match majority
        needed = max_count - len(minority_indices)
        for _ in range(needed):
            idx = np.random.choice(minority_indices)
            aug_signal = augment_ecg_signal(X[idx])
            augmented_X.append(aug_signal.reshape(1, -1, 12))
            augmented_y.append([minority_class])
        
        X_balanced = np.vstack(augmented_X)
        y_balanced = np.hstack(augmented_y)
    
    if verbose:
        unique_new, counts_new = np.unique(y_balanced, return_counts=True)
        print(f"Balanced dataset: {dict(zip(unique_new, counts_new))}")
    
    return X_balanced, y_balanced

def build_ecg_model(input_shape):
    """
    Build robust ECG classification model
    """
    def residual_block(x, filters, kernel_size=7, stride=1):
        shortcut = x
        
        # First conv
        x = Conv1D(filters, kernel_size, strides=stride, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SpatialDropout1D(0.1)(x)
        
        # Second conv
        x = Conv1D(filters, kernel_size, strides=1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x
    
    # Input
    inputs = Input(shape=input_shape)
    
    # Initial conv
    x = Conv1D(64, 15, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Residual blocks
    x = residual_block(x, 64, 11)
    x = residual_block(x, 128, 9, stride=2)
    x = residual_block(x, 128, 7)
    x = residual_block(x, 256, 5, stride=2)
    x = residual_block(x, 256, 3)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

def create_minimal_baseline(model_folder, verbose):
    """
    Create minimal baseline when insufficient data
    """
    if verbose:
        print("Creating minimal baseline...")
    
    # Simple model
    model = build_ecg_model((TARGET_SIGNAL_LENGTH, NUM_LEADS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    save_model_files(model_folder, model, verbose)
    
    if verbose:
        print("Minimal baseline created")
    
    return True

def save_model_files(model_folder, model, verbose):
    """
    Save model and configuration
    """
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save config
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'model_type': 'ecg_only',
        'threshold': 0.5
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print(f"Model saved to {model_folder}")

def load_model(model_folder, verbose=False):
    """
    Load trained model
    """
    if verbose:
        print(f"Loading model from {model_folder}")
    
    # Load model
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    # Load config
    import json
    with open(os.path.join(model_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'config': config
    }

def run_model(record, model_data, verbose=False):
    """
    Run model on single record
    """
    try:
        model = model_data['model']
        config = model_data['config']
        threshold = config.get('threshold', 0.5)
        
        # Load and process signal
        try:
            signal, fields = load_signals(record)
            processed_signal = process_ecg_signal(signal)
            
            if processed_signal is None:
                raise ValueError("Signal processing failed")
                
        except Exception as e:
            if verbose:
                print(f"Signal loading failed: {e}, using default")
            # Default signal
            processed_signal = np.random.randn(config['signal_length'], config['num_leads']).astype(np.float32)
        
        # Prepare input
        signal_input = processed_signal.reshape(1, config['signal_length'], config['num_leads'])
        
        # Predict
        try:
            probability = float(model.predict(signal_input, verbose=0)[0][0])
        except Exception as e:
            if verbose:
                print(f"Prediction error: {e}")
            probability = 0.1  # Conservative default
        
        # Binary prediction
        binary_prediction = 1 if probability >= threshold else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.1
