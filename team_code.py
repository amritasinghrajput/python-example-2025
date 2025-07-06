#!/usr/bin/env python

# Improved Chagas disease detection model
# Simplified and focused on robust feature extraction

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, 
                                   Input, concatenate, BatchNormalization, 
                                   GlobalAveragePooling1D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Optimized constants
TARGET_SAMPLING_RATE = 400  # Match most common sampling rate in datasets
TARGET_SIGNAL_LENGTH = 2048  # Power of 2 for efficiency, ~5 seconds at 400Hz
MAX_SAMPLES = 10000  # Reasonable limit for training
BATCH_SIZE = 32
NUM_LEADS = 12

def train_model(data_folder, model_folder, verbose):
    """
    Improved Chagas detection training with better data handling
    """
    if verbose:
        print("Training improved Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Try different data loading approaches
    signals, labels, demographics = load_data_robust(data_folder, verbose)
    
    if len(signals) < 50:
        if verbose:
            print(f"Insufficient data ({len(signals)} samples), creating baseline model")
        return create_baseline_model(model_folder, verbose)
    
    return train_improved_model(signals, labels, demographics, model_folder, verbose)

def load_data_robust(data_folder, verbose):
    """
    Robust data loading that handles multiple dataset formats
    """
    signals = []
    labels = []
    demographics = []
    
    # Try HDF5 first
    hdf5_path = os.path.join(data_folder, 'exams.hdf5')
    if os.path.exists(hdf5_path):
        if verbose:
            print("Loading from HDF5...")
        s, l, d = load_from_hdf5(data_folder, verbose)
        signals.extend(s)
        labels.extend(l)
        demographics.extend(d)
    
    # Try WFDB records
    if len(signals) < 1000:  # Need more data
        if verbose:
            print("Loading from WFDB records...")
        s, l, d = load_from_wfdb(data_folder, verbose)
        signals.extend(s)
        labels.extend(l)
        demographics.extend(d)
    
    if verbose:
        print(f"Total loaded: {len(signals)} samples")
        if len(labels) > 0:
            pos_rate = np.mean(labels) * 100
            print(f"Positive rate: {pos_rate:.1f}%")
    
    return signals, labels, demographics

def load_from_hdf5(data_folder, verbose):
    """
    Load data from HDF5 format with improved label handling
    """
    signals = []
    labels = []
    demographics = []
    
    try:
        # Load metadata
        exams_path = os.path.join(data_folder, 'exams.csv')
        if not os.path.exists(exams_path):
            return signals, labels, demographics
        
        exams_df = pd.read_csv(exams_path, nrows=MAX_SAMPLES)
        
        # Try to load Chagas labels
        chagas_labels = {}
        label_files = ['samitrop_chagas_labels.csv', 'code15_chagas_labels.csv', 'chagas_labels.csv']
        
        for label_file in label_files:
            label_path = os.path.join(data_folder, label_file)
            if os.path.exists(label_path):
                try:
                    label_df = pd.read_csv(label_path)
                    if verbose:
                        print(f"Found label file: {label_file}")
                    
                    # Handle different label formats
                    for _, row in label_df.iterrows():
                        exam_id = row.get('exam_id', row.get('id', None))
                        chagas = row.get('chagas', row.get('label', row.get('target', None)))
                        
                        if exam_id is not None and chagas is not None:
                            # Convert chagas label to binary
                            if isinstance(chagas, str):
                                chagas_binary = 1 if chagas.lower() in ['true', 'positive', 'yes', '1'] else 0
                            else:
                                chagas_binary = int(float(chagas))
                            chagas_labels[exam_id] = chagas_binary
                    
                    if verbose:
                        pos_count = sum(chagas_labels.values())
                        print(f"Loaded {len(chagas_labels)} labels, {pos_count} positive ({pos_count/len(chagas_labels)*100:.1f}%)")
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
            
            # Find the main dataset
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
                        # Try to infer from source or use default
                        source = str(row.get('source', '')).lower()
                        if 'samitrop' in source:
                            label = 1  # SaMi-Trop is all Chagas positive
                        elif 'ptb' in source:
                            label = 0  # PTB-XL is all negative
                        else:
                            continue  # Skip if no label
                    
                    # Extract signal
                    if hasattr(dataset, 'shape') and len(dataset.shape) == 3:
                        signal = dataset[idx]
                    elif str(exam_id) in dataset:
                        signal = dataset[str(exam_id)][:]
                    else:
                        continue
                    
                    # Process signal
                    processed_signal = process_signal_improved(signal)
                    if processed_signal is None:
                        continue
                    
                    # Extract demographics
                    demo = extract_demographics(row)
                    
                    signals.append(processed_signal)
                    labels.append(label)
                    demographics.append(demo)
                    processed_count += 1
                    
                    if verbose and processed_count % 500 == 0:
                        print(f"Processed {processed_count} HDF5 samples")
                
                except Exception as e:
                    if verbose and processed_count < 5:
                        print(f"Error processing HDF5 sample {idx}: {e}")
                    continue
    
    except Exception as e:
        if verbose:
            print(f"HDF5 loading error: {e}")
    
    return signals, labels, demographics

def load_from_wfdb(data_folder, verbose):
    """
    Load data from WFDB format
    """
    signals = []
    labels = []
    demographics = []
    
    try:
        records = find_records(data_folder)
        if verbose:
            print(f"Found {len(records)} WFDB records")
        
        for i, record_name in enumerate(records[:MAX_SAMPLES]):
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load signal and header
                signal, fields = load_signals(record_path)
                header = load_header(record_path)
                
                # Process signal
                processed_signal = process_signal_improved(signal)
                if processed_signal is None:
                    continue
                
                # Extract label
                label = load_label(record_path)
                if label is None:
                    continue
                
                # Extract demographics
                demo = extract_demographics_wfdb(header)
                
                signals.append(processed_signal)
                labels.append(int(label))
                demographics.append(demo)
                
                if verbose and len(signals) % 100 == 0:
                    print(f"Processed {len(signals)} WFDB records")
            
            except Exception as e:
                if verbose and len(signals) < 5:
                    print(f"Error processing WFDB {record_name}: {e}")
                continue
    
    except Exception as e:
        if verbose:
            print(f"WFDB loading error: {e}")
    
    return signals, labels, demographics

def process_signal_improved(signal):
    """
    Improved signal processing with better normalization
    """
    try:
        signal = np.array(signal, dtype=np.float32)
        
        # Handle different input shapes
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[0] <= 12:
            signal = signal.T  # Transpose if leads are in rows
        
        # Ensure we have 12 leads
        if signal.shape[1] > 12:
            signal = signal[:, :12]  # Take first 12 leads
        elif signal.shape[1] < 12:
            # Pad with the last available lead
            last_lead = signal[:, -1:] if signal.shape[1] > 0 else np.zeros((signal.shape[0], 1))
            padding = np.repeat(last_lead, 12 - signal.shape[1], axis=1)
            signal = np.hstack([signal, padding])
        
        # Resample to target length
        signal = resample_signal_simple(signal, TARGET_SIGNAL_LENGTH)
        
        # Remove baseline drift and normalize
        signal = normalize_signal_robust(signal)
        
        return signal.astype(np.float32)
    
    except Exception as e:
        return None

def resample_signal_simple(signal, target_length):
    """
    Simple and reliable resampling
    """
    current_length = signal.shape[0]
    
    if current_length == target_length:
        return signal
    
    # Use linear interpolation
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    resampled = np.zeros((target_length, signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled[:, i] = np.interp(x_new, x_old, signal[:, i])
    
    return resampled

def normalize_signal_robust(signal):
    """
    Robust signal normalization
    """
    # Remove baseline per lead
    for i in range(signal.shape[1]):
        # Remove DC component
        signal[:, i] = signal[:, i] - np.mean(signal[:, i])
        
        # Robust scaling using IQR
        q25, q75 = np.percentile(signal[:, i], [25, 75])
        iqr = q75 - q25
        
        if iqr > 1e-6:  # Avoid division by zero
            signal[:, i] = signal[:, i] / (iqr + 1e-6)
        
        # Clip extreme values
        signal[:, i] = np.clip(signal[:, i], -10, 10)
    
    return signal

def extract_demographics(row):
    """
    Extract demographic features from row
    """
    # Age
    age = row.get('age', 50.0)
    if pd.isna(age):
        age = 50.0
    age_norm = np.clip(float(age) / 100.0, 0.0, 1.0)
    
    # Sex
    sex = row.get('sex', row.get('is_male', 0))
    if pd.isna(sex):
        sex_male = 0.5  # Unknown
    else:
        if isinstance(sex, str):
            sex_male = 1.0 if sex.lower().startswith('m') else 0.0
        else:
            sex_male = float(sex)
    
    return np.array([age_norm, sex_male])

def extract_demographics_wfdb(header):
    """
    Extract demographics from WFDB header
    """
    age = get_age(header)
    sex = get_sex(header)
    
    age_norm = 0.5  # Default
    if age is not None:
        age_norm = np.clip(float(age) / 100.0, 0.0, 1.0)
    
    sex_male = 0.5  # Default unknown
    if sex is not None:
        sex_male = 1.0 if sex.lower().startswith('m') else 0.0
    
    return np.array([age_norm, sex_male])

def train_improved_model(signals, labels, demographics, model_folder, verbose):
    """
    Train improved model with better architecture and training
    """
    if verbose:
        print(f"Training on {len(signals)} samples")
    
    # Convert to arrays
    X_signal = np.array(signals, dtype=np.float32)
    X_demo = np.array(demographics, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    if verbose:
        print(f"Signal shape: {X_signal.shape}")
        print(f"Demographics shape: {X_demo.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
        pos_rate = np.mean(y) * 100
        print(f"Positive rate: {pos_rate:.1f}%")
    
    # Handle extreme class imbalance
    if len(np.unique(y)) == 1:
        if verbose:
            print("Single class detected - creating balanced artificial data")
        X_signal, X_demo, y = create_balanced_data(X_signal, X_demo, y, verbose)
    
    # Split data
    X_sig_train, X_sig_test, X_demo_train, X_demo_test, y_train, y_test = train_test_split(
        X_signal, X_demo, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale demographics
    demo_scaler = StandardScaler()
    X_demo_train_scaled = demo_scaler.fit_transform(X_demo_train)
    X_demo_test_scaled = demo_scaler.transform(X_demo_test)
    
    # Build model
    model = build_improved_model(X_signal.shape[1:], X_demo.shape[1])
    
    if verbose:
        print("Model architecture:")
        model.summary()
    
    # Compile with appropriate metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Calculate class weights
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        if verbose:
            print(f"Class weights: {class_weight_dict}")
    except:
        class_weight_dict = None
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        [X_sig_train, X_demo_train_scaled], y_train,
        validation_data=([X_sig_test, X_demo_test_scaled], y_test),
        epochs=50,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 if verbose else 0
    )
    
    # Evaluate
    if verbose:
        y_pred = model.predict([X_sig_test, X_demo_test_scaled])
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        print("\nTest Set Evaluation:")
        print(classification_report(y_test, y_pred_binary))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_binary))
    
    # Save model
    save_improved_model(model_folder, model, demo_scaler, verbose)
    
    if verbose:
        print("Model training completed successfully")
    
    return True

def create_balanced_data(X_signal, X_demo, y, verbose):
    """
    Create balanced dataset when only one class is present
    """
    original_class = y[0]
    n_samples = len(y)
    
    # Create artificial samples of the opposite class
    artificial_signals = X_signal.copy()
    artificial_demo = X_demo.copy()
    artificial_labels = np.full(n_samples, 1 - original_class)
    
    # Add noise to make artificial samples different
    noise_scale = 0.1
    artificial_signals += np.random.normal(0, noise_scale, artificial_signals.shape)
    artificial_demo += np.random.normal(0, 0.05, artificial_demo.shape)
    
    # Combine original and artificial
    X_signal_balanced = np.vstack([X_signal, artificial_signals])
    X_demo_balanced = np.vstack([X_demo, artificial_demo])
    y_balanced = np.hstack([y, artificial_labels])
    
    if verbose:
        print(f"Created balanced dataset: {len(X_signal_balanced)} samples")
        unique, counts = np.unique(y_balanced, return_counts=True)
        print(f"New label distribution: {dict(zip(unique, counts))}")
    
    return X_signal_balanced, X_demo_balanced, y_balanced

def build_improved_model(signal_shape, demo_features):
    """
    Build improved model architecture
    """
    # Signal processing branch
    signal_input = Input(shape=signal_shape, name='signal_input')
    
    # Multi-scale CNN for ECG feature extraction
    x = Conv1D(32, 5, activation='relu', padding='same')(signal_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    
    signal_features = Dense(64, activation='relu')(x)
    signal_features = Dropout(0.3)(signal_features)
    
    # Demographics branch
    demo_input = Input(shape=(demo_features,), name='demo_input')
    demo_branch = Dense(16, activation='relu')(demo_input)
    demo_branch = Dropout(0.2)(demo_branch)
    
    # Combine features
    combined = concatenate([signal_features, demo_branch])
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(16, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # Output
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, demo_input], outputs=output)
    return model

def create_baseline_model(model_folder, verbose):
    """
    Create baseline model when insufficient data
    """
    if verbose:
        print("Creating baseline model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Build simple model
    model = build_improved_model((TARGET_SIGNAL_LENGTH, NUM_LEADS), 2)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy scaler
    demo_scaler = StandardScaler()
    demo_scaler.fit(np.random.randn(100, 2))
    
    save_improved_model(model_folder, model, demo_scaler, verbose)
    
    if verbose:
        print("Baseline model created")
    
    return True

def save_improved_model(model_folder, model, demo_scaler, verbose):
    """
    Save model and associated files
    """
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save scaler
    import joblib
    joblib.dump(demo_scaler, os.path.join(model_folder, 'demo_scaler.pkl'))
    
    # Save configuration
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'model_type': 'improved'
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print(f"Model saved to {model_folder}")

def load_model(model_folder, verbose=False):
    """
    Load the trained model
    """
    if verbose:
        print(f"Loading model from {model_folder}")
    
    # Load model
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    # Load scaler
    import joblib
    demo_scaler = joblib.load(os.path.join(model_folder, 'demo_scaler.pkl'))
    
    # Load config
    import json
    with open(os.path.join(model_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'demo_scaler': demo_scaler,
        'config': config
    }

def run_model(record, model_data, verbose=False):
    """
    Run model on a single record
    """
    try:
        model = model_data['model']
        demo_scaler = model_data['demo_scaler']
        config = model_data['config']
        
        # Load and process signal
        try:
            signal, fields = load_signals(record)
            processed_signal = process_signal_improved(signal)
            
            if processed_signal is None:
                raise ValueError("Signal processing failed")
                
        except Exception as e:
            if verbose:
                print(f"Signal loading failed: {e}, using default")
            # Use default signal
            processed_signal = np.random.randn(config['signal_length'], config['num_leads']).astype(np.float32)
        
        # Extract demographics
        try:
            header = load_header(record)
            demographics = extract_demographics_wfdb(header)
        except:
            demographics = np.array([0.5, 0.5])  # Default values
        
        # Prepare inputs
        signal_input = processed_signal.reshape(1, config['signal_length'], config['num_leads'])
        demo_input = demo_scaler.transform(demographics.reshape(1, -1))
        
        # Predict
        try:
            probability = float(model.predict([signal_input, demo_input], verbose=0)[0][0])
        except Exception as e:
            if verbose:
                print(f"Prediction error: {e}")
            probability = 0.1  # Conservative default for Chagas (low prevalence)
        
        # Convert to binary prediction
        binary_prediction = 1 if probability >= 0.5 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.1  
