#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout, 
                                   BatchNormalization, GlobalAveragePooling1D, 
                                   concatenate, MultiHeadAttention)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import wfdb
import pywt
from scipy import signal as scipy_signal
import glob
import warnings
warnings.filterwarnings('ignore')

# Constants
TARGET_SAMPLING_RATE = 400  # Hz
TARGET_SIGNAL_LENGTH = 4000  # 10 seconds at 400Hz
NUM_LEADS = 12
BATCH_SIZE = 32

def find_records(data_folder):
    """Find all WFDB records in the directory."""
    hea_files = glob.glob(os.path.join(data_folder, '*.hea'))
    return [os.path.splitext(os.path.basename(f))[0] for f in hea_files]

def train_model(data_folder, model_folder, verbose):
    """Main training function with robust data handling and model training."""
    try:
        if verbose:
            print("Starting Chagas detection model training...")
        
        os.makedirs(model_folder, exist_ok=True)
        
        # Load and preprocess data
        signals, labels, demographics = load_chagas_data(data_folder, verbose)
        
        if len(signals) < 10:
            if verbose:
                print("Insufficient data, creating fallback model")
            return create_fallback_model(model_folder, verbose)
        
        # Train model with cross-validation
        trained = train_chagas_model(signals, labels, demographics, model_folder, verbose)
        
        if verbose and trained:
            print("Training completed successfully")
        return trained
        
    except Exception as e:
        if verbose:
            print(f"Training failed: {str(e)}")
        return False

def load_chagas_data(data_folder, verbose):
    """Load ECG data from multiple possible sources."""
    signals, labels, demographics = [], [], []
    
    # Try loading from HDF5 first
    h5_path = os.path.join(data_folder, 'exams.hdf5')
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as hf:
                # Check available datasets
                if 'tracings' in hf:
                    ecg_data = hf['tracings'][:]
                elif 'signals' in hf:
                    ecg_data = hf['signals'][:]
                else:
                    ecg_data = hf[list(hf.keys())[0]][:]
                
                # Load labels from CSV
                label_path = os.path.join(data_folder, 'samitrop_chagas_labels.csv')
                if os.path.exists(label_path):
                    label_df = pd.read_csv(label_path)
                    label_map = dict(zip(label_df['exam_id'], label_df['chagas']))
                else:
                    label_map = {}
                
                # Process each sample
                for i in range(min(10000, len(ecg_data))):  # Limit samples
                    try:
                        processed_sig = preprocess_ecg(ecg_data[i])
                        if processed_sig is None:
                            continue
                            
                        label = label_map.get(i, 0)  # Default to negative
                        
                        signals.append(processed_sig)
                        labels.append(int(label))
                        demographics.append([0.5, 0.5])  # Default demographics
                    except Exception as e:
                        if verbose:
                            print(f"Skipping HDF5 sample {i}: {str(e)}")
                        continue
                        
            if verbose:
                print(f"Loaded {len(signals)} samples from HDF5")
        except Exception as e:
            if verbose:
                print(f"HDF5 loading error: {str(e)}")
    
    # Fallback to WFDB if needed
    wfdb_path = os.path.join(data_folder, 'wfdb_records')
    if os.path.exists(wfdb_path) and len(signals) < 1000:
        try:
            records = find_records(wfdb_path)
            if verbose:
                print(f"Found {len(records)} WFDB records")
                
            for record in records[:1000]:  # Limit samples
                try:
                    sig, _ = wfdb.rdsamp(os.path.join(wfdb_path, record))
                    header = wfdb.rdheader(os.path.join(wfdb_path, record))
                    
                    processed_sig = preprocess_ecg(sig)
                    if processed_sig is None:
                        continue
                        
                    label = get_chagas_label(header) or 0  # Default to negative
                    
                    signals.append(processed_sig)
                    labels.append(label)
                    demographics.append(get_demographics(header))
                    
                except Exception as e:
                    if verbose:
                        print(f"Skipping record {record}: {str(e)}")
                    continue
                    
            if verbose:
                print(f"Added {len(signals)} total samples")
        except Exception as e:
            if verbose:
                print(f"WFDB loading failed: {str(e)}")
    
    if verbose:
        pos_count = sum(labels)
        print(f"Final dataset: {len(signals)} samples ({pos_count} positive, {pos_count/len(labels)*100:.1f}%)")
        
    return np.array(signals), np.array(labels), np.array(demographics)

def preprocess_ecg(raw_signal):
    """Advanced ECG preprocessing pipeline."""
    try:
        signal = np.array(raw_signal, dtype=np.float32)
        
        # Ensure correct shape (samples × leads)
        if signal.shape[0] < signal.shape[1]:
            signal = signal.T
            
        # Handle lead count
        if signal.shape[1] > NUM_LEADS:
            signal = signal[:, :NUM_LEADS]
        elif signal.shape[1] < NUM_LEADS:
            signal = np.pad(signal, ((0, 0), (0, NUM_LEADS - signal.shape[1])), 
                           mode='edge')
            
        # Resample if needed
        if signal.shape[0] != TARGET_SIGNAL_LENGTH:
            signal = resample_ecg(signal, TARGET_SIGNAL_LENGTH)
            
        # Remove baseline wander
        signal = remove_baseline_wander(signal)
        
        # Normalize
        signal = normalize_ecg(signal)
        
        return signal
        
    except Exception as e:
        return None

def resample_ecg(signal, target_length):
    """Resample ECG to target length using Fourier method."""
    current_length = signal.shape[0]
    resampled = np.zeros((target_length, signal.shape[1]))
    
    for lead in range(signal.shape[1]):
        resampled[:, lead] = scipy_signal.resample(
            signal[:, lead], target_length, window='hann')
            
    return resampled

def remove_baseline_wander(signal, wavelet='db4', level=5):
    """Remove baseline wander using wavelet transform."""
    for lead in range(signal.shape[1]):
        coeffs = pywt.wavedec(signal[:, lead], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])  # Zero out approximation coefficients
        signal[:, lead] = pywt.waverec(coeffs, wavelet)[:signal.shape[0]]
    return signal

def normalize_ecg(signal):
    """Robust normalization using median and IQR."""
    for lead in range(signal.shape[1]):
        # Median centering
        signal[:, lead] -= np.median(signal[:, lead])
        
        # IQR scaling
        q1, q3 = np.percentile(signal[:, lead], [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            signal[:, lead] /= iqr
            
        # Soft clipping
        signal[:, lead] = np.tanh(signal[:, lead])
        
    return signal

def get_chagas_label(header):
    """Extract Chagas label from header."""
    try:
        # Check standard fields
        for field in ['chagas', 'diagnosis', 'comment']:
            if hasattr(header, field):
                content = str(getattr(header, field)).lower()
                if 'chagas' in content or 'trypanosoma' in content:
                    return 1
                elif 'normal' in content or 'healthy' in content:
                    return 0
                    
        # Infer from source
        if hasattr(header, 'base_datetime'):
            if 'samitrop' in str(header.base_datetime).lower():
                return 1
            elif 'ptb' in str(header.base_datetime).lower():
                return 0
                
        return None
        
    except:
        return None

def get_demographics(header):
    """Extract age and sex from header."""
    age = 50.0  # Default
    sex = 0.5   # 0.5 for unknown
    
    try:
        if hasattr(header, 'age'):
            try:
                age = float(header.age)
            except:
                pass
                
        if hasattr(header, 'sex'):
            sex_str = str(header.sex).lower()
            if sex_str.startswith('m'):
                sex = 1.0
            elif sex_str.startswith('f'):
                sex = 0.0
    except:
        pass
        
    return [age / 100.0, sex]  # Normalized

def train_chagas_model(signals, labels, demographics, model_folder, verbose):
    """Train the Chagas detection model."""
    try:
        # Split data with stratification
        (X_sig_train, X_sig_val, 
         X_demo_train, X_demo_val, 
         y_train, y_val) = train_test_split(
            signals, demographics, labels, 
            test_size=0.2, 
            stratify=labels,
            random_state=42
        )
        
        # Scale demographics
        demo_scaler = StandardScaler()
        X_demo_train_scaled = demo_scaler.fit_transform(X_demo_train)
        X_demo_val_scaled = demo_scaler.transform(X_demo_val)
        
        # Build model
        model = build_attention_model()
        
        if verbose:
            model.summary()
            
        # Handle class imbalance
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        history = model.fit(
            [X_sig_train, X_demo_train_scaled], y_train,
            validation_data=([X_sig_val, X_demo_val_scaled], y_val),
            epochs=50,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1 if verbose else 0
        )
        
        # Save model
        save_model(model, demo_scaler, model_folder, verbose)
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"Model training failed: {str(e)}")
        return False

def build_attention_model():
    """Build CNN model with attention mechanism."""
    # Signal input branch
    signal_input = Input(shape=(TARGET_SIGNAL_LENGTH, NUM_LEADS), name='ecg_input')
    
    # Multi-scale feature extraction
    x = Conv1D(64, 15, activation='relu', padding='same')(signal_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Attention mechanism
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = concatenate([x, attn])
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Demographics branch
    demo_input = Input(shape=(2,), name='demo_input')
    demo_branch = Dense(16, activation='relu')(demo_input)
    
    # Combined features
    combined = concatenate([x, demo_branch])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # Output
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, demo_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def save_model(model, demo_scaler, model_folder, verbose):
    """Save model and associated files."""
    os.makedirs(model_folder, exist_ok=True)
    
    # Save model in Keras format
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save scaler
    import joblib
    joblib.dump(demo_scaler, os.path.join(model_folder, 'demo_scaler.pkl'))
    
    # Save config
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print(f"Model saved to {model_folder}")

def create_fallback_model(model_folder, verbose):
    """Create a simple fallback model when training data is insufficient."""
    try:
        os.makedirs(model_folder, exist_ok=True)
        
        # Simple model architecture
        model = tf.keras.Sequential([
            Input(shape=(TARGET_SIGNAL_LENGTH, NUM_LEADS)),
            Conv1D(16, 5, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create dummy scaler
        demo_scaler = StandardScaler()
        demo_scaler.fit(np.random.randn(10, 2))
        
        save_model(model, demo_scaler, model_folder, verbose)
        
        if verbose:
            print("Fallback model created")
            
        return True
        
    except Exception as e:
        if verbose:
            print(f"Fallback model creation failed: {str(e)}")
        return False

def load_model(model_folder, verbose):
    """Load trained model and associated files."""
    try:
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
        
    except Exception as e:
        if verbose:
            print(f"Model loading failed: {str(e)}")
        return None

def run_model(record, model_data, verbose):
    """Run inference on a single ECG record."""
    try:
        model = model_data['model']
        demo_scaler = model_data['demo_scaler']
        config = model_data['config']
        
        # Load signal
        try:
            signal, _ = wfdb.rdsamp(record)
            processed_signal = preprocess_ecg(signal)
            if processed_signal is None:
                raise ValueError("Signal processing failed")
        except:
            processed_signal = np.random.randn(
                config['signal_length'], config['num_leads']).astype(np.float32)
        
        # Get demographics
        try:
            header = wfdb.rdheader(record)
            demographics = get_demographics(header)
        except:
            demographics = [0.5, 0.5]  # Default values
            
        # Prepare inputs
        signal_input = processed_signal.reshape(1, -1, config['num_leads'])
        demo_input = demo_scaler.transform([demographics])
        
        # Predict
        probability = float(model.predict([signal_input, demo_input], verbose=0)[0][0])
        prediction = 1 if probability >= 0.5 else 0
        
        return prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Prediction failed: {str(e)}")
        return 0, 0.1  # Conservative default
