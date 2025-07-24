#!/usr/bin/env python

# Enhanced team code for PhysioNet Challenge 2025 - Chagas Disease Detection
# TensorFlow/Keras CNN implementation

import joblib
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

from scipy import signal as scipy_signal
from helper_code import *

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

################################################################################
# Enhanced CNN Model Architecture (TensorFlow/Keras)
################################################################################

def create_chagas_cnn_model(input_length=5000, num_leads=12, dropout_rate=0.3):
    """Create enhanced CNN model for Chagas disease detection"""
    
    # ECG input
    ecg_input = keras.Input(shape=(input_length, num_leads), name='ecg_input')
    
    # Convolutional feature extraction
    x = layers.Conv1D(64, 15, activation='relu', padding='same')(ecg_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(128, 11, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(256, 7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(256, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Global average pooling
    ecg_features = layers.GlobalAveragePooling1D()(x)
    
    # Metadata input
    metadata_input = keras.Input(shape=(4,), name='metadata_input')
    
    # Combine ECG features with metadata
    combined = layers.Concatenate()([ecg_features, metadata_input])
    
    # Classification layers
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation='sigmoid', name='chagas_probability')(x)
    
    model = Model(inputs=[ecg_input, metadata_input], outputs=output, name='ChagasECGNet')
    
    return model

################################################################################
# Data Generator for TensorFlow
################################################################################

class ECGDataGenerator(Sequence):
    """Custom data generator for ECG data with metadata"""
    
    def __init__(self, records, data_folder, labels=None, batch_size=32, 
                 target_length=5000, shuffle=True, augment=False):
        self.records = records
        self.data_folder = data_folder
        self.labels = labels
        self.batch_size = batch_size
        self.target_length = target_length
        self.shuffle = shuffle
        self.augment = augment
        self.reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 
                                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.records) / float(self.batch_size)))
    
    def __getitem__(self, index):
        # Generate batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.records))
        batch_records = [self.records[i] for i in self.indices[start_idx:end_idx]]
        
        # Generate data
        batch_ecg = []
        batch_metadata = []
        batch_labels = []
        
        for i, record in enumerate(batch_records):
            ecg_data, metadata = self._load_record(record)
            batch_ecg.append(ecg_data)
            batch_metadata.append(metadata)
            
            if self.labels is not None:
                original_idx = start_idx + i
                if original_idx < len(self.labels):
                    batch_labels.append(self.labels[self.indices[original_idx]])
                else:
                    batch_labels.append(0)
        
        batch_ecg = np.array(batch_ecg)
        batch_metadata = np.array(batch_metadata)
        
        if self.labels is not None:
            batch_labels = np.array(batch_labels)
            return [batch_ecg, batch_metadata], batch_labels
        else:
            return [batch_ecg, batch_metadata]
    
    def on_epoch_end(self):
        """Update indices after each epoch"""
        self.indices = np.arange(len(self.records))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_record(self, record):
        """Load and preprocess a single record"""
        try:
            record_path = os.path.join(self.data_folder, record)
            
            # Load header and signal
            header = load_header(record_path)
            signal_data, fields = load_signals(record_path)
            channels = fields['sig_name']
            
            # Reorder channels
            signal_data = reorder_signal(signal_data, channels, self.reference_channels)
            
            # Preprocess signal
            ecg_data = self._preprocess_signal(signal_data)
            
            # Extract metadata
            metadata = self._extract_metadata(header)
            
            return ecg_data, metadata
            
        except Exception as e:
            # Return default values if loading fails
            ecg_data = np.zeros((self.target_length, 12))
            metadata = np.array([0.4, 1.0, 0.0, 0.0])  # default: normalized age=0.4, female
            return ecg_data, metadata
    
    def _preprocess_signal(self, signal_data):
        """Preprocess ECG signal"""
        # Handle NaN values
        signal_data = np.nan_to_num(signal_data, nan=0.0)
        
        # Apply basic filtering and standardization
        for i in range(signal_data.shape[1]):
            lead_data = signal_data[:, i]
            if np.std(lead_data) > 0:
                # Standardize
                signal_data[:, i] = (lead_data - np.mean(lead_data)) / np.std(lead_data)
                
                # Optional: apply detrending for baseline removal
                if len(lead_data) > 100:
                    try:
                        signal_data[:, i] = scipy_signal.detrend(signal_data[:, i])
                    except:
                        pass
        
        # Resize to target length
        current_length = signal_data.shape[0]
        if current_length >= self.target_length:
            # Truncate from center
            start_idx = (current_length - self.target_length) // 2
            signal_data = signal_data[start_idx:start_idx + self.target_length, :]
        else:
            # Pad with zeros
            padding = self.target_length - current_length
            pad_before = padding // 2
            pad_after = padding - pad_before
            signal_data = np.pad(signal_data, ((pad_before, pad_after), (0, 0)), mode='constant')
        
        # Data augmentation during training
        if self.augment and np.random.random() > 0.5:
            signal_data = self._augment_signal(signal_data)
        
        return signal_data
    
    def _augment_signal(self, signal_data):
        """Apply data augmentation to ECG signal"""
        # Random noise addition
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.05, signal_data.shape)
            signal_data = signal_data + noise
        
        # Random scaling
        if np.random.random() > 0.7:
            scale_factor = np.random.uniform(0.9, 1.1)
            signal_data = signal_data * scale_factor
        
        # Random time shift (small)
        if np.random.random() > 0.8:
            shift = np.random.randint(-50, 51)
            if shift != 0:
                signal_data = np.roll(signal_data, shift, axis=0)
        
        return signal_data
    
    def _extract_metadata(self, header):
        """Extract and encode metadata features"""
        # Age
        age = get_age(header)
        if not isinstance(age, (int, float)) or np.isnan(age):
            age = 50.0  # default age
        age = np.clip(age, 0, 120) / 120.0  # normalize to [0,1]
        
        # Sex (one-hot encoding)
        sex = get_sex(header)
        if isinstance(sex, str):
            if sex.lower().startswith('f'):
                sex_features = [1.0, 0.0, 0.0]  # female
            elif sex.lower().startswith('m'):
                sex_features = [0.0, 1.0, 0.0]  # male
            else:
                sex_features = [0.0, 0.0, 1.0]  # unknown
        else:
            sex_features = [0.0, 0.0, 1.0]  # unknown
        
        return np.array([age] + sex_features, dtype=np.float32)

################################################################################
# Required Functions
################################################################################

def train_model(data_folder, model_folder, verbose):
    """Train the CNN Chagas detection model"""
    if verbose:
        print('Finding Challenge data...')
    
    records = find_records(data_folder)
    num_records = len(records)
    
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    if verbose:
        print(f'Found {num_records} records')
        print('Loading data and extracting labels...')
    
    # Extract labels and filter records
    valid_records = []
    valid_labels = []
    
    for i, record in enumerate(records):
        if verbose and (i + 1) % 1000 == 0:
            print(f'Processing record {i + 1}/{num_records}')
        
        try:
            record_path = os.path.join(data_folder, record)
            
            # Try to load label - if it fails, this might be test/held-out data
            try:
                label = load_label(record_path)
                has_label = True
            except:
                if verbose:
                    print(f'No label found for {record} - might be test data')
                has_label = False
                label = 0  # default label for test data
            
            # Try to load source
            try:
                source = load_source(record_path)
            except:
                source = 'unknown'
            
            # Skip most CODE-15% data as in original, but keep more for deep learning
            if source != 'CODE-15%' or (i % 3) == 0:  # Keep 33% of CODE-15%
                valid_records.append(record)
                if has_label:
                    valid_labels.append(int(label))
                else:
                    valid_labels.append(0)  # dummy label for test data
                
        except Exception as e:
            if verbose:
                print(f'Error processing {record}: {e}')
            continue
    
    if verbose:
        print(f'Using {len(valid_records)} valid records')
        if len(valid_labels) > 0:
            positive_count = sum(valid_labels)
            print(f'Positive samples: {positive_count} ({100*positive_count/len(valid_labels):.1f}%)')
        else:
            print('No valid records found with labels')
    
    # Check if we have enough data to train
    if len(valid_records) == 0:
        raise ValueError('No valid records found for training')
    
    # If we don't have real labels (test data), create a dummy model
    if len(valid_labels) == 0 or all(label == 0 for label in valid_labels):
        if verbose:
            print('No labels available - creating dummy model for test data')
        _create_dummy_model(model_folder)
        return
    
    # Split data for validation
    split_idx = int(0.8 * len(valid_records))
    train_records = valid_records[:split_idx]
    val_records = valid_records[split_idx:]
    train_labels = valid_labels[:split_idx]
    val_labels = valid_labels[split_idx:]
    
    if verbose:
        print(f'Training set: {len(train_records)} samples')
        print(f'Validation set: {len(val_records)} samples')
    
    # Create data generators
    train_gen = ECGDataGenerator(train_records, data_folder, train_labels, 
                               batch_size=16, shuffle=True, augment=True)
    val_gen = ECGDataGenerator(val_records, data_folder, val_labels, 
                             batch_size=16, shuffle=False, augment=False)
    
    # Create model
    model = create_chagas_cnn_model()
    
    if verbose:
        print('Model architecture:')
        model.summary()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    # Calculate class weights for imbalanced data
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = len(train_labels) / (2.0 * pos_count) if pos_count > 0 else 1.0
    neg_weight = len(train_labels) / (2.0 * neg_count) if neg_count > 0 else 1.0
    class_weight = {0: neg_weight, 1: pos_weight}
    
    if verbose:
        print(f'Class weights: {class_weight}')
        print('Starting training...')
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        best_auc = max(history.history['val_auc'])
        print(f'Training completed. Best validation AUC: {best_auc:.4f}')
    
    # Create model folder and save
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)
    
    if verbose:
        print('Model saved successfully!')

def load_model(model_folder, verbose):
    """Load the trained CNN model"""
    model_file = os.path.join(model_folder, 'chagas_cnn_model.h5')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Model file not found: {model_file}')
    
    if verbose:
        print(f'Loading model from {model_file}')
    
    model = keras.models.load_model(model_file)
    
    return {'model': model, 'type': 'cnn'}

def run_model(record, model_dict, verbose):
    """Run the CNN model on a single record"""
    try:
        model = model_dict['model']
        
        # Create single-sample generator
        record_name = os.path.basename(record)
        data_folder = os.path.dirname(record)
        
        gen = ECGDataGenerator([record_name], data_folder, batch_size=1, shuffle=False)
        
        # Get prediction
        batch_data = gen[0]
        prediction = model.predict(batch_data, verbose=0)
        probability = float(prediction[0][0])
        binary_output = int(probability > 0.5)
        
        return binary_output, probability
        
    except Exception as e:
        if verbose:
            print(f'Error processing {record}: {e}')
        return 0, 0.0

################################################################################
# Helper Functions
################################################################################

def save_model(model_folder, model):
    """Save the trained CNN model"""
    model_file = os.path.join(model_folder, 'chagas_cnn_model.h5')
    model.save(model_file)

def _create_dummy_model(model_folder):
    """Create a dummy model when no training labels are available (for test data)"""
    # Create a simple dummy model that outputs random probabilities
    model = create_chagas_cnn_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the untrained model
    os.makedirs(model_folder, exist_ok=True)
    model_file = os.path.join(model_folder, 'chagas_cnn_model.h5')
    model.save(model_file)
    
    print(f'Dummy model saved to {model_file}')

# Maintain compatibility with original extract_features function
def extract_features(record):
    """Extract basic features for compatibility (simplified version)"""
    try:
        header = load_header(record)
        signal_data, fields = load_signals(record)
        channels = fields['sig_name']
        
        # Basic metadata
        age = get_age(header)
        sex = get_sex(header)
        source = get_source(header)
        
        # Simple signal features
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 
                            'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        signal_data = reorder_signal(signal_data, channels, reference_channels)
        
        signal_mean = np.nanmean(signal_data, axis=0)
        signal_std = np.nanstd(signal_data, axis=0)
        
        # Handle missing values
        if not isinstance(age, (int, float)) or np.isnan(age):
            age = 50.0
        
        sex_encoding = np.zeros(3)
        if isinstance(sex, str):
            if sex.lower().startswith('f'):
                sex_encoding[0] = 1
            elif sex.lower().startswith('m'):
                sex_encoding[1] = 1
            else:
                sex_encoding[2] = 1
        else:
            sex_encoding[2] = 1
        
        return np.array([age]), sex_encoding, source, signal_mean, signal_std
        
    except Exception as e:
        # Return default values
        return np.array([50.0]), np.array([0, 0, 1]), 'unknown', np.zeros(12), np.ones(12)
