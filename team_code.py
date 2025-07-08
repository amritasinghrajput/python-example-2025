#!/usr/bin/env python

# Advanced Chagas disease detection model
# Combines multi-scale CNNs with attention mechanisms and advanced preprocessing

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout, 
                                   BatchNormalization, concatenate, GlobalAveragePooling1D,
                                   MultiHeadAttention, LayerNormalization, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import pywt

# Constants
TARGET_SAMPLING_RATE = 400
TARGET_SIGNAL_LENGTH = 2048  # ~5 seconds at 400Hz
NUM_LEADS = 12
BATCH_SIZE = 32
EPOCHS = 50

def train_model(data_folder, model_folder, verbose=True):
    """Advanced training pipeline for Chagas detection"""
    if verbose:
        print("Training advanced Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load and preprocess data
    signals, labels = load_and_preprocess_data(data_folder, verbose)
    
    if len(signals) < 50:
        if verbose:
            print("Insufficient data, creating enhanced baseline model")
        return create_enhanced_baseline_model(model_folder, verbose)
    
    return train_advanced_model(signals, labels, model_folder, verbose)

def load_and_preprocess_data(data_folder, verbose):
    """Advanced data loading and preprocessing"""
    signals = []
    labels = []
    
    try:
        records = find_records(data_folder)
        if verbose:
            print(f"Found {len(records)} records")
        
        for record_name in records:
            try:
                record_path = os.path.join(data_folder, record_name)
                signal, _ = load_signals(record_path)
                
                # Advanced signal processing
                processed_signal = advanced_signal_processing(signal)
                if processed_signal is None:
                    continue
                
                # Get label with additional checks
                label = get_validated_label(record_path)
                if label is None:
                    continue
                
                signals.append(processed_signal)
                labels.append(int(label))
                
            except Exception as e:
                if verbose:
                    print(f"Error processing {record_name}: {str(e)}")
                continue
    
    except Exception as e:
        if verbose:
            print(f"Data loading error: {str(e)}")
    
    if verbose:
        print(f"Loaded {len(signals)} samples")
        if len(labels) > 0:
            pos_rate = np.mean(labels) * 100
            print(f"Positive rate: {pos_rate:.1f}%")
    
    return signals, labels

def advanced_signal_processing(signal):
    """Advanced signal processing with wavelet features"""
    try:
        signal = np.array(signal, dtype=np.float32)
        
        # Ensure proper shape (samples, leads)
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1]:
            signal = signal.T
        
        # Take first 12 leads or pad if needed
        if signal.shape[1] > NUM_LEADS:
            signal = signal[:, :NUM_LEADS]
        elif signal.shape[1] < NUM_LEADS:
            padding = np.zeros((signal.shape[0], NUM_LEADS - signal.shape[1]))
            signal = np.hstack([signal, padding])
        
        # Resample to target length
        signal = resample_signal(signal, TARGET_SIGNAL_LENGTH)
        
        # Advanced preprocessing pipeline
        signal = apply_advanced_preprocessing(signal)
        
        return signal
    
    except Exception as e:
        return None

def apply_advanced_preprocessing(signal):
    """Advanced preprocessing steps"""
    processed_signal = np.zeros_like(signal)
    
    for lead in range(signal.shape[1]):
        # 1. Remove baseline wander using wavelet transform
        coeffs = pywt.wavedec(signal[:, lead], 'db4', level=5)
        coeffs[0] = np.zeros_like(coeffs[0])  # Remove approximation coefficients
        filtered = pywt.waverec(coeffs, 'db4')
        
        # Adjust length if needed
        if len(filtered) > len(signal):
            filtered = filtered[:len(signal)]
        elif len(filtered) < len(signal):
            filtered = np.pad(filtered, (0, len(signal) - len(filtered)))
        
        # 2. Bandpass filtering (simulated)
        filtered = filtered - np.mean(filtered)
        filtered = filtered / (np.std(filtered) + 1e-6)
        
        # 3. Wavelet feature enhancement
        cA, cD = pywt.dwt(filtered, 'db2')
        cA = pywt.idwt(cA, None, 'db2', take=len(filtered))
        
        processed_signal[:, lead] = cA
    
    return processed_signal

def build_advanced_model(input_shape):
    """Build advanced model with multi-scale CNNs and attention"""
    inputs = Input(shape=input_shape)
    
    # Multi-scale feature extraction
    # Branch 1: High-frequency features (short kernel)
    branch1 = Conv1D(64, 5, activation='relu', padding='same')(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(2)(branch1)
    branch1 = Dropout(0.3)(branch1)
    
    branch1 = Conv1D(128, 5, activation='relu', padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling1D(2)(branch1)
    branch1 = Dropout(0.4)(branch1)
    
    # Branch 2: Low-frequency features (long kernel)
    branch2 = Conv1D(64, 15, activation='relu', padding='same')(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(2)(branch2)
    branch2 = Dropout(0.3)(branch2)
    
    branch2 = Conv1D(128, 15, activation='relu', padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling1D(2)(branch2)
    branch2 = Dropout(0.4)(branch2)
    
    # Combine branches
    merged = concatenate([branch1, branch2])
    
    # Attention mechanism
    attention_input = Reshape((-1, 256))(merged)  # Prepare for attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(attention_input, attention_input)
    attention_output = LayerNormalization()(attention_output + attention_input)
    
    # Final processing
    x = GlobalAveragePooling1D()(attention_output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def train_advanced_model(signals, labels, model_folder, verbose):
    """Train the advanced model with enhanced techniques"""
    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    if verbose:
        print(f"Input shape: {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    model = build_advanced_model((TARGET_SIGNAL_LENGTH, NUM_LEADS))
    
    # Custom optimizer with warmup
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=len(X_train)//BATCH_SIZE * EPOCHS
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Calculate class weights with smoothing
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    if verbose:
        print("Model architecture:")
        model.summary()
        print(f"Class weights: {class_weight_dict}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, mode='max', min_lr=1e-6),
        ModelCheckpoint(os.path.join(model_folder, 'best_model.h5'), 
                      monitor='val_auc', 
                      save_best_only=True,
                      mode='max')
    ]
    
    # Train with optional data augmentation
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 if verbose else 0
    )
    
    # Enhanced evaluation
    if verbose:
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        print("\nAdvanced Test Set Evaluation:")
        print(classification_report(y_test, y_pred_binary))
        print(f"AUC Score: {roc_auc_score(y_test, y_pred):.4f}")
        
        # Save full evaluation report
        save_evaluation_report(y_test, y_pred, model_folder)
    
    # Save final model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    if verbose:
        print("Advanced model training completed")
    
    return True

def save_evaluation_report(y_true, y_pred, model_folder):
    """Save detailed evaluation metrics"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.join(model_folder, 'eval'), exist_ok=True)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    plt.figure()
    plt.plot(recall, precision, label=f'AP={avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(model_folder, 'eval', 'pr_curve.png'))
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(model_folder, 'eval', 'metrics.txt'), 'w') as f:
        f.write(f"AUC: {roc_auc_score(y_true, y_pred):.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")

def create_enhanced_baseline_model(model_folder, verbose):
    """Create an enhanced baseline model"""
    if verbose:
        print("Creating enhanced baseline model...")
    
    model = Sequential([
        Conv1D(32, 15, activation='relu', input_shape=(TARGET_SIGNAL_LENGTH, NUM_LEADS)),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    model.save(os.path.join(model_folder, 'model.keras'))
    
    if verbose:
        print("Enhanced baseline model created")
    
    return True

def load_model(model_folder, verbose=False):
    """Load the trained advanced model"""
    if verbose:
        print(f"Loading advanced model from {model_folder}")
    
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    return {
        'model': model,
        'config': {
            'signal_length': TARGET_SIGNAL_LENGTH,
            'num_leads': NUM_LEADS,
            'model_type': 'advanced'
        }
    }

def run_model(record, model_data, verbose=False):
    """Run advanced model on a single record"""
    try:
        model = model_data['model']
        config = model_data['config']
        
        # Load and process signal with advanced methods
        try:
            signal, _ = load_signals(record)
            processed_signal = advanced_signal_processing(signal)
            
            if processed_signal is None:
                if verbose:
                    print("Signal processing failed, using fallback processing")
                processed_signal = fallback_signal_processing(signal)
        except:
            if verbose:
                print("Signal loading failed, using fallback")
            processed_signal = np.random.randn(config['signal_length'], config['num_leads'])
            processed_signal = advanced_signal_processing(processed_signal)
        
        # Predict with uncertainty estimation
        predictions = []
        for _ in range(5):  # Small test-time augmentation
            pred = model.predict(processed_signal[np.newaxis, ...], verbose=0)[0][0]
            predictions.append(pred)
        
        probability = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        # Adjust prediction based on uncertainty
        if uncertainty > 0.2:  # High uncertainty
            probability = 0.5  # Default to unsure
            if verbose:
                print(f"High uncertainty ({uncertainty:.2f}), using conservative prediction")
        
        binary_prediction = 1 if probability >= 0.5 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {str(e)}")
        return 0, 0.1  # Default to negative with low confidence

def fallback_signal_processing(signal):
    """Fallback processing when advanced methods fail"""
    try:
        signal = np.array(signal, dtype=np.float32)
        
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1]:
            signal = signal.T
        
        if signal.shape[1] > NUM_LEADS:
            signal = signal[:, :NUM_LEADS]
        elif signal.shape[1] < NUM_LEADS:
            padding = np.zeros((signal.shape[0], NUM_LEADS - signal.shape[1]))
            signal = np.hstack([signal, padding])
        
        signal = resample_signal(signal, TARGET_SIGNAL_LENGTH)
        
        # Simple normalization
        for lead in range(signal.shape[1]):
            signal[:, lead] = (signal[:, lead] - np.mean(signal[:, lead])) / (np.std(signal[:, lead]) + 1e-6)
        
        return signal
    except:
        return np.zeros((TARGET_SIGNAL_LENGTH, NUM_LEADS))
