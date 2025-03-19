#!/usr/bin/env python

# SaMi-Trop ECG Chagas Disease Detection using CNN model

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Build a CNN model for ECG classification
def build_cnn_model(input_shape, num_classes=1):
    """
    Build a basic CNN model for ECG classification.
    """
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv1D(filters=32, kernel_size=5, activation='relu', 
                padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # Second convolutional block
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # Third convolutional block
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # Fourth convolutional block
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    else:
        outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class
    
    model = Model(inputs, outputs)
    
    # Compile model
    if num_classes == 1:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), 
                    tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    else:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def load_raw_data(df, sampling_rate, data_path):
    """
    Load raw ECG data from HDF5 file for the SaMi-Trop dataset.
    Enhanced with additional error handling and format detection.
    """
    print(f"Loading ECG signals from {data_path}...")
    
    # First check if the file exists
    if not os.path.exists(data_path):
        print(f"ERROR: File does not exist: {data_path}")
        return np.array([]), np.array([])
    
    ecg_data = []
    loaded_ids = []
    
    try:
        with h5py.File(data_path, 'r') as hdf:
            # Print dataset structure to help with debugging
            print(f"HDF5 file structure:")
            print(f"Top-level keys: {list(hdf.keys())}")
            
            # Try the original 'tracings' key format
            if 'tracings' in hdf:
                print("Found 'tracings' key - using SaMi-Trop format")
                # Load all tracings at once
                all_tracings = np.array(hdf['tracings'])
                print(f"Loaded tracings with shape: {all_tracings.shape}")
                
                # Determine number of records
                n_records = all_tracings.shape[0]
                print(f"Number of ECG records: {n_records}")
                
                # Generate sequential IDs if needed
                if df is not None and 'exam_id' in df.columns and len(df) == n_records:
                    exam_ids = df['exam_id'].values
                    print("Using exam IDs from metadata DataFrame")
                else:
                    # For SaMi-Trop, we want to use sequential IDs starting from 1
                    exam_ids = [i+1 for i in range(n_records)]
                    print("Using generated sequential exam IDs (1 to n)")
                
                # Loop through each record
                for i in range(n_records):
                    # Display progress
                    if (i+1) % 100 == 0 or i == n_records - 1:
                        print(f"Processed {i+1}/{n_records} records")
                    
                    try:
                        # Get the ECG for this record
                        signal = all_tracings[i]
                        
                        # Check if we have 12-lead ECG
                        if len(signal.shape) > 1 and signal.shape[1] == 12:
                            # Standardize to a fixed length if needed
                            target_length = 4000  # 10 seconds at 400Hz
                            
                            if signal.shape[0] > target_length:
                                # Trim to target length
                                signal = signal[:target_length, :]
                            elif signal.shape[0] < target_length:
                                # Pad with zeros to target length
                                padding = np.zeros((target_length - signal.shape[0], signal.shape[1]))
                                signal = np.vstack((signal, padding))
                            
                            # Add to dataset
                            ecg_data.append(signal)
                            
                            # Get the exam ID
                            exam_id = exam_ids[i]
                            loaded_ids.append(exam_id)
                    except Exception as e:
                        print(f"Error processing record {i}: {e}")
                        continue
            
            # Try alternative known formats
            elif 'ecg' in hdf:
                print("Found 'ecg' key - using alternative format")
                all_tracings = np.array(hdf['ecg'])
                print(f"Loaded ECG data with shape: {all_tracings.shape}")
                
                # Process similar to 'tracings' format
                n_records = all_tracings.shape[0]
                print(f"Number of ECG records: {n_records}")
                
                # Generate sequential IDs
                exam_ids = [i+1 for i in range(n_records)]
                
                # Loop through each record
                for i in range(n_records):
                    if (i+1) % 100 == 0 or i == n_records - 1:
                        print(f"Processed {i+1}/{n_records} records")
                    
                    try:
                        signal = all_tracings[i]
                        
                        # Adapt to the data shape
                        if len(signal.shape) == 1:
                            # Handle single-lead ECG
                            signal = signal.reshape(-1, 1)
                        
                        # Check and standardize length
                        target_length = 4000
                        if signal.shape[0] > target_length:
                            signal = signal[:target_length, :]
                        elif signal.shape[0] < target_length:
                            padding = np.zeros((target_length - signal.shape[0], signal.shape[1]))
                            signal = np.vstack((signal, padding))
                        
                        ecg_data.append(signal)
                        loaded_ids.append(i+1)
                    except Exception as e:
                        print(f"Error processing record {i}: {e}")
                        continue
            else:
                # Try standard HDF5 format where each key is an exam ID
                exam_ids = list(hdf.keys())
                print(f"Found {len(exam_ids)} records in HDF5 file")
                
                # If no recognizable structure, try to inspect and adapt
                if len(exam_ids) == 0:
                    print("No recognizable format found. File structure:")
                    def print_attrs(name, obj):
                        print(f"  - {name}: {type(obj)}")
                        if isinstance(obj, h5py.Dataset):
                            print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                    hdf.visititems(print_attrs)
                else:
                    # Process using exam IDs as keys
                    for i, exam_id in enumerate(exam_ids):
                        if (i+1) % 100 == 0:
                            print(f"Loaded {i+1}/{len(exam_ids)} records")
                        
                        try:
                            # Get ECG data
                            signal = np.array(hdf[exam_id])
                            
                            # Ensure 2D array (samples x leads)
                            if len(signal.shape) == 1:
                                signal = signal.reshape(-1, 1)
                            
                            # Check if shape makes sense for ECG data
                            if signal.shape[1] <= 12:  # Up to 12 leads
                                # Standardize to a fixed length
                                target_length = 4000
                                if signal.shape[0] > target_length:
                                    signal = signal[:target_length, :]
                                elif signal.shape[0] < target_length:
                                    padding = np.zeros((target_length - signal.shape[0], signal.shape[1]))
                                    signal = np.vstack((signal, padding))
                                
                                ecg_data.append(signal)
                                loaded_ids.append(exam_id)
                        except Exception as e:
                            print(f"Error loading ECG for exam_id {exam_id}: {e}")
                            continue
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
    
    print(f"Successfully loaded {len(ecg_data)} ECG records with IDs")
    
    return np.array(ecg_data), np.array(loaded_ids)

def clean_scp_codes(dicts):
    """
    Placeholder function to match team_code.py structure.
    For SaMi-Trop, we don't have SCP codes, so this is just a pass-through.
    """
    return dicts

def create_chagas_related_labels(loaded_ids, chagas_labels_path):
    """
    Create binary labels for SaMi-Trop dataset.
    For each record, assign 1 (positive) if the ID is in the chagas_labels_path CSV,
    otherwise assign 0 (negative).
    """
    # Initialize all as negative
    labels = np.zeros(len(loaded_ids), dtype=int)
    
    try:
        # Load positive Chagas labels
        labels_df = pd.read_csv(chagas_labels_path)
        positive_ids = set(labels_df['exam_id'].values)
        print(f"Loaded {len(positive_ids)} positive Chagas cases from labels file")
        
        # Mark cases as positive if they are in the positive_ids set
        for i, exam_id in enumerate(loaded_ids):
            try:
                # Convert exam_id to int if it's a string
                exam_id_int = int(exam_id) if isinstance(exam_id, str) else exam_id
                if exam_id_int in positive_ids:
                    labels[i] = 1
            except (ValueError, TypeError):
                # If conversion fails, just continue (keeping as negative)
                continue
    except Exception as e:
        print(f"Error loading Chagas labels: {e}")
    
    # Count positive cases
    positive_count = np.sum(labels)
    print(f"Identified {positive_count} positive Chagas cases out of {len(labels)} records")
    
    # Verify that we found close to 815 positive cases
    if positive_count > 0 and positive_count < 800:
        print("Warning: Expected around 815 positive cases but only found {positive_count}.")
        print("Check if exam IDs in the dataset match those in the labels file.")
    
    return labels

def prepare_data(X, y):
    """
    Prepare ECG data for model training.
    """
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Get dimensions
    n_samples, n_timesteps, n_features = X_train.shape

    # Reshape for standardization (combine time and leads)
    X_train_flat = X_train.reshape(n_samples, n_timesteps * n_features)
    X_val_flat = X_val.reshape(X_val.shape[0], n_timesteps * n_features)

    # Standardize
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    # Reshape back to original dimensions
    X_train = X_train_flat.reshape(n_samples, n_timesteps, n_features)
    X_val = X_val_flat.reshape(X_val.shape[0], n_timesteps, n_features)

    return X_train, X_val, y_train, y_val, scaler

def preprocess_ecg(X, scaler):
    """
    Preprocess ECG data using saved scaler.
    """
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape(n_samples, n_timesteps * n_features)
    X_scaled = scaler.transform(X_flat)
    X_reshaped = X_scaled.reshape(n_samples, n_timesteps, n_features)
    return X_reshaped

def train_model(data_directory, model_directory, verbose=1):
    """
    Train model using data in data_directory and save trained model in model_directory.
    
    Args:
        data_directory: Directory containing the input data files.
        model_directory: Directory where the model will be saved.
        verbose: Level of verbosity (0: silent, 1: normal, 2: detailed).
    
    This function is required by the challenge.
    """
    if verbose >= 1:
        print('Finding challenge data...')
    
    # Define possible paths for HDF5 file
    possible_hdf5_paths = [
        os.path.join(data_directory, 'exams.hdf5'),
        os.path.join(data_directory, 'data', 'exams.hdf5'),
        os.path.join(data_directory, 'ecg_data.hdf5'),
        os.path.join(data_directory, 'data', 'ecg_data.hdf5'),
        # Add more potential paths if needed
    ]
    
    # Find the actual HDF5 file
    hdf5_path = None
    for path in possible_hdf5_paths:
        if os.path.exists(path):
            hdf5_path = path
            if verbose >= 1:
                print(f"Found HDF5 file at: {path}")
            break
    
    if hdf5_path is None:
        # List all files in the data directory to help with debugging
        print(f"Error: HDF5 file not found. Looking in paths: {possible_hdf5_paths}")
        print(f"Contents of data_directory '{data_directory}':")
        for root, dirs, files in os.walk(data_directory):
            print(f"  Directory: {root}")
            for file in files:
                print(f"    File: {file}")
                
        # Try to list files that might contain ECG data
        print("Searching for potential ECG data files:")
        for root, dirs, files in os.walk(data_directory):
            for file in files:
                if file.endswith('.hdf5') or file.endswith('.h5') or file.endswith('.mat'):
                    print(f"  Potential data file found: {os.path.join(root, file)}")
        
        # Try to continue with default path, but it will likely fail
        hdf5_path = os.path.join(data_directory, 'exams.hdf5')
    
    # Define possible paths for labels file
    chagas_labels_path = os.path.join(data_directory, 'data', 'samitrop_chagas_labels.csv')
    
    # Check if labels file exists in specified path, if not try alternative paths
    if not os.path.exists(chagas_labels_path):
        alternative_paths = [
            os.path.join(data_directory, 'samitrop_chagas_labels.csv'),
            os.path.join(data_directory, 'labels', 'samitrop_chagas_labels.csv'),
            os.path.join(data_directory, 'chagas_labels.csv'),
            # Add more paths if needed
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                chagas_labels_path = alt_path
                if verbose >= 1:
                    print(f"Found labels file at alternative path: {alt_path}")
                break
        
        if not os.path.exists(chagas_labels_path):
            print(f"Warning: Labels file not found. Checked paths: {alternative_paths}")
            
            # Try to find any CSV file that might contain labels
            print("Searching for potential labels files:")
            for root, dirs, files in os.walk(data_directory):
                for file in files:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        print(f"  Potential labels file found: {csv_path}")
                        
                        # Try to peek at the file to see if it contains relevant columns
                        try:
                            df_peek = pd.read_csv(csv_path, nrows=5)
                            print(f"    Columns: {list(df_peek.columns)}")
                            if 'exam_id' in df_peek.columns:
                                print(f"    This file contains 'exam_id' column and might be useful")
                                if not chagas_labels_path or not os.path.exists(chagas_labels_path):
                                    chagas_labels_path = csv_path
                        except Exception as e:
                            print(f"    Error peeking at CSV: {e}")
    
    if verbose >= 1:
        print(f"Using labels file: {chagas_labels_path}")
        print(f"Using HDF5 file: {hdf5_path}")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_directory, exist_ok=True)
    
    if verbose >= 1:
        print('Loading ECG data...')
    
    # Load raw ECG data
    X, loaded_ids = load_raw_data(None, None, hdf5_path)
    
    if len(X) == 0:
        print("Error: No ECG data loaded. Please check the file paths.")
        return
    
    # Create labels for the loaded IDs
    if verbose >= 1:
        print('Creating Chagas labels...')
    y = create_chagas_related_labels(loaded_ids, chagas_labels_path)
    
    if verbose >= 1:
        print('Preparing data...')
    # Prepare and standardize data
    X_train, X_val, y_train, y_val, scaler = prepare_data(X, y)
    
    # Save the scaler for use during inference
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if verbose >= 1:
        print('Training model...')
    # Build and train the model
    model = build_cnn_model(input_shape)
    
    if verbose >= 2:
        # Print model summary with higher verbosity
        model.summary()
    
    # Define callbacks
    callbacks = []
    
    # Always use early stopping for best performance
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1 if verbose >= 1 else 0
    )
    callbacks.append(early_stopping)
    
    # Add model checkpoint to save the best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_directory, 'best_model.h5'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1 if verbose >= 1 else 0
    )
    callbacks.append(model_checkpoint)
    
    # Add learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1 if verbose >= 1 else 0
    )
    callbacks.append(reduce_lr)
    
    # Calculate class weights for imbalanced data
    class_weight = None
    if np.sum(y_train == 0) > 0 and np.sum(y_train == 1) > 0:
        weight_for_0 = 1.0
        weight_for_1 = np.sum(y_train == 0) / np.sum(y_train == 1)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        if verbose >= 1:
            print(f"Using class weights: {class_weight}")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1 if verbose >= 1 else 0
    )
    
    if verbose >= 1:
        print('Saving model...')
    # Save the final model
    model.save(os.path.join(model_directory, 'chagas_cnn_model.h5'))
    
    # Plot and save training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_directory, 'training_history.png'))
    
    # Evaluate on validation set
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Print and save classification report
    if verbose >= 1:
        print("\nValidation Set Performance:")
        report = classification_report(y_val, y_pred)
        print(report)
        
        with open(os.path.join(model_directory, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    if verbose >= 1:
        print('Done training model.')

def load_model(model_directory, verbose=1):
    """
    Load trained model from model_directory.
    
    Args:
        model_directory: Directory where the model is saved.
        verbose: Level of verbosity (0: silent, 1: normal, 2: detailed).
    
    Returns:
        A tuple containing the model and scaler.
    
    This function is required by the challenge.
    """
    if verbose >= 1:
        print('Loading model...')
    
    # Load the keras model - first try best model, then fall back to final model
    best_model_path = os.path.join(model_directory, 'best_model.h5')
    final_model_path = os.path.join(model_directory, 'chagas_cnn_model.h5')
    
    if os.path.exists(best_model_path):
        if verbose >= 1:
            print(f"Loading best model from: {best_model_path}")
        model = keras_load_model(best_model_path)
    elif os.path.exists(final_model_path):
        if verbose >= 1:
            print(f"Loading final model from: {final_model_path}")
        model = keras_load_model(final_model_path)
    else:
        raise FileNotFoundError(f"No model found in {model_directory}")
    
    # Load the scaler parameters
    scaler_mean_path = os.path.join(model_directory, 'scaler_mean.npy')
    scaler_scale_path = os.path.join(model_directory, 'scaler_scale.npy')
    
    if verbose >= 1:
        print(f"Loading scaler from: {scaler_mean_path} and {scaler_scale_path}")
    
    scaler_mean = np.load(scaler_mean_path)
    scaler_scale = np.load(scaler_scale_path)
    
    # Recreate the scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    if verbose >= 2:
        print("Model summary:")
        model.summary()
        print(f"Scaler mean shape: {scaler_mean.shape}")
        print(f"Scaler scale shape: {scaler_scale.shape}")
    
    # Return both the model and scaler
    return model, scaler

def run_model(model, data, recordings, verbose=1):
    """
    Run trained model on data.
    
    Args:
        model: Trained model and scaler tuple.
        data: Patient data (ignored in this implementation).
        recordings: ECG recordings to analyze.
        verbose: Level of verbosity (0: silent, 1: normal, 2: detailed).
    
    Returns:
        Tuple containing binary predictions and probability scores.
    
    This function is required by the challenge.
    """
    # Unpack the model and scaler
    keras_model, scaler = model
    
    if verbose >= 2:
        print(f"Input recordings shape: {recordings.shape}")
    
    # Preprocess the ECG data
    X_processed = preprocess_ecg(recordings, scaler)
    
    if verbose >= 2:
        print(f"Processed recordings shape: {X_processed.shape}")
    
    # Make predictions
    y_pred_prob = keras_model.predict(X_processed, verbose=0 if verbose < 2 else 1)
    
    # Convert to binary prediction (threshold at 0.5)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    if verbose >= 2:
        print(f"Prediction counts: Positive={np.sum(y_pred)}, Negative={len(y_pred) - np.sum(y_pred)}")
        print(f"Probability range: Min={np.min(y_pred_prob):.4f}, Max={np.max(y_pred_prob):.4f}, Mean={np.mean(y_pred_prob):.4f}")
    
    # Return both binary prediction and probability
    return y_pred.flatten(), y_pred_prob.flatten()

# If run as a script
if __name__ == '__main__':
    # Example usage:
    # python team_code.py <data_directory> <model_directory>
    import sys
    
    if len(sys.argv) != 3:
        print('Usage: python team_code.py <data_directory> <model_directory>')
    else:
        data_directory = sys.argv[1]
        model_directory = sys.argv[2]
        train_model(data_directory, model_directory)
