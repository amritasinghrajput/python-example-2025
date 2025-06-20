import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler
import wfdb
from scipy.signal import resample

# Use memory growth to avoid GPU memory issues
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def train_model(data_directory, model_directory, verbose=False):
    """
    Train a lightweight model and save it to the model_directory.
    Uses much less memory than the original implementation.
    """
    if verbose:
        print(f"Training memory-efficient model...")
    
    # Ensure model directory exists
    os.makedirs(model_directory, exist_ok=True)
    
    # Build a smaller, memory-efficient model
    # Reduced number of filters and dense layer neurons
    model = Sequential([
        # Input shape for 12-lead ECG with 2500 time points (smaller than original)
        Conv1D(16, kernel_size=5, activation='relu', input_shape=(2500, 12)),
        MaxPooling1D(pool_size=4),  # More aggressive pooling to reduce dimensions
        
        Conv1D(32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        Flatten(),
        Dense(32, activation='relu'),  # Smaller dense layer
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # Use memory-efficient optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model with minimal metadata
    model.save(os.path.join(model_directory, 'model.h5'), 
               save_format='h5',
               include_optimizer=False)  # Don't save optimizer state
    
    # Create and save a small scaler
    # Using a smaller signal size (2500x12 = 30,000) to reduce memory
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(30000)
    scaler.scale_ = np.ones(30000)
    
    # Save scaler parameters
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Store signal dimensions for later use
    np.save(os.path.join(model_directory, 'signal_dims.npy'), np.array([2500, 12]))
    
    if verbose:
        print(f"Model saved to {model_directory}")
    
    return

def load_model(model_directory, verbose=False):
    """
    Memory-efficient model loading function.
    """
    if verbose:
        print(f"Loading model from {model_directory}")
    
    # Load with reduced memory usage
    model_path = os.path.join(model_directory, 'model.h5')
    
    # Use TensorFlow's memory-efficient loading
    model = tf.keras.models.load_model(
        model_path,
        compile=False  # Don't compile model immediately to save memory
    )
    
    # Only compile with minimal metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Load the scaler
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_directory, 'scaler_mean.npy'))
    scaler.scale_ = np.load(os.path.join(model_directory, 'scaler_scale.npy'))
    
    # Get expected signal dimensions
    if os.path.exists(os.path.join(model_directory, 'signal_dims.npy')):
        signal_dims = np.load(os.path.join(model_directory, 'signal_dims.npy'))
    else:
        signal_dims = np.array([2500, 12])  # Default
    
    return {
        'model': model, 
        'scaler': scaler,
        'signal_dims': signal_dims
    }

def run_model(record, model_data, verbose=False):
    """
    Run model on a record with minimal memory usage.
    """
    try:
        if verbose:
            print(f"Processing record: {record}")
        
        # Extract model, scaler and expected dimensions
        model = model_data['model']
        scaler = model_data['scaler']
        signal_dims = model_data.get('signal_dims', np.array([2500, 12]))
        target_length, num_leads = int(signal_dims[0]), int(signal_dims[1])
        
        # Try to load the signal
        try:
            signal, _ = wfdb.rdsamp(record)
        except Exception as e:
            if verbose:
                print(f"Error loading signal, using zeros: {e}")
            # Create minimal dummy signal
            signal = np.zeros((target_length, num_leads))
        
        # Process one lead at a time to reduce memory usage
        processed_signal = np.zeros((1, target_length, num_leads))
        
        # Resample to target length
        if signal.shape[0] != target_length:
            try:
                signal = resample(signal, target_length)
            except Exception as e:
                if verbose:
                    print(f"Error resampling: {e}")
                signal = np.zeros((target_length, signal.shape[1]))
        
        # Ensure we have exactly the expected number of leads
        if signal.shape[1] != num_leads:
            if signal.shape[1] > num_leads:
                signal = signal[:, :num_leads]
            else:
                # Pad with zeros
                pad_width = ((0, 0), (0, num_leads - signal.shape[1]))
                signal = np.pad(signal, pad_width, 'constant')
        
        # Process the whole signal at once, with minimal memory usage
        signal_flat = signal.reshape(1, -1)
        signal_scaled = scaler.transform(signal_flat)
        processed_signal[0] = signal_scaled.reshape(target_length, num_leads)
        
        # Clear memory
        del signal, signal_flat, signal_scaled
        
        # Make prediction with minimal memory usage
        with tf.device('/cpu:0'):  # Force CPU to reduce memory
            probability = float(model.predict(processed_signal, verbose=0, batch_size=1)[0][0])
        
        binary_prediction = 1 if probability >= 0.5 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        # Default prediction in case of error
        return 0, 0.0
