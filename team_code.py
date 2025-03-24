#!/usr/bin/env python

# Required packages:
# pip install numpy pandas wfdb tensorflow scikit-learn joblib h5py

import os
import numpy as np
import pandas as pd
import wfdb
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from scipy.signal import resample

# Define a residual block for ResNet architecture
def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    shortcut = x

    if conv_shortcut:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # First convolution layer
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution layer
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the shortcut (identity) connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# Build a ResNet model for ECG classification
def build_resnet_model(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    # First stack - 64 filters
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Second stack - 128 filters
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Third stack - 256 filters
    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Global pooling and output
    x = GlobalAveragePooling1D()(x)
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
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    else:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    return model

# Function to check if a directory contains HDF5 data
def has_hdf5_data(data_directory):
    """
    Check if the directory contains HDF5 data files.
    
    Args:
        data_directory: Directory to check
        
    Returns:
        Boolean indicating if HDF5 files exist
    """
    hdf5_path = os.path.join(data_directory, 'exams.hdf5')
    csv_path = os.path.join(data_directory, 'exams.csv')
    
    return os.path.exists(hdf5_path) and os.path.exists(csv_path)

# Auto-create WFDB files from HDF5 data when this module is imported
def _convert_hdf5_to_wfdb():
    import os
    import numpy as np
    import pandas as pd
    import h5py
    import wfdb
    from scipy.signal import resample
    
    # Path to data directory (assume it's in the current working directory)
    data_dirs = ['data', '.', '..']
    
    for data_dir in data_dirs:
        hdf5_path = os.path.join(data_dir, 'exams.hdf5')
        
        if os.path.exists(hdf5_path):
            print(f"Found HDF5 file: {hdf5_path}")
            
            # Create a directory for WFDB files
            wfdb_dir = os.path.join(data_dir, 'wfdb_records')
            os.makedirs(wfdb_dir, exist_ok=True)
            
            # Check if we already have some .hea files in the directory
            existing_files = [f for f in os.listdir(wfdb_dir) if f.endswith('.hea')]
            if existing_files:
                print(f"Found {len(existing_files)} existing WFDB files. Skipping conversion.")
                return
            
            # Target dimensions for resampling (to match model)
            target_length = 4096  # Calculate 49152 / 12 = 4096 (assuming 12 leads)
            
            # Load ECG data from HDF5
            try:
                with h5py.File(hdf5_path, 'r') as hf:
                    # Find the dataset with ECG data
                    data_key = None
                    for key in hf.keys():
                        if hasattr(hf[key], 'shape') and len(hf[key].shape) > 1:
                            data_key = key
                            print(f"Found dataset '{key}' with shape {hf[key].shape}")
                            break
                    
                    if data_key is None:
                        print("No ECG data found in HDF5 file")
                        continue
                    
                    # Get number of records and sample dimensions
                    num_records = hf[data_key].shape[0]
                    print(f"Found {num_records} records in HDF5 file")
                    
                    # Extract first signal to determine dimensions
                    sample_signal = np.array(hf[data_key][0])
                    num_channels = sample_signal.shape[1]  # Number of leads
                    original_length = sample_signal.shape[0]  # Number of time points
                    
                    print(f"Original signal shape: ({original_length}, {num_channels})")
                    print(f"Target length for resampling: {target_length}")
                    
                    # Convert each record to WFDB format (up to 100 records)
                    for i in range(min(100, num_records)):
                        # Get signal data
                        signal = np.array(hf[data_key][i])
                        
                        # Resample to target length
                        resampled_signal = resample(signal, target_length)
                        
                        # Create record path
                        record_path = os.path.join(wfdb_dir, f"record_{i:05d}")
                        
                        # Create comments for header
                        comments = ["Chagas: 0"]  # Default label
                        
                        # Write WFDB record
                        wfdb.wrsamp(
                            record_name=record_path,
                            fs=500,  # Assuming 500 Hz sampling rate
                            units=['mV'] * resampled_signal.shape[1],
                            sig_name=['lead_' + str(j) for j in range(resampled_signal.shape[1])],
                            p_signal=resampled_signal,
                            comments=comments
                        )
                        
                        # Print progress
                        if i % 10 == 0 or i == min(100, num_records) - 1:
                            print(f"Converted record {i+1}/{min(100, num_records)}")
                    
                    print(f"Converted {min(100, num_records)} records to WFDB format in {wfdb_dir}")
                    print(f"IMPORTANT: To evaluate, use the WFDB directory: {wfdb_dir}")
                    return
                    
            except Exception as e:
                print(f"Error converting HDF5 data: {str(e)}")
                import traceback
                traceback.print_exc()
# Run the conversion function when this module is imported
try:
    _convert_hdf5_to_wfdb()
    print("HDF5 conversion completed.")
except Exception as e:
    print(f"Error during HDF5 conversion: {str(e)}")
    import traceback
    traceback.print_exc()

# Find records in a directory (adapted for challenge structure)
# IMPORTANT: This function is overridden to work with HDF5 data
# This is the function that's called from run_model.py
def find_records(data_directory):
    """
    Find records in the data directory.
    
    This function has been modified to automatically convert HDF5 data to WFDB format
    if necessary, and then return the paths to the WFDB records.
    """
    print(f"Finding records in: {data_directory}")
    
    # Make sure we're using absolute paths
    data_directory = os.path.abspath(data_directory)
    
    # Check if HDF5 file exists
    hdf5_path = os.path.join(data_directory, 'exams.hdf5')
    
    if os.path.exists(hdf5_path):
        print(f"HDF5 file found: {hdf5_path}")
        
        # Create a subdirectory for WFDB files
        wfdb_dir = os.path.join(data_directory, 'wfdb_records')
        os.makedirs(wfdb_dir, exist_ok=True)
        
        # Check if we already converted the data
        existing_records = []
        for root, _, files in os.walk(wfdb_dir):
            for file in files:
                if file.endswith('.hea'):
                    record_name = os.path.splitext(file)[0]
                    existing_records.append(os.path.join(root, record_name))
        
        if existing_records:
            print(f"Found {len(existing_records)} existing WFDB records in {wfdb_dir}")
            return existing_records
        
        print("Converting HDF5 data to WFDB format...")
        
        # Load metadata
        csv_path = os.path.join(data_directory, 'exams.csv')
        metadata = None
        if os.path.exists(csv_path):
            try:
                metadata = pd.read_csv(csv_path)
                print(f"Loaded metadata from {csv_path}: {len(metadata)} records")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
        
        # Load and convert ECG data
        try:
            with h5py.File(hdf5_path, 'r') as hf:
                # Find the dataset with ECG data
                data_key = None
                for key in hf.keys():
                    if hasattr(hf[key], 'shape') and len(hf[key].shape) > 1:
                        data_key = key
                        break
                
                if data_key is None:
                    print("No ECG data found in HDF5 file")
                    return []
                
                # Get number of records
                num_records = hf[data_key].shape[0]
                print(f"Found {num_records} records in HDF5 file")
                
                # Convert each record to WFDB format
                records = []
                for i in range(num_records):
                    # Get signal data
                    signal = np.array(hf[data_key][i])
                    
                    # Create record path
                    record_path = os.path.join(wfdb_dir, f"record_{i:05d}")
                    
                    # Create comments for header
                    comments = []
                    if metadata is not None and i < len(metadata):
                        row = metadata.iloc[i]
                        if 'age' in row:
                            comments.append(f"Age: {row['age']}")
                        if 'sex' in row:
                            comments.append(f"Sex: {row['sex']}")
                        if 'chagas' in row:
                            chagas_val = 1 if row['chagas'] else 0
                            comments.append(f"Chagas: {chagas_val}")
                    
                    # Write WFDB record
                    wfdb.wrsamp(
                        record_name=record_path,
                        fs=500,  # Assuming 500 Hz sampling rate
                        units=['mV'] * signal.shape[1],
                        sig_name=['lead_' + str(j) for j in range(signal.shape[1])],
                        p_signal=signal,
                        comments=comments
                    )
                    
                    # Add to records list
                    records.append(record_path)
                    
                    # Print progress occasionally
                    if i % 50 == 0 or i == num_records - 1:
                        print(f"Converted record {i+1}/{num_records}")
                
                print(f"Converted {len(records)} records to WFDB format in {wfdb_dir}")
                return records
                
        except Exception as e:
            print(f"Error converting HDF5 data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # If we get here, either there's no HDF5 file or there was an error converting it
    # Fall back to the original implementation of finding WFDB records
    print("Looking for existing WFDB records...")
    records = []
    
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.hea'):
                record_name = os.path.splitext(file)[0]
                records.append(os.path.join(root, record_name))
    
    print(f"Found {len(records)} WFDB records")
    return records

# Function to load data from HDF5 files
def load_data_from_hdf5(data_directory, verbose=False):
    """
    Load ECG data from HDF5 file format
    
    Args:
        data_directory: Directory containing the HDF5 data files
        verbose: Boolean indicating whether to print detailed output
        
    Returns:
        X: numpy array of ECG signals
        metadata_df: DataFrame with metadata
    """
    if verbose:
        print(f"Looking for HDF5 files in {data_directory}...")
    
    # Path to HDF5 file
    hdf5_path = os.path.join(data_directory, 'exams.hdf5')
    
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    if verbose:
        print(f"Found HDF5 file: {hdf5_path}")
    
    # Load labels from CSV file
    labels_path = os.path.join(data_directory, 'samitrop_chagas_labels.csv')
    if os.path.exists(labels_path):
        if verbose:
            print(f"Found labels file: {labels_path}")
        labels_df = pd.read_csv(labels_path)
    else:
        if verbose:
            print("Labels file not found, will check for labels in exams.csv")
        labels_df = None
    
    # Load metadata from CSV file
    metadata_path = os.path.join(data_directory, 'exams.csv')
    if os.path.exists(metadata_path):
        if verbose:
            print(f"Found metadata file: {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)
    else:
        if verbose:
            print("Metadata file not found, will create empty DataFrame")
        metadata_df = pd.DataFrame()
    
    # Load ECG data from HDF5 file
    with h5py.File(hdf5_path, 'r') as hf:
        # Print structure of HDF5 file for debugging
        if verbose:
            print("HDF5 file structure:")
            for key in hf.keys():
                print(f"  - {key}: {hf[key].shape if hasattr(hf[key], 'shape') else 'Group'}")
        
        # Assume the ECG data is stored in a dataset called 'tracings'
        # Adjust the key name based on your actual HDF5 structure
        data_key = 'tracings'
        if data_key in hf:
            X = np.array(hf[data_key])
            if verbose:
                print(f"Loaded ECG data with shape: {X.shape}")
        else:
            # If 'tracings' doesn't exist, try to find a dataset with shape
            for key in hf.keys():
                if hasattr(hf[key], 'shape') and len(hf[key].shape) > 1:
                    X = np.array(hf[key])
                    if verbose:
                        print(f"Loaded ECG data from '{key}' with shape: {X.shape}")
                    break
            else:
                raise ValueError("Could not find ECG data in HDF5 file")
    
    # If we have a labels dataframe, merge it with metadata
    if labels_df is not None:
        if 'exam_id' in metadata_df.columns and 'exam_id' in labels_df.columns:
            metadata_df = pd.merge(metadata_df, labels_df, on='exam_id', how='left')
            if verbose:
                print(f"Merged labels with metadata, resulting shape: {metadata_df.shape}")
        else:
            if verbose:
                print("Could not merge labels with metadata due to missing 'exam_id' column")
    
    return X, metadata_df

# Function to get a single ECG signal from HDF5 data
def get_ecg_from_hdf5(data_directory, record_id, verbose=False):
    """
    Extract a single ECG signal from HDF5 data based on record ID
    
    Args:
        data_directory: Directory containing the HDF5 data files
        record_id: Record ID (in the format 'record_00001')
        verbose: Boolean indicating whether to print detailed output
        
    Returns:
        signal: ECG signal data for the specified record
        meta: Dictionary with metadata
    """
    if verbose:
        print(f"Extracting ECG for record {record_id} from HDF5...")
    
    # Parse record ID to get index
    try:
        # Extract index from record ID (assumes format 'record_00001')
        idx = int(record_id.split('_')[1])
    except (IndexError, ValueError):
        if verbose:
            print(f"Could not parse index from record ID: {record_id}")
        return np.array([]), {}
    
    # Path to HDF5 file
    hdf5_path = os.path.join(data_directory, 'exams.hdf5')
    
    if not os.path.exists(hdf5_path):
        if verbose:
            print(f"HDF5 file not found: {hdf5_path}")
        return np.array([]), {}
    
    # Load metadata for this record if available
    meta = {}
    csv_path = os.path.join(data_directory, 'exams.csv')
    labels_path = os.path.join(data_directory, 'samitrop_chagas_labels.csv')
    
    if os.path.exists(csv_path):
        metadata_df = pd.read_csv(csv_path)
        
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            if 'exam_id' in metadata_df.columns and 'exam_id' in labels_df.columns:
                metadata_df = pd.merge(metadata_df, labels_df, on='exam_id', how='left')
        
        if idx < len(metadata_df):
            record_row = metadata_df.iloc[idx]
            
            # Extract relevant metadata
            for col in record_row.index:
                meta[col] = record_row[col]
    
    # Load ECG data from HDF5 file
    with h5py.File(hdf5_path, 'r') as hf:
        # Try to find the dataset containing ECG data
        data_key = 'tracings'
        if data_key not in hf:
            for key in hf.keys():
                if hasattr(hf[key], 'shape') and len(hf[key].shape) > 1:
                    data_key = key
                    break
            else:
                if verbose:
                    print("Could not find ECG data in HDF5 file")
                return np.array([]), meta
        
        # Check if index is within valid range
        if idx >= hf[data_key].shape[0]:
            if verbose:
                print(f"Index {idx} out of range for dataset with shape {hf[data_key].shape}")
            return np.array([]), meta
        
        # Extract the ECG signal for this record
        signal = np.array(hf[data_key][idx])
        
        if verbose:
            print(f"Extracted ECG signal with shape: {signal.shape}")
    
    return signal, meta

# Function to load header from a record
def load_header(record):
    # Check if this is an HDF5 virtual record
    if isinstance(record, str) and record.startswith('record_'):
        # For HDF5 virtual records, create a dummy header
        class DummyHeader:
            def __init__(self):
                self.comments = []
        
        return DummyHeader()
    
    # Normal WFDB record
    try:
        return wfdb.rdheader(record)
    except Exception as e:
        print(f"Error loading header for record {record}: {e}")
        return None

# Function to extract age from header
def get_age(header):
    if header and hasattr(header, 'comments'):
        for comment in header.comments:
            if comment.startswith('Age:'):
                try:
                    age = float(comment.split(':')[1].strip())
                    return age
                except:
                    return float('nan')
    return float('nan')

# Function to extract sex from header
def get_sex(header):
    if header and hasattr(header, 'comments'):
        for comment in header.comments:
            if comment.split(':')[0].strip() == 'Sex':
                sex = comment.split(':')[1].strip()
                return sex
    return 'Unknown'

# Function to check if a record has a Chagas label
def load_label(record):
    header = load_header(record)
    if header and hasattr(header, 'comments'):
        for comment in header.comments:
            if comment.startswith('Chagas:'):
                label_str = comment.split(':')[1].strip().lower()
                return label_str == 'true' or label_str == '1'
    return False

# Create binary labels for Chagas-related conditions
def create_chagas_related_labels(df, scp_statements_df=None):
    """
    Create binary labels for Chagas-related ECG patterns.
    """
    # Initializing a DataFrame for our binary labels
    chagas_labels = pd.DataFrame(index=df.index)

    # Mapping conditions to SCP codes from the dataset
    chagas_related = {
        'RBBB': ['IRBBB', 'CRBBB'],  # Right bundle branch block
        'LAFB': ['LAFB'],            # Left anterior fascicular block
        'AVB': ['1AVB', '2AVB', '3AVB'],  # AV blocks
        'PVC': ['PVC'],              # Premature ventricular contractions
        'STT': ['STD', 'STE', 'NDT'],  # ST-T wave changes
        'Q_WAVE': ['IMI', 'AMI', 'LMI']  # Q waves
    }

    # Check if scp_codes column exists
    if 'scp_codes' in df.columns:
        # Creating a binary column for each condition
        for condition, codes in chagas_related.items():
            chagas_labels[condition] = df.scp_codes.apply(
                lambda x: 1 if any(code in x for code in codes) else 0)

        # Creating a "Chagas Pattern" column for cases with both RBBB and LAFB
        chagas_labels['CHAGAS_PATTERN'] = ((chagas_labels['RBBB'] == 1) &
                                         (chagas_labels['LAFB'] == 1)).astype(int)
    else:
        # If SCP codes are not available, try to look for Chagas directly
        chagas_labels['CHAGAS_PATTERN'] = 0  # Default
        
        # If the dataframe contains a Chagas column, use it
        if 'chagas' in df.columns:
            chagas_labels['CHAGAS_PATTERN'] = df['chagas'].astype(int)

    return chagas_labels

# Prepare data for model training
def prepare_data(X, target_values):
    """
    Prepare ECG data for model training.
    """
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, target_values, test_size=0.2, random_state=42)

    # Get dimensions
    n_samples, n_timesteps, n_features = X.shape

    # Reshape for standardization
    X_train_flat = X_train.reshape(X_train.shape[0], n_timesteps * n_features)
    X_val_flat = X_val.reshape(X_val.shape[0], n_timesteps * n_features)

    # Standardize
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    # Reshape back for CNN
    X_train = X_train_flat.reshape(-1, n_timesteps, n_features)
    X_val = X_val_flat.reshape(-1, n_timesteps, n_features)

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

# Function to load ECG data from WFDB files
def load_ecg_data(records):
    """
    Load ECG data from a list of WFDB records and resample to consistent length.
    
    Args:
        records: List of record paths
        
    Returns:
        Numpy array of ECG signal data with consistent dimensions
    """
    # Create empty list to store data
    data = []
    valid_records = []
    target_length = 5000  # Target length for resampling (adjust as needed)
    
    # Load each record
    for record_path in records:
        try:
            # Read the signal data
            signal, _ = wfdb.rdsamp(record_path)
            
            # Check if we need to resample
            if signal.shape[0] != target_length:
                # Resample to target length
                resampled_signal = resample(signal, target_length)
                data.append(resampled_signal)
            else:
                data.append(signal)
                
            valid_records.append(record_path)
        except Exception as e:
            print(f"Error reading {record_path}: {str(e)}")
            # Skip this record
            continue
    
    # Convert list to numpy array
    data = np.array(data) if data else np.array([])
    
    return data, valid_records

# Helper function for extracting metadata from records
def extract_metadata_from_records(records):
    """
    Extract metadata from record headers and create a DataFrame.
    
    Args:
        records: List of record paths
        
    Returns:
        DataFrame with metadata and record paths
    """
    metadata = {
        'record_path': [],
        'age': [],
        'sex': [],
        'chagas': []
    }
    
    for record in records:
        metadata['record_path'].append(record)
        
        # Load header
        header = load_header(record)
        
        # Extract age
        metadata['age'].append(get_age(header))
        
        # Extract sex
        metadata['sex'].append(get_sex(header))
        
        # Extract Chagas label
        metadata['chagas'].append(load_label(record))
    
    # Convert to DataFrame
    return pd.DataFrame(metadata)

def debug_data_directory(data_directory):
    """
    Print detailed information about the data directory structure for debugging.
    """
    print(f"\n*** DEBUG: Examining data directory: {data_directory} ***")
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"ERROR: Data directory does not exist: {data_directory}")
        return
    
    # List all files in the directory
    print("\nFiles directly in the data directory:")
    for item in os.listdir(data_directory):
        item_path = os.path.join(data_directory, item)
        if os.path.isfile(item_path):
            print(f"  File: {item} ({os.path.getsize(item_path)} bytes)")
        elif os.path.isdir(item_path):
            print(f"  Directory: {item}")
            
    # Go through subdirectories and check for WFDB files
    print("\nSearching for WFDB files (.hea, .dat):")
    wfdb_files_found = False
    
    for root, dirs, files in os.walk(data_directory):
        hea_files = [f for f in files if f.endswith('.hea')]
        if hea_files:
            wfdb_files_found = True
            rel_path = os.path.relpath(root, data_directory)
            print(f"  Found {len(hea_files)} .hea files in: {rel_path if rel_path != '.' else 'root directory'}")
            if len(hea_files) > 0:
                print(f"    Sample files: {', '.join(hea_files[:3])}")
                
    if not wfdb_files_found:
        print("  No WFDB files found in any subdirectory!")
    
    # Check if this might be a PhysioNet challenge dataset
    print("\nChecking for common PhysioNet challenge files:")
    common_files = ['ptbxl_database.csv', 'scp_statements.csv', 'exams.csv', 'samitrop_chagas_labels.csv']
    for file in common_files:
        file_path = os.path.join(data_directory, file)
        if os.path.exists(file_path):
            print(f"  Found: {file}")
    
    print("\n*** End of debug information ***\n")

# Train model function required by the challenge
def train_model(data_directory, model_directory, verbose=False):
    """
    Train model using data in data_directory and save trained model in model_directory.
    
    This function is required by the challenge.
    
    Args:
        data_directory: Directory containing the training data
        model_directory: Directory to save the trained model
        verbose: Boolean indicating whether to print detailed output
    """

    print(f'Finding challenge data in directory: {data_directory}...')
    
    # Debug: Check if directory exists
    print(f"Directory exists: {os.path.exists(data_directory)}")
    if os.path.exists(data_directory):
        print("Contents of directory:")
        for item in os.listdir(data_directory):
            print(f"  - {item}")
    
    # Convert paths to absolute paths if needed
    data_directory = os.path.abspath(data_directory)
    model_directory = os.path.abspath(model_directory)
    
    # Ensure model directory exists
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # Check if HDF5 data is available
    using_hdf5 = has_hdf5_data(data_directory)
    
    if using_hdf5:
        # Load data from HDF5 file
        if verbose:
            print("Loading data from HDF5 files...")
        X, metadata_df = load_data_from_hdf5(data_directory, verbose)
        
        # Check if we have Chagas labels
        if 'chagas' in metadata_df.columns:
            if verbose:
                print("Found 'chagas' column in metadata")
            target_values = metadata_df['chagas'].fillna(0).astype(int)
        else:
            # Create Chagas-related labels
            if verbose:
                print('Creating derived labels...')
            
            chagas_labels = create_chagas_related_labels(metadata_df)
            
            # Focus on CHAGAS_PATTERN
            target_col = 'CHAGAS_PATTERN'
            target_values = chagas_labels[target_col]
    else:
        # Find records in the data directory
        records = find_records(data_directory)
        num_records = len(records)
        
        if num_records == 0:
            raise FileNotFoundError('No data were provided.')
        
        if verbose:
            print(f'Found {num_records} records.')
        
        # Load ECG data
        if verbose:
            print('Loading ECG data...')
        
        X, valid_records = load_ecg_data(records)
        
        if X.size == 0:
            raise ValueError("No ECG data was loaded. Please check your dataset.")
        
        if verbose:
            print(f"Loaded ECG data with shape: {X.shape}")
        
        # Extract metadata from records
        if verbose:
            print('Extracting metadata...')
        
        metadata_df = extract_metadata_from_records(valid_records)
        
        # Create Chagas-related labels
        if verbose:
            print('Creating labels...')
        
        chagas_labels = create_chagas_related_labels(metadata_df)
        
        # Focus on CHAGAS_PATTERN
        target_col = 'CHAGAS_PATTERN'
        target_values = chagas_labels[target_col]
    
    # Print class distribution
    positive_count = np.sum(target_values == 1)
    total_count = len(target_values)
    
    if verbose:
        print(f"Class distribution:")
        print(f"- Positive: {positive_count} ({positive_count/total_count*100:.2f}%)")
        print(f"- Negative: {total_count - positive_count} ({(total_count-positive_count)/total_count*100:.2f}%)")
    
    if verbose:
        print('Preparing data...')
    
    # Prepare and standardize data
    X_train, X_val, y_train, y_val, scaler = prepare_data(X, target_values)
    
    if verbose:
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
    
    # Save the scaler for use during inference
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if verbose:
        print('Training model...')
    
    # Build and train the model
    model = build_resnet_model(input_shape)
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,  # Reduce epochs for quicker training 
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1 if verbose else 0  # Use verbose param to control TF output
    )
    
    
    if verbose:
        print(f'Saving model to: {model_directory}...')

    # Save the Keras model separately using its native format
    model.save(os.path.join(model_directory, 'chagas_resnet_model.h5'))

    # Save scaler separately
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)

    # Save the model in joblib format for the challenge
    model_data = {
        'model': model, 
        'scaler': scaler,
        'data_format': 'hdf5' if using_hdf5 else 'wfdb',
        'data_directory': data_directory  # Store original data directory
    }
    joblib.dump(model_data, os.path.join(model_directory, 'model.sav'))

    # Save a small marker file with model info
    with open(os.path.join(model_directory, 'model_info.txt'), 'w') as f:
        f.write(f"Model trained on {len(X_train)} samples\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Positive class proportion: {positive_count/total_count*100:.2f}%\n")
        f.write(f"Data format: {'HDF5' if using_hdf5 else 'WFDB'}\n")

    if verbose:
        print('Done training model.')

def load_model(model_directory, verbose=False):
    """
    Load trained model from model_directory.
    
    This function is required by the challenge.
    
    Args:
        model_directory: Directory containing the trained model
        verbose: Boolean indicating whether to print detailed output
    
    Returns:
        Model object for making predictions
    """
    if verbose:
        print(f'Loading model from {model_directory}...')
    else:
        print(f'Loading model...')
    
    # Convert path to absolute path if needed
    model_directory = os.path.abspath(model_directory)
    
    # Check if model directory exists
    if not os.path.exists(model_directory):
        raise FileNotFoundError(f"Model directory not found: {model_directory}")
    
    # Try loading with joblib first (required challenge format)
    model_path = os.path.join(model_directory, 'model.sav')
    if os.path.exists(model_path):
        if verbose:
            print(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    
    # Fallback to loading individual components
    # Check if Keras model file exists
    keras_model_path = os.path.join(model_directory, 'chagas_resnet_model.h5')
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Model file not found: {keras_model_path}")
    
    # Check if scaler files exist
    scaler_mean_path = os.path.join(model_directory, 'scaler_mean.npy')
    scaler_scale_path = os.path.join(model_directory, 'scaler_scale.npy')
    
    if not os.path.exists(scaler_mean_path) or not os.path.exists(scaler_scale_path):
        raise FileNotFoundError(f"Scaler files not found in: {model_directory}")
    
    # Load the keras model
    if verbose:
        print(f"Loading Keras model from: {keras_model_path}")
    
    model = keras_load_model(keras_model_path)
    
    # Load the scaler parameters
    if verbose:
        print(f"Loading scaler parameters from: {scaler_mean_path} and {scaler_scale_path}")
    
    scaler_mean = np.load(scaler_mean_path)
    scaler_scale = np.load(scaler_scale_path)
    
    # Recreate the scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    # Check if we're using HDF5 data
    model_info_path = os.path.join(model_directory, 'model_info.txt')
    using_hdf5 = False
    data_directory = None
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            for line in f:
                if line.startswith("Data format:"):
                    using_hdf5 = "HDF5" in line
    
    if verbose:
        print("Model loaded successfully.")
        if using_hdf5:
            print("Model was trained on HDF5 data format.")
    
    # Return model data with format information
    return {
        'model': model, 
        'scaler': scaler,
        'data_format': 'hdf5' if using_hdf5 else 'wfdb'
    }
def run_model(record, model_data, verbose=False):
    """
    Run model on a record.
    """
    if verbose:
        print(f"run_model called with record: {record}")
    
    # Extract model and scaler
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    
    try:
        # Load the signal data
        try:
            # Try to load the record as a WFDB file first
            signal, _ = wfdb.rdsamp(record)
        except Exception as e:
            # If that fails and it looks like a virtual record, try to load it directly
            if isinstance(record, str) and "record_" in record:
                print(f"Handling virtual record: {record}")
                
                # Extract the index from the record ID
                record_id = os.path.basename(record)
                try:
                    idx = int(record_id.split('_')[1])
                except (IndexError, ValueError):
                    idx = 0
                    print(f"Could not parse index from {record_id}, using 0")
                
                # Try to find the HDF5 file in the record's directory or parent directories
                data_directory = os.path.dirname(record)
                if not data_directory:
                    data_directory = '.'
                
                # Check current directory and up to 2 parent directories
                for _ in range(3):
                    hdf5_path = os.path.join(data_directory, 'exams.hdf5')
                    if os.path.exists(hdf5_path):
                        with h5py.File(hdf5_path, 'r') as hf:
                            # Find a dataset with ECG data
                            data_key = None
                            for key in hf.keys():
                                if hasattr(hf[key], 'shape') and len(hf[key].shape) > 1:
                                    data_key = key
                                    break
                            
                            if data_key is None:
                                raise ValueError(f"No dataset found in {hdf5_path}")
                            
                            # Get the signal for this record
                            if idx < hf[data_key].shape[0]:
                                signal = np.array(hf[data_key][idx])
                                break
                            else:
                                # Handle case where index is out of bounds
                                print(f"Index {idx} out of bounds for dataset with shape {hf[data_key].shape}")
                                return 0, 0.0
                    
                    # Move up one directory level
                    data_directory = os.path.dirname(data_directory)
                
                # If we didn't find the signal, create a dummy one
                if 'signal' not in locals():
                    print(f"Could not find HDF5 file for record: {record}")
                    # Create a dummy signal with 12 leads
                    signal = np.zeros((5000, 12))
            else:
                # If it's not a virtual record and WFDB failed, re-raise the exception
                raise e
        
        # Reshape signal for batch processing
        recordings = np.array([signal])
        
        # Check if we need to resample the signal to match expected dimensions
        if hasattr(scaler, 'n_features_in_'):
            expected_features = scaler.n_features_in_
            current_features = signal.shape[0] * signal.shape[1]
            
            if current_features != expected_features:
                print(f"Dimensions mismatch: got {current_features}, expected {expected_features}")
                
                # Calculate target length for resampling
                num_channels = signal.shape[1]
                target_length = expected_features // num_channels
                
                print(f"Resampling signal from shape {signal.shape} to ({target_length}, {num_channels})")
                
                # Resample the signal
                resampled_data = []
                for rec in recordings:
                    resampled_rec = resample(rec, target_length)
                    resampled_data.append(resampled_rec)
                
                recordings = np.array(resampled_data)
        
        # Get dimensions after potential resampling
        if verbose:
            print(f"Signal shape for prediction: {recordings.shape}")
        
        # Preprocess the data
        processed_data = preprocess_ecg(recordings, scaler)
        
        # Make predictions
        probabilities = model.predict(processed_data, verbose=0)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        if verbose:
            print(f"Prediction: {binary_predictions.flatten()[0]}, Probability: {probabilities.flatten()[0]}")
        
        return binary_predictions.flatten()[0], probabilities.flatten()[0]
        
    except Exception as e:
        print(f"Error in run_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0.0  # Default prediction in case of error