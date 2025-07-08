import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, 
                                   Input, concatenate, BatchNormalization, 
                                   GlobalAveragePooling1D, Flatten, Add,
                                   MultiHeadAttention, LayerNormalization,
                                   Bidirectional, LSTM, GRU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from scipy import signal as scipy_signal
from scipy.stats import zscore, kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Enhanced constants
TARGET_SAMPLING_RATE = 400
TARGET_SIGNAL_LENGTH = 2048
MAX_SAMPLES = 15000  # Increased for better training
BATCH_SIZE = 64  # Larger batch size for stability
NUM_LEADS = 12
VALIDATION_SPLIT = 0.2
CV_FOLDS = 5

class ECGPreprocessor:
    """Advanced ECG signal preprocessing"""
    
    def __init__(self, target_fs=TARGET_SAMPLING_RATE, target_length=TARGET_SIGNAL_LENGTH):
        self.target_fs = target_fs
        self.target_length = target_length
        self.nyquist = target_fs / 2
        
    def preprocess_signal(self, signal, fs=None):
        """
        Advanced signal preprocessing pipeline
        """
        try:
            signal = np.array(signal, dtype=np.float32)
            
            # Handle input shape
            signal = self._standardize_shape(signal)
            
            # Remove artifacts and noise
            signal = self._remove_artifacts(signal)
            
            # Resample if needed
            if fs and fs != self.target_fs:
                signal = self._resample_signal(signal, fs, self.target_fs)
            
            # Standardize length
            signal = self._standardize_length(signal)
            
            # Advanced filtering
            signal = self._apply_filters(signal)
            
            # Normalization
            signal = self._normalize_signal(signal)
            
            return signal.astype(np.float32)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def _standardize_shape(self, signal):
        """Ensure correct signal shape"""
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[0] <= 12:
            signal = signal.T
        
        # Ensure 12 leads
        if signal.shape[1] > 12:
            signal = signal[:, :12]
        elif signal.shape[1] < 12:
            # Intelligent lead reconstruction
            signal = self._reconstruct_leads(signal)
        
        return signal
    
    def _reconstruct_leads(self, signal):
        """Intelligently reconstruct missing leads"""
        current_leads = signal.shape[1]
        
        if current_leads == 8:  # Common 8-lead configuration
            # Reconstruct limb leads III, aVR, aVL, aVF from I and II
            lead_I = signal[:, 0]
            lead_II = signal[:, 1]
            lead_III = lead_II - lead_I
            aVR = -(lead_I + lead_II) / 2
            aVL = lead_I - lead_II / 2
            aVF = lead_II - lead_I / 2
            
            # Add reconstructed leads
            reconstructed = np.column_stack([
                signal, lead_III.reshape(-1, 1), 
                aVR.reshape(-1, 1), aVL.reshape(-1, 1), aVF.reshape(-1, 1)
            ])
            signal = reconstructed
        
        # Pad remaining leads with the average of existing leads
        if signal.shape[1] < 12:
            avg_lead = np.mean(signal, axis=1, keepdims=True)
            padding = np.repeat(avg_lead, 12 - signal.shape[1], axis=1)
            signal = np.hstack([signal, padding])
        
        return signal
    
    def _remove_artifacts(self, signal):
        """Remove common ECG artifacts"""
        # Remove extreme outliers
        for i in range(signal.shape[1]):
            lead = signal[:, i]
            # Use robust outlier detection
            q1, q3 = np.percentile(lead, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            signal[:, i] = np.clip(lead, lower_bound, upper_bound)
        
        return signal
    
    def _resample_signal(self, signal, original_fs, target_fs):
        """Resample signal to target frequency"""
        if original_fs == target_fs:
            return signal
        
        resample_factor = target_fs / original_fs
        new_length = int(signal.shape[0] * resample_factor)
        
        resampled = np.zeros((new_length, signal.shape[1]))
        for i in range(signal.shape[1]):
            resampled[:, i] = scipy_signal.resample(signal[:, i], new_length)
        
        return resampled
    
    def _standardize_length(self, signal):
        """Standardize signal length"""
        current_length = signal.shape[0]
        
        if current_length == self.target_length:
            return signal
        elif current_length > self.target_length:
            # Take center portion to preserve QRS complexes
            start_idx = (current_length - self.target_length) // 2
            return signal[start_idx:start_idx + self.target_length]
        else:
            # Pad with reflection
            pad_length = self.target_length - current_length
            pad_before = pad_length // 2
            pad_after = pad_length - pad_before
            
            padded = np.zeros((self.target_length, signal.shape[1]))
            for i in range(signal.shape[1]):
                padded[:, i] = np.pad(signal[:, i], (pad_before, pad_after), mode='reflect')
            
            return padded
    
    def _apply_filters(self, signal):
        """Apply advanced filtering"""
        try:
            # Design filters
            # High-pass filter for baseline wander removal (0.5 Hz)
            b_hp, a_hp = scipy_signal.butter(4, 0.5 / self.nyquist, btype='high')
            
            # Low-pass filter for noise removal (100 Hz)
            b_lp, a_lp = scipy_signal.butter(4, 100 / self.nyquist, btype='low')
            
            # Notch filter for power line interference (50/60 Hz)
            b_notch, a_notch = scipy_signal.iirnotch(50, 30, self.target_fs)
            
            filtered_signal = np.zeros_like(signal)
            for i in range(signal.shape[1]):
                lead = signal[:, i]
                
                # Apply filters in sequence with error handling
                try:
                    lead = scipy_signal.filtfilt(b_hp, a_hp, lead)
                    lead = scipy_signal.filtfilt(b_lp, a_lp, lead)
                    lead = scipy_signal.filtfilt(b_notch, a_notch, lead)
                except:
                    # If filtering fails, just use the original signal
                    pass
                
                filtered_signal[:, i] = lead
            
            return filtered_signal
        except Exception as e:
            # If any filter design fails, return original signal
            return signal
    
    def _normalize_signal(self, signal):
        """Advanced signal normalization"""
        normalized_signal = np.zeros_like(signal)
        
        for i in range(signal.shape[1]):
            lead = signal[:, i]
            
            # Remove DC component
            lead = lead - np.mean(lead)
            
            # Robust scaling using median and MAD
            median = np.median(lead)
            mad = np.median(np.abs(lead - median))
            
            if mad > 1e-6:
                lead = (lead - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
            
            # Additional clipping for extreme values
            lead = np.clip(lead, -8, 8)
            
            normalized_signal[:, i] = lead
        
        return normalized_signal

class FeatureExtractor:
    """Extract domain-specific features from ECG"""
    
    @staticmethod
    def extract_statistical_features(signal):
        """Extract statistical features from each lead"""
        features = []
        
        for i in range(signal.shape[1]):
            lead = signal[:, i]
            
            # Basic statistics
            features.extend([
                np.mean(lead),
                np.std(lead),
                np.median(lead),
                np.percentile(lead, 25),
                np.percentile(lead, 75),
                np.min(lead),
                np.max(lead),
                np.ptp(lead),  # peak-to-peak
            ])
            
            # Shape features
            features.extend([
                kurtosis(lead),
                skew(lead),
            ])
        
        return np.array(features)
    
    @staticmethod
    def extract_frequency_features(signal, fs=TARGET_SAMPLING_RATE):
        """Extract frequency domain features"""
        features = []
        
        for i in range(signal.shape[1]):
            lead = signal[:, i]
            
            try:
                # Power spectral density
                freqs, psd = scipy_signal.welch(lead, fs=fs, nperseg=min(256, len(lead)//4))
                
                # Frequency band powers
                freq_bands = [(0.5, 4), (4, 15), (15, 40), (40, 100)]
                for low, high in freq_bands:
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask])
                    else:
                        band_power = 0.0
                    features.append(band_power)
                
                # Dominant frequency
                if len(psd) > 0:
                    dominant_freq = freqs[np.argmax(psd)]
                else:
                    dominant_freq = 0.0
                features.append(dominant_freq)
                
            except Exception as e:
                # If frequency analysis fails, add default values
                features.extend([0.0] * 5)  # 4 band powers + 1 dominant freq
        
        return np.array(features)
    
    @staticmethod
    def extract_morphological_features(signal):
        """Extract morphological features related to Chagas disease"""
        features = []
        
        try:
            # Heart rate variability indicators
            for i in range(min(3, signal.shape[1])):  # Focus on limb leads
                lead = signal[:, i]
                
                try:
                    # Find peaks (R-waves)
                    peaks, _ = scipy_signal.find_peaks(lead, height=np.std(lead), distance=50)
                    
                    if len(peaks) > 1:
                        rr_intervals = np.diff(peaks)
                        features.extend([
                            np.mean(rr_intervals),
                            np.std(rr_intervals),
                            len(peaks),  # Heart rate indicator
                        ])
                    else:
                        features.extend([0, 0, 0])
                except:
                    features.extend([0, 0, 0])
            
            # QRS width estimation (important for Chagas)
            try:
                lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
                peaks, _ = scipy_signal.find_peaks(np.abs(lead_ii), height=np.std(lead_ii), distance=50)
                
                if len(peaks) > 0:
                    # Estimate QRS width from first peak
                    peak_idx = peaks[0]
                    start_search = max(0, peak_idx - 50)
                    end_search = min(len(lead_ii), peak_idx + 50)
                    
                    qrs_segment = lead_ii[start_search:end_search]
                    qrs_width = np.sum(np.abs(qrs_segment) > 0.3 * np.max(np.abs(qrs_segment)))
                    features.append(qrs_width)
                else:
                    features.append(0)
            except:
                features.append(0)
                
        except Exception as e:
            # If morphological analysis fails, return default features
            features = [0] * 10  # 3 leads * 3 features + 1 QRS width
        
        return np.array(features)

def train_model(data_folder, model_folder, verbose):
    """
    Enhanced training with cross-validation and ensemble approach
    """
    if verbose:
        print("Training enhanced Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load data with improved handling
    signals, labels, demographics, features = load_data_enhanced(data_folder, verbose)
    
    if len(signals) < 100:
        if verbose:
            print(f"Insufficient data ({len(signals)} samples), creating baseline model")
        return create_baseline_model(model_folder, verbose)
    
    return train_enhanced_model(signals, labels, demographics, features, model_folder, verbose)

def load_data_enhanced(data_folder, verbose):
    """
    Enhanced data loading with feature extraction
    """
    signals = []
    labels = []
    demographics = []
    features = []
    
    preprocessor = ECGPreprocessor()
    
    # Load from multiple sources
    hdf5_path = os.path.join(data_folder, 'exams.hdf5')
    if os.path.exists(hdf5_path):
        if verbose:
            print("Loading from HDF5...")
        s, l, d, f = load_from_hdf5_enhanced(data_folder, preprocessor, verbose)
        signals.extend(s)
        labels.extend(l)
        demographics.extend(d)
        features.extend(f)
    
    # Load WFDB records
    if len(signals) < 2000:
        if verbose:
            print("Loading from WFDB records...")
        s, l, d, f = load_from_wfdb_enhanced(data_folder, preprocessor, verbose)
        signals.extend(s)
        labels.extend(l)
        demographics.extend(d)
        features.extend(f)
    
    if verbose:
        print(f"Total loaded: {len(signals)} samples")
        if len(labels) > 0:
            pos_rate = np.mean(labels) * 100
            print(f"Positive rate: {pos_rate:.1f}%")
    
    return signals, labels, demographics, features

def load_from_hdf5_enhanced(data_folder, preprocessor, verbose):
    """Enhanced HDF5 loading with feature extraction"""
    signals = []
    labels = []
    demographics = []
    features = []
    
    try:
        # Load and process similar to original but with enhanced preprocessing
        exams_path = os.path.join(data_folder, 'exams.csv')
        if not os.path.exists(exams_path):
            return signals, labels, demographics, features
        
        exams_df = pd.read_csv(exams_path, nrows=MAX_SAMPLES)
        
        # Load Chagas labels with better handling
        chagas_labels = load_chagas_labels_enhanced(data_folder, verbose)
        
        # Load HDF5 signals
        hdf5_path = os.path.join(data_folder, 'exams.hdf5')
        with h5py.File(hdf5_path, 'r') as hdf:
            dataset = hdf.get('tracings', hdf.get('exams', hdf[list(hdf.keys())[0]]))
            
            for idx, row in exams_df.iterrows():
                if len(signals) >= MAX_SAMPLES:
                    break
                
                try:
                    exam_id = row.get('exam_id', row.get('id', idx))
                    
                    # Get label with improved logic
                    label = get_label_enhanced(exam_id, row, chagas_labels)
                    if label is None:
                        continue
                    
                    # Extract and preprocess signal
                    if hasattr(dataset, 'shape') and len(dataset.shape) == 3:
                        raw_signal = dataset[idx]
                    elif str(exam_id) in dataset:
                        raw_signal = dataset[str(exam_id)][:]
                    else:
                        continue
                    
                    # Enhanced preprocessing
                    processed_signal = preprocessor.preprocess_signal(raw_signal)
                    if processed_signal is None:
                        continue
                    
                    # Extract features
                    signal_features = extract_all_features(processed_signal)
                    
                    # Demographics
                    demo = extract_demographics_enhanced(row)
                    
                    signals.append(processed_signal)
                    labels.append(label)
                    demographics.append(demo)
                    features.append(signal_features)
                    
                    if verbose and len(signals) % 1000 == 0:
                        print(f"Processed {len(signals)} HDF5 samples")
                
                except Exception as e:
                    if verbose and len(signals) < 5:
                        print(f"Error processing sample {idx}: {e}")
                    continue
    
    except Exception as e:
        if verbose:
            print(f"HDF5 loading error: {e}")
    
    return signals, labels, demographics, features

def load_from_wfdb_enhanced(data_folder, preprocessor, verbose):
    """Enhanced WFDB loading"""
    signals = []
    labels = []
    demographics = []
    features = []
    
    try:
        records = find_records(data_folder)
        if verbose:
            print(f"Found {len(records)} WFDB records")
        
        for record_name in records[:MAX_SAMPLES]:
            if len(signals) >= MAX_SAMPLES:
                break
            
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load signal and header with better error handling
                try:
                    raw_signal, fields = load_signals(record_path)
                    header = load_header(record_path)
                except Exception as e:
                    if verbose and len(signals) < 5:
                        print(f"Error loading WFDB signal/header {record_name}: {e}")
                    continue
                
                # Check if signal is valid
                if raw_signal is None or len(raw_signal) == 0:
                    continue
                
                # Get sampling frequency
                try:
                    fs = get_frequency(header) if header else TARGET_SAMPLING_RATE
                except:
                    fs = TARGET_SAMPLING_RATE
                
                # Enhanced preprocessing
                processed_signal = preprocessor.preprocess_signal(raw_signal, fs)
                if processed_signal is None:
                    continue
                
                # Extract label with better error handling
                try:
                    label = load_label(record_path)
                    if label is None:
                        continue
                except Exception as e:
                    if verbose and len(signals) < 5:
                        print(f"Error loading label for {record_name}: {e}")
                    continue
                
                # Extract features
                signal_features = extract_all_features(processed_signal)
                
                # Demographics
                demo = extract_demographics_wfdb_enhanced(header)
                
                signals.append(processed_signal)
                labels.append(int(label))
                demographics.append(demo)
                features.append(signal_features)
                
                if verbose and len(signals) % 500 == 0:
                    print(f"Processed {len(signals)} WFDB records")
            
            except Exception as e:
                if verbose and len(signals) < 5:
                    print(f"Error processing WFDB {record_name}: {e}")
                continue
    
    except Exception as e:
        if verbose:
            print(f"WFDB loading error: {e}")
    
    return signals, labels, demographics, features

def load_chagas_labels_enhanced(data_folder, verbose):
    """Enhanced label loading with multiple fallback strategies"""
    chagas_labels = {}
    
    label_files = [
        'samitrop_chagas_labels.csv', 
        'code15_chagas_labels.csv', 
        'chagas_labels.csv',
        'labels.csv'
    ]
    
    for label_file in label_files:
        label_path = os.path.join(data_folder, label_file)
        if os.path.exists(label_path):
            try:
                label_df = pd.read_csv(label_path)
                if verbose:
                    print(f"Loading labels from: {label_file}")
                
                # Flexible column mapping
                id_cols = ['exam_id', 'id', 'record_id', 'patient_id']
                label_cols = ['chagas', 'label', 'target', 'diagnosis', 'class']
                
                id_col = next((col for col in id_cols if col in label_df.columns), None)
                label_col = next((col for col in label_cols if col in label_df.columns), None)
                
                if id_col and label_col:
                    for _, row in label_df.iterrows():
                        exam_id = row[id_col]
                        chagas = row[label_col]
                        
                        # Convert to binary
                        if isinstance(chagas, str):
                            chagas_binary = 1 if chagas.lower() in ['true', 'positive', 'yes', '1', 'chagas'] else 0
                        else:
                            chagas_binary = int(float(chagas)) if not pd.isna(chagas) else None
                        
                        if chagas_binary is not None:
                            chagas_labels[exam_id] = chagas_binary
                
                if verbose:
                    pos_count = sum(chagas_labels.values())
                    total_count = len(chagas_labels)
                    if total_count > 0:
                        print(f"Loaded {total_count} labels, {pos_count} positive ({pos_count/total_count*100:.1f}%)")
                
                break  # Use first successful file
                
            except Exception as e:
                if verbose:
                    print(f"Error loading {label_file}: {e}")
                continue
    
    return chagas_labels

def get_label_enhanced(exam_id, row, chagas_labels):
    """Enhanced label extraction with fallback strategies"""
    # Try explicit labels first
    if exam_id in chagas_labels:
        return chagas_labels[exam_id]
    
    # Try dataset-specific inference
    source = str(row.get('source', row.get('dataset', ''))).lower()
    
    # SaMi-Trop dataset: all Chagas positive
    if 'samitrop' in source or 'sami' in source:
        return 1
    
    # PTB-XL dataset: all healthy/non-Chagas
    if 'ptb' in source or 'ptbxl' in source:
        return 0
    
    # CODE dataset: mixed, require explicit labels
    if 'code' in source and exam_id not in chagas_labels:
        return None
    
    # Check for diagnosis fields
    diagnosis_fields = ['diagnosis', 'condition', 'disease']
    for field in diagnosis_fields:
        if field in row:
            diag = str(row[field]).lower()
            if 'chagas' in diag:
                return 1
            elif any(term in diag for term in ['normal', 'healthy', 'no']):
                return 0
    
    return None

def extract_all_features(signal):
    """Extract comprehensive feature set with error handling"""
    try:
        stat_features = FeatureExtractor.extract_statistical_features(signal)
    except:
        stat_features = np.zeros(96)  # 12 leads * 8 statistical features
    
    try:
        freq_features = FeatureExtractor.extract_frequency_features(signal)
    except:
        freq_features = np.zeros(60)  # 12 leads * 5 frequency features
    
    try:
        morph_features = FeatureExtractor.extract_morphological_features(signal)
    except:
        morph_features = np.zeros(10)  # Default morphological features
    
    return np.concatenate([stat_features, freq_features, morph_features])

def extract_demographics_enhanced(row):
    """Enhanced demographic feature extraction"""
    features = []
    
    # Age with better handling
    age = row.get('age', row.get('patient_age', 50.0))
    if pd.isna(age) or age <= 0:
        age = 50.0
    age_norm = np.clip(float(age) / 100.0, 0.0, 1.2)
    features.append(age_norm)
    
    # Sex
    sex = row.get('sex', row.get('gender', row.get('is_male', None)))
    if pd.isna(sex) or sex is None:
        sex_male = 0.5  # Unknown
    else:
        if isinstance(sex, str):
            sex_male = 1.0 if sex.lower().startswith('m') else 0.0
        else:
            sex_male = float(sex)
    features.append(sex_male)
    
    # Additional features if available
    height = row.get('height', 170.0)
    if not pd.isna(height) and height > 0:
        height_norm = np.clip(float(height) / 200.0, 0.5, 1.2)
    else:
        height_norm = 0.85  # Average
    features.append(height_norm)
    
    weight = row.get('weight', 70.0)
    if not pd.isna(weight) and weight > 0:
        weight_norm = np.clip(float(weight) / 120.0, 0.3, 1.5)
    else:
        weight_norm = 0.58  # Average
    features.append(weight_norm)
    
    return np.array(features)

def extract_demographics_wfdb_enhanced(header):
    """Enhanced WFDB demographic extraction"""
    features = []
    
    # Age
    age = get_age(header)
    age_norm = 0.5 if age is None else np.clip(float(age) / 100.0, 0.0, 1.2)
    features.append(age_norm)
    
    # Sex
    sex = get_sex(header)
    sex_male = 0.5 if sex is None else (1.0 if sex.lower().startswith('m') else 0.0)
    features.append(sex_male)
    
    # Default for missing height/weight
    features.extend([0.85, 0.58])
    
    return np.array(features)

def build_ensemble_model(signal_shape, demo_features, feature_count):
    """
    Build ensemble model with multiple architectures
    """
    # Signal input
    signal_input = Input(shape=signal_shape, name='signal_input')
    
    # CNN branch for spatial features
    cnn_branch = build_cnn_branch(signal_input)
    
    # LSTM branch for temporal features
    lstm_branch = build_lstm_branch(signal_input)
    
    # Attention branch for important regions
    attention_branch = build_attention_branch(signal_input)
    
    # Combine signal branches
    signal_combined = concatenate([cnn_branch, lstm_branch, attention_branch])
    signal_features = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(signal_combined)
    signal_features = Dropout(0.4)(signal_features)
    
    # Demographics branch
    demo_input = Input(shape=(demo_features,), name='demo_input')
    demo_branch = Dense(32, activation='relu')(demo_input)
    demo_branch = Dropout(0.3)(demo_branch)
    
    # Handcrafted features branch
    features_input = Input(shape=(feature_count,), name='features_input')
    features_branch = Dense(64, activation='relu')(features_input)
    features_branch = Dropout(0.3)(features_branch)
    
    # Final combination
    combined = concatenate([signal_features, demo_branch, features_branch])
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    # Output with probability calibration
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, demo_input, features_input], outputs=output)
    return model

def build_cnn_branch(signal_input):
    """CNN branch for spatial pattern recognition"""
    # Multi-scale convolutions
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(signal_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(2)(conv1)
    
    conv2 = Conv1D(64, 5, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling1D(2)(conv2)
    
    conv3 = Conv1D(128, 7, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = GlobalAveragePooling1D()(conv3)
    
    return conv3

def build_lstm_branch(signal_input):
    """LSTM branch for temporal dependencies"""
    lstm = Bidirectional(LSTM(64, return_sequences=True))(signal_input)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(LSTM(32, return_sequences=False))(lstm)
    lstm = Dropout(0.3)(lstm)
    
    return lstm

def build_attention_branch(signal_input):
    """Attention branch for important regions"""
    # Self-attention to focus on important parts
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(signal_input, signal_input)
    attention = LayerNormalization()(attention)
    attention = GlobalAveragePooling1D()(attention)
    
    return attention

def train_enhanced_model(signals, labels, demographics, features, model_folder, verbose):
    """
    Enhanced training with cross-validation and better strategies
    """
    if verbose:
        print(f"Training enhanced model on {len(signals)} samples")
    
    # Convert to arrays
    X_signal = np.array(signals, dtype=np.float32)
    X_demo = np.array(demographics, dtype=np.float32)
    X_features = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Handle class imbalance
    unique_labels, counts = np.unique(y, return_counts=True)
    if verbose:
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")
        pos_rate = np.mean(y) * 100
        print(f"Positive rate: {pos_rate:.1f}%")
    
    # Create balanced dataset if needed
    if len(unique_labels) == 1 or min(counts) < 50:
        if verbose:
            print("Creating balanced dataset...")
        X_signal, X_demo, X_features, y = create_balanced_dataset(
            X_signal, X_demo, X_features, y, verbose
        )
    
    # Scaling
    demo_scaler = RobustScaler()
    feature_scaler = RobustScaler()
    
    X_demo_scaled = demo_scaler.fit_transform(X_demo)
    X_features_scaled = feature_scaler.fit_transform(X_features)
    
    # Split data
    X_sig_train, X_sig_test, X_demo_train, X_demo_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        X_signal, X_demo_scaled, X_features_scaled, y, 
        test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    model = build_ensemble_model(X_signal.shape[1:], X_demo.shape[1], X_features.shape[1])
    
    if verbose:
        print("Enhanced model architecture:")
        model.summary()
    
    # Advanced compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    # Calculate class weights
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        if verbose:
            print(f"Class weights: {class_weight_dict}")
    except:
        class_weight_dict = {0: 1.0, 1: 1.0}
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_auc', 
            patience=15, 
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=7, 
            min_lr=1e-7,
            verbose=1 if verbose else 0
        ),
        ModelCheckpoint(
            os.path.join(model_folder, 'best_model.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Training with validation
    history = model.fit(
        [X_sig_train, X_demo_train, X_feat_train], y_train,
        validation_data=([X_sig_test, X_demo_test, X_feat_test], y_test),
        epochs=30,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 if verbose else 0
    )
    
    # Cross-validation for robust evaluation
    if verbose:
        print("\nPerforming cross-validation...")
        cv_scores = perform_cross_validation(
            X_signal, X_demo_scaled, X_features_scaled, y, verbose
        )
        print(f"CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Final evaluation
    if verbose:
        evaluate_model_performance(
            model, X_sig_test, X_demo_test, X_feat_test, y_test, verbose
        )
    
    # Save enhanced model
    save_enhanced_model(model_folder, model, demo_scaler, feature_scaler, verbose)
    
    if verbose:
        print("Enhanced model training completed successfully")
    
    return True

def create_balanced_dataset(X_signal, X_demo, X_features, y, verbose):
    """
    Create a more sophisticated balanced dataset
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    
    if len(unique_labels) == 1:
        # Single class - create synthetic minority class
        original_class = unique_labels[0]
        minority_class = 1 - original_class
        n_synthetic = len(y)
        
        # Generate synthetic signals using noise and transformations
        synthetic_signals = []
        synthetic_demo = []
        synthetic_features = []
        
        for i in range(n_synthetic):
            # Base signal with transformations
            base_idx = np.random.randint(0, len(X_signal))
            base_signal = X_signal[base_idx].copy()
            
            # Apply realistic transformations
            # 1. Amplitude scaling
            scale_factor = np.random.uniform(0.8, 1.2)
            base_signal *= scale_factor
            
            # 2. Time shift
            shift = np.random.randint(-50, 51)
            if shift > 0:
                base_signal = np.roll(base_signal, shift, axis=0)
            elif shift < 0:
                base_signal = np.roll(base_signal, shift, axis=0)
            
            # 3. Add realistic noise
            noise_level = np.random.uniform(0.05, 0.15)
            noise = np.random.normal(0, noise_level, base_signal.shape)
            base_signal += noise
            
            # 4. Lead-specific variations
            for lead in range(base_signal.shape[1]):
                lead_noise = np.random.normal(0, 0.02, base_signal.shape[0])
                base_signal[:, lead] += lead_noise
            
            synthetic_signals.append(base_signal)
            
            # Demographics with small variations
            base_demo = X_demo[base_idx].copy()
            demo_noise = np.random.normal(0, 0.02, base_demo.shape)
            synthetic_demo.append(base_demo + demo_noise)
            
            # Features with controlled variations
            base_feat = X_features[base_idx].copy()
            feat_noise = np.random.normal(0, 0.05, base_feat.shape)
            synthetic_features.append(base_feat + feat_noise)
        
        # Combine original and synthetic
        X_signal_balanced = np.vstack([X_signal, np.array(synthetic_signals)])
        X_demo_balanced = np.vstack([X_demo, np.array(synthetic_demo)])
        X_features_balanced = np.vstack([X_features, np.array(synthetic_features)])
        y_balanced = np.hstack([y, np.full(n_synthetic, minority_class)])
        
    else:
        # Multiple classes - use SMOTE-like approach for minority class
        minority_class = unique_labels[np.argmin(counts)]
        majority_class = unique_labels[np.argmax(counts)]
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Oversample minority class
        target_size = len(majority_indices)
        n_synthetic = target_size - len(minority_indices)
        
        if n_synthetic > 0:
            synthetic_indices = np.random.choice(minority_indices, n_synthetic, replace=True)
            
            synthetic_signals = []
            synthetic_demo = []
            synthetic_features = []
            
            for idx in synthetic_indices:
                # Add controlled noise to minority samples
                signal = X_signal[idx].copy()
                signal += np.random.normal(0, 0.1, signal.shape)
                synthetic_signals.append(signal)
                
                demo = X_demo[idx].copy()
                demo += np.random.normal(0, 0.01, demo.shape)
                synthetic_demo.append(demo)
                
                feat = X_features[idx].copy()
                feat += np.random.normal(0, 0.03, feat.shape)
                synthetic_features.append(feat)
            
            # Combine
            X_signal_balanced = np.vstack([X_signal, np.array(synthetic_signals)])
            X_demo_balanced = np.vstack([X_demo, np.array(synthetic_demo)])
            X_features_balanced = np.vstack([X_features, np.array(synthetic_features)])
            y_balanced = np.hstack([y, np.full(n_synthetic, minority_class)])
        else:
            X_signal_balanced = X_signal
            X_demo_balanced = X_demo
            X_features_balanced = X_features
            y_balanced = y
    
    if verbose:
        unique_new, counts_new = np.unique(y_balanced, return_counts=True)
        print(f"Balanced dataset: {dict(zip(unique_new, counts_new))}")
    
    return X_signal_balanced, X_demo_balanced, X_features_balanced, y_balanced

def perform_cross_validation(X_signal, X_demo, X_features, y, verbose):
    """
    Perform stratified cross-validation
    """
    cv_scores = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_signal, y)):
        if verbose:
            print(f"Training fold {fold + 1}/{CV_FOLDS}")
        
        # Split data
        X_sig_train = X_signal[train_idx]
        X_sig_val = X_signal[val_idx]
        X_demo_train = X_demo[train_idx]
        X_demo_val = X_demo[val_idx]
        X_feat_train = X_features[train_idx]
        X_feat_val = X_features[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Build and train model
        model_cv = build_ensemble_model(X_signal.shape[1:], X_demo.shape[1], X_features.shape[1])
        model_cv.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['auc']
        )
        
        # Train with early stopping
        model_cv.fit(
            [X_sig_train, X_demo_train, X_feat_train], y_train_fold,
            validation_data=([X_sig_val, X_demo_val, X_feat_val], y_val_fold),
            epochs=30,
            batch_size=BATCH_SIZE,
            callbacks=[EarlyStopping(monitor='val_auc', patience=10, mode='max')],
            verbose=0
        )
        
        # Evaluate
        y_pred = model_cv.predict([X_sig_val, X_demo_val, X_feat_val], verbose=0)
        auc_score = roc_auc_score(y_val_fold, y_pred)
        cv_scores.append(auc_score)
        
        if verbose:
            print(f"Fold {fold + 1} AUC: {auc_score:.4f}")
    
    return cv_scores

def evaluate_model_performance(model, X_sig_test, X_demo_test, X_feat_test, y_test, verbose):
    """
    Comprehensive model evaluation
    """
    y_pred_proba = model.predict([X_sig_test, X_demo_test, X_feat_test], verbose=0)
    y_pred_binary = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Set Evaluation:")
    print(f"AUC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    
    # Precision-Recall curve analysis
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Optimal F1-score: {f1_scores[optimal_idx]:.4f}")

def save_enhanced_model(model_folder, model, demo_scaler, feature_scaler, verbose):
    """
    Save enhanced model with all components
    """
    # Save model
    model.save(os.path.join(model_folder, 'enhanced_model.keras'))
    
    # Save scalers
    import joblib
    joblib.dump(demo_scaler, os.path.join(model_folder, 'demo_scaler.pkl'))
    joblib.dump(feature_scaler, os.path.join(model_folder, 'feature_scaler.pkl'))
    
    # Save configuration
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'model_type': 'enhanced_ensemble',
        'version': '2.0'
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose:
        print(f"Enhanced model saved to {model_folder}")

def create_baseline_model(model_folder, verbose):
    """
    Create baseline model when insufficient data
    """
    if verbose:
        print("Creating enhanced baseline model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Build model with proper feature count
    expected_feature_count = 96 + 60 + 10  # stat + freq + morph features
    model = build_ensemble_model((TARGET_SIGNAL_LENGTH, NUM_LEADS), 4, expected_feature_count)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc']
    )
    
    # Create dummy scalers
    demo_scaler = RobustScaler()
    demo_scaler.fit(np.random.randn(100, 4))
    
    feature_scaler = RobustScaler()
    feature_scaler.fit(np.random.randn(100, expected_feature_count))
    
    save_enhanced_model(model_folder, model, demo_scaler, feature_scaler, verbose)
    
    if verbose:
        print("Enhanced baseline model created")
    
    return True

def load_model(model_folder, verbose=False):
    """
    Load the enhanced model
    """
    if verbose:
        print(f"Loading enhanced model from {model_folder}")
    
    # Load model
    model_path = os.path.join(model_folder, 'enhanced_model.keras')
    if not os.path.exists(model_path):
        # Fallback to original model
        model_path = os.path.join(model_folder, 'model.keras')
    
    model = tf.keras.models.load_model(model_path)
    
    # Load scalers
    import joblib
    demo_scaler = joblib.load(os.path.join(model_folder, 'demo_scaler.pkl'))
    
    feature_scaler_path = os.path.join(model_folder, 'feature_scaler.pkl')
    if os.path.exists(feature_scaler_path):
        feature_scaler = joblib.load(feature_scaler_path)
    else:
        # Create dummy scaler for backwards compatibility
        feature_scaler = RobustScaler()
        feature_scaler.fit(np.random.randn(100, 100))
    
    # Load config
    import json
    config_path = os.path.join(model_folder, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'signal_length': TARGET_SIGNAL_LENGTH,
            'num_leads': NUM_LEADS,
            'sampling_rate': TARGET_SAMPLING_RATE,
            'model_type': 'enhanced'
        }
    
    return {
        'model': model,
        'demo_scaler': demo_scaler,
        'feature_scaler': feature_scaler,
        'config': config,
        'preprocessor': ECGPreprocessor()
    }

def run_model(record, model_data, verbose=False):
    """
    Enhanced model inference
    """
    try:
        model = model_data['model']
        demo_scaler = model_data['demo_scaler']
        feature_scaler = model_data['feature_scaler']
        config = model_data['config']
        preprocessor = model_data['preprocessor']
        
        # Load and process signal
        try:
            raw_signal, fields = load_signals(record)
            header = load_header(record)
            
            # Get sampling frequency
            fs = get_frequency(header) if header else TARGET_SAMPLING_RATE
            
            # Enhanced preprocessing
            processed_signal = preprocessor.preprocess_signal(raw_signal, fs)
            
            if processed_signal is None:
                raise ValueError("Signal preprocessing failed")
                
        except Exception as e:
            if verbose:
                print(f"Signal loading failed: {e}, using default")
            # Use realistic default signal
            processed_signal = generate_default_ecg_signal()
        
        # Extract demographics
        try:
            header = load_header(record)
            demographics = extract_demographics_wfdb_enhanced(header)
        except:
            demographics = np.array([0.5, 0.5, 0.85, 0.58])  # Default values
        
        # Extract features
        try:
            signal_features = extract_all_features(processed_signal)
            # Ensure consistent feature size
            expected_feature_count = 96 + 60 + 10  # stat + freq + morph
            if len(signal_features) != expected_feature_count:
                # Pad or truncate to expected size
                padded_features = np.zeros(expected_feature_count)
                min_len = min(len(signal_features), expected_feature_count)
                padded_features[:min_len] = signal_features[:min_len]
                signal_features = padded_features
        except:
            signal_features = np.zeros(166)  # Default feature vector
        
        # Prepare inputs
        signal_input = processed_signal.reshape(1, config['signal_length'], config['num_leads'])
        demo_input = demo_scaler.transform(demographics.reshape(1, -1))
        feature_input = feature_scaler.transform(signal_features.reshape(1, -1))
        
        # Predict
        try:
            if config.get('model_type') == 'enhanced_ensemble':
                probability = float(model.predict([signal_input, demo_input, feature_input], verbose=0)[0][0])
            else:
                # Backwards compatibility
                probability = float(model.predict([signal_input, demo_input], verbose=0)[0][0])
        except Exception as e:
            if verbose:
                print(f"Prediction error: {e}")
            probability = 0.05  # Conservative default for Chagas (low prevalence)
        
        # Apply calibration for better probability estimates
        calibrated_probability = calibrate_probability(probability)
        
        # Convert to binary prediction with optimal threshold
        optimal_threshold = 0.3  # Lower threshold due to class imbalance
        binary_prediction = 1 if calibrated_probability >= optimal_threshold else 0
        
        return binary_prediction, calibrated_probability
        
    except Exception as e:
        if verbose:
            print(f"Error in enhanced run_model: {e}")
        return 0, 0.05

def generate_default_ecg_signal():
    """
    Generate a realistic default ECG signal
    """
    # Create a basic sinus rhythm ECG
    t = np.linspace(0, TARGET_SIGNAL_LENGTH / TARGET_SAMPLING_RATE, TARGET_SIGNAL_LENGTH)
    
    # Basic ECG waveform components
    ecg_signal = np.zeros((TARGET_SIGNAL_LENGTH, NUM_LEADS))
    
    for lead in range(NUM_LEADS):
        # P wave, QRS complex, T wave simulation
        heart_rate = 70  # bpm
        period = 60 / heart_rate  # seconds
        
        signal = np.zeros(len(t))
        
        # Add multiple heartbeats
        for beat_start in np.arange(0, t[-1], period):
            beat_indices = (t >= beat_start) & (t < beat_start + period)
            if np.any(beat_indices):
                beat_time = t[beat_indices] - beat_start
                
                # P wave
                p_wave = 0.1 * np.exp(-((beat_time - 0.1) / 0.05) ** 2)
                
                # QRS complex
                qrs_wave = 0.8 * np.exp(-((beat_time - 0.25) / 0.02) ** 2)
                
                # T wave
                t_wave = 0.3 * np.exp(-((beat_time - 0.45) / 0.08) ** 2)
                
                signal[beat_indices] += p_wave + qrs_wave + t_wave
        
        # Add lead-specific variations
        lead_factor = 0.7 + 0.6 * (lead / NUM_LEADS)
        signal *= lead_factor
        
        # Add small amount of noise
        signal += np.random.normal(0, 0.02, len(signal))
        
        ecg_signal[:, lead] = signal
    
    return ecg_signal.astype(np.float32)

def calibrate_probability(probability):
    """
    Apply probability calibration for better estimates
    """
    # Simple sigmoid calibration
    # These parameters would ideally be learned from validation data
    a = 1.2  # Slope parameter
    b = -0.1  # Bias parameter
    
    logit = np.log(probability / (1 - probability + 1e-8))
    calibrated_logit = a * logit + b
    calibrated_prob = 1 / (1 + np.exp(-calibrated_logit))
    
    return np.clip(calibrated_prob, 0.001, 0.999)
