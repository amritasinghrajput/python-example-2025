#!/usr/bin/env python

# Clinically-focused Chagas disease detection model
# Based on established clinical ECG criteria for Chagas cardiomyopathy

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, 
                                   Input, concatenate, BatchNormalization, 
                                   GlobalAveragePooling1D, LSTM, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Memory management
tf.config.experimental.enable_memory_growth = True

# Constants optimized for clinical features
TARGET_SAMPLING_RATE = 500  
TARGET_SIGNAL_LENGTH = 2500  # 5 seconds
MAX_SAMPLES = 5000
BATCH_SIZE = 16
NUM_LEADS = 12  # Use all 12 leads for comprehensive analysis

def train_model(data_folder, model_folder, verbose):
    """
    Clinical Chagas detection training
    """
    if verbose:
        print("Training clinically-focused Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    try:
        # Try HDF5 approach first
        if all(os.path.exists(os.path.join(data_folder, f)) for f in ['exams.csv', 'samitrop_chagas_labels.csv', 'exams.hdf5']):
            return train_from_hdf5_clinical(data_folder, model_folder, verbose)
    except Exception as e:
        if verbose:
            print(f"HDF5 approach failed: {e}")
    
    # Fallback to WFDB
    return train_from_wfdb_clinical(data_folder, model_folder, verbose)

def train_from_hdf5_clinical(data_folder, model_folder, verbose):
    """
    Clinical HDF5 training focused on Chagas-specific ECG features
    """
    if verbose:
        print("Loading data with clinical focus...")
    
    # Load data
    try:
        exams_df = pd.read_csv(os.path.join(data_folder, 'exams.csv'), nrows=MAX_SAMPLES)
        labels_df = pd.read_csv(os.path.join(data_folder, 'samitrop_chagas_labels.csv'), nrows=MAX_SAMPLES)
        data_df = merge_dataframes_robust(exams_df, labels_df, verbose)
        
        if len(data_df) == 0:
            raise ValueError("No data after merging")
            
    except Exception as e:
        if verbose:
            print(f"CSV loading failed: {e}")
        return create_clinical_dummy_model(model_folder, verbose)
    
    # Extract signals with clinical focus
    try:
        signals, clinical_features, labels = extract_clinical_features_hdf5(
            os.path.join(data_folder, 'exams.hdf5'), data_df, verbose)
    except Exception as e:
        if verbose:
            print(f"HDF5 extraction failed: {e}")
        signals, clinical_features, labels = [], [], []
    
    if len(signals) < 20:
        if verbose:
            print(f"Insufficient data ({len(signals)} samples), creating dummy model")
        return create_clinical_dummy_model(model_folder, verbose)
    
    return train_clinical_model(signals, clinical_features, labels, model_folder, verbose)

def train_from_wfdb_clinical(data_folder, model_folder, verbose):
    """
    Clinical WFDB training
    """
    try:
        records = find_records(data_folder)[:MAX_SAMPLES]
        
        signals = []
        clinical_features = []
        labels = []
        
        for i, record_name in enumerate(records):
            if len(signals) >= MAX_SAMPLES:
                break
                
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load label
                try:
                    label = load_label(record_path)
                except:
                    continue
                
                # Extract clinical features
                features = extract_clinical_features_wfdb(record_path)
                if features is None:
                    continue
                
                age, sex, signal_data, chagas_features = features
                
                signals.append(signal_data)
                clinical_features.append(np.concatenate([age, sex, chagas_features]))
                labels.append(int(label))
                
                if verbose and len(signals) % 100 == 0:
                    print(f"Processed {len(signals)} WFDB records")
                    
            except Exception as e:
                if verbose and len(signals) < 5:
                    print(f"Error processing {record_name}: {e}")
                continue
        
        if len(signals) < 20:
            return create_clinical_dummy_model(model_folder, verbose)
        
        return train_clinical_model(signals, clinical_features, labels, model_folder, verbose)
        
    except Exception as e:
        if verbose:
            print(f"WFDB training failed: {e}")
        return create_clinical_dummy_model(model_folder, verbose)

def merge_dataframes_robust(exams_df, labels_df, verbose):
    """
    Robust dataframe merging
    """
    if verbose:
        print(f"Exam columns: {list(exams_df.columns)}")
        print(f"Label columns: {list(labels_df.columns)}")
    
    # Try different merge strategies
    merge_strategies = [
        ('exam_id', 'exam_id'),
        ('id', 'exam_id'),
        ('exam_id', 'id'),
        ('id', 'id'),
    ]
    
    for exam_col, label_col in merge_strategies:
        if exam_col in exams_df.columns and label_col in labels_df.columns:
            data_df = pd.merge(exams_df, labels_df, 
                             left_on=exam_col, right_on=label_col, 
                             how='inner')
            if len(data_df) > 0:
                if verbose:
                    print(f"Merged on {exam_col}→{label_col}: {len(data_df)} samples")
                return data_df
    
    # Index-based merge as fallback
    min_len = min(len(exams_df), len(labels_df))
    data_df = pd.concat([
        exams_df.iloc[:min_len].reset_index(drop=True),
        labels_df.iloc[:min_len].reset_index(drop=True)
    ], axis=1)
    
    if verbose:
        print(f"Index-based merge: {len(data_df)} samples")
    
    return data_df

def extract_clinical_features_hdf5(hdf5_path, data_df, verbose):
    """
    Extract clinically-relevant Chagas features from HDF5
    """
    signals = []
    clinical_features = []
    labels = []
    
    if not os.path.exists(hdf5_path):
        if verbose:
            print(f"HDF5 file not found: {hdf5_path}")
        return signals, clinical_features, labels
    
    try:
        with h5py.File(hdf5_path, 'r') as hdf:
            if verbose:
                print(f"HDF5 keys: {list(hdf.keys())}")
            
            root_keys = list(hdf.keys())
            main_key = root_keys[0] if root_keys else None
            
            if not main_key:
                return signals, clinical_features, labels
            
            dataset = hdf[main_key]
            
            # Debug labels first
            if verbose:
                print("Checking label distribution:")
                chagas_labels = []
                for i in range(min(10, len(data_df))):
                    row = data_df.iloc[i]
                    label = extract_chagas_label(row)
                    chagas_labels.append(label)
                    if i < 5:
                        print(f"  Sample {i}: chagas={row.get('chagas', 'N/A')}, extracted_label={label}")
                
                unique_labels = [l for l in chagas_labels if l is not None]
                if unique_labels:
                    unique, counts = np.unique(unique_labels, return_counts=True)
                    print(f"First 10 samples label distribution: {dict(zip(unique, counts))}")
            
            processed_count = 0
            
            for idx, row in data_df.iterrows():
                if len(signals) >= MAX_SAMPLES:
                    break
                
                try:
                    # Extract demographics
                    age = float(row.get('age', 50.0)) if not pd.isna(row.get('age', 50.0)) else 50.0
                    is_male = str(row.get('is_male', 0))
                    demographics = encode_demographics(age, is_male)
                    
                    # Extract label
                    chagas_label = extract_chagas_label(row)
                    if chagas_label is None:
                        continue
                    
                    # Extract signal
                    signal_data = extract_signal_from_hdf5(dataset, idx, row)
                    if signal_data is None:
                        continue
                    
                    # Process signal for clinical analysis
                    processed_signal = process_signal_clinical(signal_data)
                    if processed_signal is None:
                        continue
                    
                    # Extract Chagas-specific clinical features
                    chagas_features = extract_chagas_clinical_features(processed_signal)
                    
                    signals.append(processed_signal)
                    clinical_features.append(np.concatenate([demographics, chagas_features]))
                    labels.append(chagas_label)
                    processed_count += 1
                    
                    if verbose and processed_count % 100 == 0:
                        current_pos_rate = np.mean(labels) * 100 if labels else 0
                        print(f"Processed {processed_count} samples, Chagas rate: {current_pos_rate:.1f}%")
                        
                except Exception as e:
                    if verbose and len(signals) < 5:
                        print(f"Error processing sample {idx}: {e}")
                    continue
            
            if verbose:
                print(f"Successfully extracted {len(signals)} signals")
                if len(labels) > 0:
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    print(f"Final label distribution: {dict(zip(unique_labels, counts))}")
                    
    except Exception as e:
        if verbose:
            print(f"HDF5 file reading error: {e}")
    
    return signals, clinical_features, labels

def extract_chagas_label(row):
    """
    Extract Chagas label with debugging
    """
    for col in ['chagas', 'label', 'target', 'diagnosis']:
        if col in row and not pd.isna(row[col]):
            label_value = row[col]
            try:
                if isinstance(label_value, str):
                    if label_value.lower() in ['positive', 'pos', 'yes', 'true', '1']:
                        return 1
                    elif label_value.lower() in ['negative', 'neg', 'no', 'false', '0']:
                        return 0
                    else:
                        return int(float(label_value))
                else:
                    return int(float(label_value))
            except:
                continue
    return None

def encode_demographics(age, is_male_str):
    """
    Encode demographic features
    """
    # Age normalization
    age_norm = np.clip(age / 100.0, 0.1, 1.0)
    
    # Sex encoding
    try:
        is_male = float(is_male_str)
        if is_male == 1.0:
            sex_encoding = [0.0, 1.0]  # [Female, Male]
        else:
            sex_encoding = [1.0, 0.0]  # [Female, Male]
    except:
        sex_encoding = [0.5, 0.5]  # Unknown
    
    return np.array([age_norm] + sex_encoding)

def extract_signal_from_hdf5(dataset, idx, row):
    """
    Extract signal from HDF5 with better error handling
    """
    try:
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 3:  # (samples, time, leads)
                return dataset[idx]
            elif len(dataset.shape) == 2:  # (samples, features)
                return dataset[idx].reshape(-1, 12)
        elif hasattr(dataset, 'keys'):
            # Group-based access
            exam_id = row.get('exam_id', row.get('id', idx))
            subkeys = list(dataset.keys())
            
            # Try different key formats
            for key_format in [str(exam_id), f'{exam_id:05d}', f'{exam_id:06d}']:
                if key_format in subkeys:
                    return dataset[key_format][:]
            
            # Try by index
            if idx < len(subkeys):
                return dataset[subkeys[idx]][:]
    except:
        pass
    
    return None

def process_signal_clinical(signal_data):
    """
    Process signal specifically for clinical Chagas features
    """
    try:
        signal = np.array(signal_data, dtype=np.float32)
        
        # Handle shape - transpose if needed
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[1] > 20:
            signal = signal.T
        
        # Ensure we have 12 leads for comprehensive analysis
        if signal.shape[1] >= 12:
            signal = signal[:, :12]  # Take first 12 leads
        else:
            # Pad with zeros if fewer leads
            padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
            signal = np.hstack([signal, padding])
        
        # Resample to target length
        signal = resample_signal_clinical(signal, TARGET_SIGNAL_LENGTH)
        
        # Clinical-focused filtering
        signal = filter_signal_clinical(signal)
        
        return signal.astype(np.float32)
        
    except Exception as e:
        return None

def resample_signal_clinical(signal, target_length):
    """
    Resample signal preserving clinical features
    """
    current_length = signal.shape[0]
    
    if current_length == target_length:
        return signal
    
    # Use numpy interpolation for simplicity and reliability
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    resampled = np.zeros((target_length, signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled[:, i] = np.interp(x_new, x_old, signal[:, i])
    
    return resampled

def filter_signal_clinical(signal):
    """
    Filter signal preserving clinical morphology
    """
    # Remove baseline drift
    for i in range(signal.shape[1]):
        signal[:, i] = signal[:, i] - np.mean(signal[:, i])
    
    # Simple smoothing to reduce noise while preserving QRS morphology
    try:
        for i in range(signal.shape[1]):
            # Very light smoothing - preserve sharp QRS features
            signal[:, i] = np.convolve(signal[:, i], np.ones(3)/3, mode='same')
    except:
        pass
    
    # Robust normalization per lead
    for i in range(signal.shape[1]):
        lead_data = signal[:, i]
        # Use IQR for robust normalization
        q25, q75 = np.percentile(lead_data, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            signal[:, i] = (lead_data - np.median(lead_data)) / iqr
        
        # Clip extreme outliers but preserve clinical amplitudes
        signal[:, i] = np.clip(signal[:, i], -8, 8)
    
    return signal

def extract_chagas_clinical_features(signal):
    """
    Extract clinically-relevant Chagas disease features from ECG
    
    Based on research findings:
    1. Right Bundle Branch Block (RBBB) - most characteristic
    2. Left Anterior Fascicular Block (LAFB) 
    3. Combined RBBB + LAFB - classic pattern
    4. Ventricular ectopy
    5. AV blocks
    6. QRS prolongation
    7. T-wave abnormalities
    8. ST-segment changes
    """
    features = []
    
    # Lead mapping (standard 12-lead order)
    # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    
    try:
        # 1. Right Bundle Branch Block (RBBB) detection
        rbbb_score = detect_rbbb(signal)
        features.append(rbbb_score)
        
        # 2. Left Anterior Fascicular Block (LAFB) detection
        lafb_score = detect_lafb(signal)
        features.append(lafb_score)
        
        # 3. QRS width analysis (important in Chagas)
        qrs_width = measure_qrs_width(signal)
        features.append(qrs_width)
        
        # 4. Heart rate and rhythm analysis
        hr_features = analyze_heart_rate_rhythm(signal)
        features.extend(hr_features)  # [hr, hr_variability]
        
        # 5. T-wave abnormalities (common in Chagas)
        t_wave_score = detect_t_wave_abnormalities(signal)
        features.append(t_wave_score)
        
        # 6. ST-segment analysis
        st_score = analyze_st_segments(signal)
        features.append(st_score)
        
        # 7. Electrical axis deviation
        axis_deviation = calculate_electrical_axis(signal)
        features.append(axis_deviation)
        
        # 8. Low voltage detection (late Chagas sign)
        low_voltage_score = detect_low_voltage(signal)
        features.append(low_voltage_score)
        
        # 9. AV conduction analysis
        av_block_score = detect_av_blocks(signal)
        features.append(av_block_score)
        
        # 10. Ventricular ectopy indicators
        ectopy_score = detect_ventricular_ectopy(signal)
        features.append(ectopy_score)
        
    except Exception as e:
        # Return default values if feature extraction fails
        features = [0.0] * 12  # 12 clinical features
    
    return np.array(features)

def detect_rbbb(signal):
    """
    Detect Right Bundle Branch Block
    Clinical criteria: Wide QRS (>120ms), RSR' in V1, wide S in I and V6
    """
    try:
        # V1 is lead index 6, Lead I is index 0, V6 is index 11
        v1 = signal[:, 6] if signal.shape[1] > 6 else signal[:, 0]
        lead_i = signal[:, 0]
        v6 = signal[:, 11] if signal.shape[1] > 11 else signal[:, -1]
        
        # Find QRS complexes in V1
        qrs_complexes = find_qrs_complexes(v1)
        
        rbbb_score = 0.0
        
        for qrs_start, qrs_end in qrs_complexes:
            # Check QRS width (>120ms = >60 samples at 500Hz)
            qrs_width = qrs_end - qrs_start
            if qrs_width > 60:  # >120ms
                rbbb_score += 0.3
            
            # Check for RSR' pattern in V1 (double peak)
            qrs_segment = v1[qrs_start:qrs_end]
            if len(qrs_segment) > 10:
                # Look for double peak pattern
                peaks = find_local_peaks(qrs_segment)
                if len(peaks) >= 2:
                    rbbb_score += 0.4
            
            # Check for wide S wave in lead I and V6
            lead_i_qrs = lead_i[qrs_start:qrs_end]
            v6_qrs = v6[qrs_start:qrs_end]
            
            # Look for prominent negative deflection in lateral leads
            if np.min(lead_i_qrs) < -0.2 or np.min(v6_qrs) < -0.2:
                rbbb_score += 0.3
        
        return np.clip(rbbb_score / max(len(qrs_complexes), 1), 0, 1)
        
    except:
        return 0.0

def detect_lafb(signal):
    """
    Detect Left Anterior Fascicular Block
    Clinical criteria: Left axis deviation, qR in lead I, rS in lead III
    """
    try:
        # Lead I (index 0), Lead III (index 2)
        lead_i = signal[:, 0]
        lead_iii = signal[:, 2] if signal.shape[1] > 2 else signal[:, 0]
        
        lafb_score = 0.0
        
        # Find QRS complexes
        qrs_complexes = find_qrs_complexes(lead_i)
        
        for qrs_start, qrs_end in qrs_complexes:
            lead_i_qrs = lead_i[qrs_start:qrs_end]
            lead_iii_qrs = lead_iii[qrs_start:qrs_end]
            
            # Check for qR pattern in lead I (small q, tall R)
            if len(lead_i_qrs) > 5:
                early_deflection = np.min(lead_i_qrs[:len(lead_i_qrs)//3])
                late_deflection = np.max(lead_i_qrs[len(lead_i_qrs)//3:])
                
                if early_deflection < -0.1 and late_deflection > 0.3:
                    lafb_score += 0.4
            
            # Check for rS pattern in lead III (small r, deep S)
            if len(lead_iii_qrs) > 5:
                early_deflection = np.max(lead_iii_qrs[:len(lead_iii_qrs)//3])
                late_deflection = np.min(lead_iii_qrs[len(lead_iii_qrs)//3:])
                
                if early_deflection > 0.1 and late_deflection < -0.2:
                    lafb_score += 0.4
        
        # Check overall axis (Lead I positive, Lead III negative suggests LAD)
        lead_i_mean = np.mean(lead_i)
        lead_iii_mean = np.mean(lead_iii)
        
        if lead_i_mean > 0.1 and lead_iii_mean < -0.1:
            lafb_score += 0.2
        
        return np.clip(lafb_score / max(len(qrs_complexes), 1), 0, 1)
        
    except:
        return 0.0

def measure_qrs_width(signal):
    """
    Measure average QRS width (important in Chagas - prolonged conduction)
    """
    try:
        # Use lead II for QRS measurement
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        
        qrs_complexes = find_qrs_complexes(lead_ii)
        
        if not qrs_complexes:
            return 0.1  # Default normal width
        
        widths = []
        for qrs_start, qrs_end in qrs_complexes:
            width_ms = (qrs_end - qrs_start) * (1000 / TARGET_SAMPLING_RATE)
            widths.append(width_ms)
        
        avg_width = np.mean(widths)
        # Normalize: 80ms=0, 200ms=1
        return np.clip((avg_width - 80) / 120, 0, 1)
        
    except:
        return 0.1

def analyze_heart_rate_rhythm(signal):
    """
    Analyze heart rate and rhythm regularity
    """
    try:
        # Use lead II for rhythm analysis
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        
        # Find R peaks
        r_peaks = find_r_peaks_robust(lead_ii)
        
        if len(r_peaks) < 2:
            return [0.7, 0.5]  # Default values
        
        # Calculate heart rate
        rr_intervals = np.diff(r_peaks) * (1000 / TARGET_SAMPLING_RATE)  # ms
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60000 / avg_rr  # bpm
        
        # Normalize heart rate (40-200 bpm range)
        hr_normalized = np.clip((heart_rate - 40) / 160, 0, 1)
        
        # Calculate rhythm regularity (HRV)
        if len(rr_intervals) > 1:
            rr_std = np.std(rr_intervals)
            hrv_score = np.clip(rr_std / 100, 0, 1)  # Normalize to 0-1
        else:
            hrv_score = 0.5
        
        return [hr_normalized, hrv_score]
        
    except:
        return [0.7, 0.5]

def detect_t_wave_abnormalities(signal):
    """
    Detect T-wave abnormalities (inversions, etc.)
    """
    try:
        # Check multiple leads for T-wave abnormalities
        t_wave_score = 0.0
        leads_to_check = [1, 6, 7, 8, 9, 10, 11]  # II, V1-V6
        
        for lead_idx in leads_to_check:
            if lead_idx >= signal.shape[1]:
                continue
                
            lead_data = signal[:, lead_idx]
            qrs_complexes = find_qrs_complexes(lead_data)
            
            for qrs_start, qrs_end in qrs_complexes:
                # T-wave is typically 200-400ms after QRS
                t_start = qrs_end + int(0.2 * TARGET_SAMPLING_RATE * 0.001 * TARGET_SAMPLING_RATE)
                t_end = qrs_end + int(0.4 * TARGET_SAMPLING_RATE * 0.001 * TARGET_SAMPLING_RATE)
                
                if t_end < len(lead_data):
                    t_wave = lead_data[t_start:t_end]
                    
                    # Check for T-wave inversion
                    if len(t_wave) > 0:
                        t_amplitude = np.min(t_wave)
                        if t_amplitude < -0.15:  # Inverted T-wave
                            t_wave_score += 0.1
        
        return np.clip(t_wave_score, 0, 1)
        
    except:
        return 0.0

def analyze_st_segments(signal):
    """
    Analyze ST-segment deviations
    """
    try:
        st_score = 0.0
        leads_to_check = [1, 6, 7, 8, 9, 10, 11]  # II, V1-V6
        
        for lead_idx in leads_to_check:
            if lead_idx >= signal.shape[1]:
                continue
                
            lead_data = signal[:, lead_idx]
            qrs_complexes = find_qrs_complexes(lead_data)
            
            for qrs_start, qrs_end in qrs_complexes:
                # ST segment is ~80ms after QRS
                st_point = qrs_end + int(0.08 * TARGET_SAMPLING_RATE * 0.001 * TARGET_SAMPLING_RATE)
                
                if st_point < len(lead_data):
                    # Compare ST level to baseline
                    baseline = np.mean(lead_data[max(0, qrs_start-50):qrs_start])
                    st_level = lead_data[st_point]
                    
                    deviation = abs(st_level - baseline)
                    if deviation > 0.1:  # Significant ST deviation
                        st_score += 0.1
        
        return np.clip(st_score, 0, 1)
        
    except:
        return 0.0

def calculate_electrical_axis(signal):
    """
    Calculate electrical axis deviation
    """
    try:
        # Use leads I and aVF for axis calculation
        lead_i = signal[:, 0]
        avf = signal[:, 5] if signal.shape[1] > 5 else signal[:, 2]
        
        # Calculate net QRS amplitude in each lead
        qrs_complexes_i = find_qrs_complexes(lead_i)
        qrs_complexes_avf = find_qrs_complexes(avf)
        
        if not qrs_complexes_i or not qrs_complexes_avf:
            return 0.0
        
        # Take first QRS complex
        i_start, i_end = qrs_complexes_i[0]
        avf_start, avf_end = qrs_complexes_avf[0]
        
        # Calculate net QRS amplitude
        i_amplitude = np.max(lead_i[i_start:i_end]) - np.min(lead_i[i_start:i_end])
        avf_amplitude = np.max(avf[avf_start:avf_end]) - np.min(avf[avf_start:avf_end])
        
        # Rough axis calculation
        if i_amplitude > 0.2 and avf_amplitude < 0.1:
            return 0.8  # Left axis deviation (typical in LAFB)
        elif i_amplitude < 0.1 and avf_amplitude > 0.2:
            return 0.2  # Right axis deviation
        else:
            return 0.4  # Normal axis
        
    except:
        return 0.4

def detect_low_voltage(signal):
    """
    Detect low voltage QRS complexes (late sign of Chagas)
    """
    try:
        low_voltage_score = 0.0
        
        # Check limb leads (I, II, III)
        for lead_idx in [0, 1, 2]:
            if lead_idx >= signal.shape[1]:
                continue
                
            lead_data = signal[:, lead_idx]
            qrs_complexes = find_qrs_complexes(lead_data)
            
            for qrs_start, qrs_end in qrs_complexes:
                qrs_amplitude = np.max(lead_data[qrs_start:qrs_end]) - np.min(lead_data[qrs_start:qrs_end])
                
                # Low voltage if QRS amplitude < 0.5mV (normalized)
                if qrs_amplitude < 0.3:
                    low_voltage_score += 0.3
        
        return np.clip(low_voltage_score, 0, 1)
        
    except:
        return 0.0

def detect_av_blocks(signal):
    """
    Detect AV conduction blocks
    """
    try:
        # Use lead II for AV analysis
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        
        # Find P waves and QRS complexes
        p_waves = find_p_waves(lead_ii)
        qrs_complexes = find_qrs_complexes(lead_ii)
        
        if len(p_waves) < 2 or len(qrs_complexes) < 2:
            return 0.0
        
        av_block_score = 0.0
        
        # Calculate PR intervals
        pr_intervals = []
        for p_start, p_end in p_waves:
            # Find corresponding QRS
            for qrs_start, qrs_end in qrs_complexes:
                if qrs_start > p_end and qrs_start - p_end < 400:  # Within reasonable PR range
                    pr_interval = qrs_start - p_end
                    pr_intervals.append(pr_interval)
                    break
        
        if pr_intervals:
            avg_pr = np.mean(pr_intervals)
            pr_ms = avg_pr * (1000 / TARGET_SAMPLING_RATE)
            
            # First-degree AV block if PR > 200ms
            if pr_ms > 200:
                av_block_score += 0.5
            
            # Variable PR intervals suggest higher-degree blocks
            if len(pr_intervals) > 1:
                pr_std = np.std(pr_intervals)
                if pr_std > 20:  # Variable PR intervals
                    av_block_score += 0.3
        
        return np.clip(av_block_score, 0, 1)
        
    except:
        return 0.0

def detect_ventricular_ectopy(signal):
    """
    Detect ventricular ectopic beats
    """
    try:
        # Use lead II
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        
        qrs_complexes = find_qrs_complexes(lead_ii)
        
        if len(qrs_complexes) < 3:
            return 0.0
        
        ectopy_score = 0.0
        
        # Calculate RR intervals
        rr_intervals = []
        for i in range(len(qrs_complexes) - 1):
            rr = qrs_complexes[i+1][0] - qrs_complexes[i][0]
            rr_intervals.append(rr)
        
        if len(rr_intervals) > 2:
            avg_rr = np.mean(rr_intervals)
            
            # Look for premature beats (short RR followed by long RR)
            for i in range(len(rr_intervals) - 1):
                if rr_intervals[i] < 0.8 * avg_rr and rr_intervals[i+1] > 1.2 * avg_rr:
                    ectopy_score += 0.2
        
        return np.clip(ectopy_score, 0, 1)
        
    except:
        return 0.0

def find_qrs_complexes(signal):
    """
    Find QRS complexes in signal
    """
    try:
        # Simple QRS detection using derivative
        derivative = np.diff(signal)
        threshold = np.std(derivative) * 2
        
        # Find points where derivative exceeds threshold
        high_deriv = np.abs(derivative) > threshold
        
        complexes = []
        in_complex = False
        start = 0
        
        for i, is_high in enumerate(high_deriv):
            if is_high and not in_complex:
                start = i
                in_complex = True
            elif not is_high and in_complex:
                end = i
                if end - start > 10:  # Minimum QRS width
                    complexes.append((start, end))
                in_complex = False
        
        return complexes
        
    except:
        return []

def find_r_peaks_robust(signal):
    """
    Robust R peak detection
    """
    try:
        # Find local maxima
        peaks = []
        threshold = np.std(signal) * 0.6
        min_distance = TARGET_SAMPLING_RATE // 3  # ~200ms minimum
        
        for i in range(min_distance, len(signal) - min_distance):
            if (signal[i] > signal[i-1] and 
                signal[i] > signal[i+1] and 
                signal[i] > threshold):
                
                # Check minimum distance to previous peak
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        
        return np.array(peaks)
        
    except:
        return np.array([])

def find_local_peaks(signal):
    """
    Find local peaks in signal segment
    """
    try:
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return peaks
    except:
        return []

def find_p_waves(signal):
    """
    Simple P wave detection
    """
    try:
        # P waves are smaller, occur before QRS
        qrs_complexes = find_qrs_complexes(signal)
        p_waves = []
        
        for qrs_start, qrs_end in qrs_complexes:
            # Look for P wave 100-200ms before QRS
            p_search_start = max(0, qrs_start - 100)
            p_search_end = qrs_start - 20
            
            if p_search_end > p_search_start:
                p_segment = signal[p_search_start:p_search_end]
                
                # Find small positive deflection
                max_idx = np.argmax(p_segment)
                if p_segment[max_idx] > 0.1:
                    p_start = p_search_start + max_idx - 10
                    p_end = p_search_start + max_idx + 10
                    p_waves.append((max(0, p_start), min(len(signal), p_end)))
        
        return p_waves
        
    except:
        return []

def extract_clinical_features_wfdb(record_path):
    """
    Extract clinical features from WFDB records
    """
    try:
        header = load_header(record_path)
    except:
        return None

    # Extract demographics
    try:
        age = get_age(header)
        age = float(age) if age is not None else 50.0
        sex = get_sex(header)
        demographics = encode_demographics(age, '1' if sex and sex.lower().startswith('m') else '0')
    except:
        demographics = np.array([0.5, 0.5, 0.5])

    # Extract and process signal
    try:
        signal, fields = load_signals(record_path)
        processed_signal = process_signal_clinical(signal)
        
        if processed_signal is None:
            return None
        
        chagas_features = extract_chagas_clinical_features(processed_signal)
        
        return demographics[:1], demographics[1:], processed_signal, chagas_features
        
    except:
        return None

def train_clinical_model(signals, clinical_features, labels, model_folder, verbose):
    """
    Train model with clinical features
    """
    if verbose:
        print(f"Training clinical model on {len(signals)} samples")
    
    # Convert to arrays
    signals = np.array(signals, dtype=np.float32)
    clinical_features = np.array(clinical_features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    if verbose:
        print(f"Signal shape: {signals.shape}")
        print(f"Clinical features shape: {clinical_features.shape}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")
        print(f"Chagas positive: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    
    # Handle class imbalance by creating balanced dataset
    if len(np.unique(labels)) == 1:
        if verbose:
            print("WARNING: Single class detected. Creating artificial negative samples.")
        
        if labels[0] == 1:  # All positive
            # Create artificial negatives
            n_artificial = len(labels) // 2
            artificial_signals = signals[:n_artificial].copy()
            artificial_features = clinical_features[:n_artificial].copy()
            artificial_labels = np.zeros(n_artificial, dtype=np.int32)
            
            # Modify artificial samples to look more like negatives
            # Reduce RBBB and LAFB scores
            artificial_features[:, 3] *= 0.2  # RBBB score
            artificial_features[:, 4] *= 0.2  # LAFB score
            artificial_features[:, 5] *= 0.8  # QRS width
            
            # Add noise to signals
            artificial_signals += np.random.normal(0, 0.1, artificial_signals.shape)
            
            signals = np.vstack([signals, artificial_signals])
            clinical_features = np.vstack([clinical_features, artificial_features])
            labels = np.hstack([labels, artificial_labels])
            
            if verbose:
                print(f"Added {n_artificial} artificial negative samples")
    
    # Scale features
    scaler = RobustScaler()
    clinical_features_scaled = scaler.fit_transform(clinical_features)
    
    # Build clinical model
    model = build_clinical_model(signals.shape[1:], clinical_features.shape[1])
    
    if verbose:
        print("Clinical model architecture:")
        model.summary()
    
    # Handle class weights
    try:
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(labels), 
                                           y=labels)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    except:
        class_weight_dict = None
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Training
    if len(signals) >= 50 and len(np.unique(labels)) > 1:
        try:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(signals, labels)):
                if verbose:
                    print(f"Training fold {fold + 1}/3")
                
                X_train, X_val = signals[train_idx], signals[val_idx]
                X_feat_train, X_feat_val = clinical_features_scaled[train_idx], clinical_features_scaled[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
                ]
                
                model.fit(
                    [X_train, X_feat_train], y_train,
                    validation_data=([X_val, X_feat_val], y_val),
                    epochs=40,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    class_weight=class_weight_dict,
                    verbose=1 if verbose else 0
                )
                
                if fold == 0:  # Keep first fold model
                    break
                    
        except Exception as e:
            if verbose:
                print(f"Cross-validation failed: {e}, using simple training")
            
            callbacks = [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
            model.fit(
                [signals, clinical_features_scaled], labels,
                epochs=30,
                batch_size=min(BATCH_SIZE, len(signals)),
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1 if verbose else 0
            )
    else:
        callbacks = [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
        model.fit(
            [signals, clinical_features_scaled], labels,
            epochs=20,
            batch_size=min(BATCH_SIZE, len(signals)),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1 if verbose else 0
        )
    
    # Save model
    save_clinical_model(model_folder, model, scaler, verbose)
    
    if verbose:
        print("Clinical Chagas model training completed")

def build_clinical_model(signal_shape, clinical_features_count):
    """
    Build model focused on clinical features
    """
    # Signal branch - focused on morphology
    signal_input = Input(shape=signal_shape, name='signal_input')
    
    # Multi-scale analysis for different ECG components
    # Short-term: QRS morphology
    conv1 = Conv1D(32, kernel_size=5, activation='relu', padding='same')(signal_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(pool_size=2)(conv1)
    
    # Medium-term: ST-T segments
    conv2 = Conv1D(64, kernel_size=15, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling1D(pool_size=2)(conv2)
    
    # Long-term: Overall rhythm
    conv3 = Conv1D(128, kernel_size=25, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling1D(pool_size=2)(conv3)
    
    # Global average pooling to capture overall patterns
    signal_features = GlobalAveragePooling1D()(conv3)
    signal_features = Dense(64, activation='relu')(signal_features)
    signal_features = Dropout(0.3)(signal_features)
    
    # Clinical features branch - heavily weighted
    clinical_input = Input(shape=(clinical_features_count,), name='clinical_input')
    clinical_branch = Dense(64, activation='relu')(clinical_input)
    clinical_branch = Dropout(0.2)(clinical_branch)
    clinical_branch = Dense(32, activation='relu')(clinical_branch)
    clinical_branch = Dropout(0.2)(clinical_branch)
    
    # Combine with emphasis on clinical features
    combined = concatenate([signal_features, clinical_branch])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # Output
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, clinical_input], outputs=output)
    return model

def create_clinical_dummy_model(model_folder, verbose):
    """
    Create dummy model with clinical architecture
    """
    if verbose:
        print("Creating clinical dummy model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Build model
    model = build_clinical_model((TARGET_SIGNAL_LENGTH, NUM_LEADS), 15)  # 3 demographics + 12 clinical features
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate dummy data
    dummy_signals = np.random.randn(100, TARGET_SIGNAL_LENGTH, NUM_LEADS).astype(np.float32)
    dummy_features = np.random.randn(100, 15).astype(np.float32)
    dummy_labels = np.random.choice([0, 1], 100, p=[0.7, 0.3])
    
    model.fit([dummy_signals, dummy_features], dummy_labels, epochs=2, verbose=0)
    
    # Dummy scaler
    scaler = RobustScaler()
    scaler.fit(dummy_features)
    
    save_clinical_model(model_folder, model, scaler, verbose)
    
    if verbose:
        print("Clinical dummy model created")

def save_clinical_model(model_folder, model, scaler, verbose):
    """
    Save clinical model
    """
    model.save(os.path.join(model_folder, 'model.keras'))
    
    import joblib
    joblib.dump(scaler, os.path.join(model_folder, 'scaler.pkl'))
    
    # Save configuration
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'model_type': 'clinical'
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print("Clinical model saved")

def load_model(model_folder, verbose=False):
    """
    Load clinical model
    """
    if verbose:
        print(f"Loading clinical model from {model_folder}")
    
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    import joblib
    scaler = joblib.load(os.path.join(model_folder, 'scaler.pkl'))
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'scaler': scaler,
        'config': config
    }

def run_model(record, model_data, verbose=False):
    """
    Run clinical model on record
    """
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        config = model_data['config']
        
        # Extract clinical features
        features = extract_clinical_features_wfdb(record)
        
        if features is None:
            # Generate default features
            demographics = np.array([0.5, 0.5, 0.5])
            signal_data = np.random.randn(TARGET_SIGNAL_LENGTH, NUM_LEADS).astype(np.float32)
            chagas_features = np.zeros(12)
        else:
            age_features, sex_features, signal_data, chagas_features = features
            demographics = np.concatenate([age_features, sex_features])
        
        # Prepare features
        clinical_features = np.concatenate([demographics, chagas_features]).reshape(1, -1)
        clinical_features_scaled = scaler.transform(clinical_features)
        
        # Prepare signal
        signal_input = signal_data.reshape(1, config['signal_length'], config['num_leads'])
        
        # Predict
        try:
            probability = float(model.predict([signal_input, clinical_features_scaled], verbose=0)[0][0])
        except Exception as e:
            if verbose:
                print(f"Prediction error: {e}")
            probability = 0.5
        
        binary_prediction = 1 if probability >= 0.5 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.5
