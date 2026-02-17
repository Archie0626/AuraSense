"""
Data preprocessing utilities for HAR system
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class HARPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.sampling_rate = 50  # Hz (typical for smartphone sensors)
        
    def load_uci_har_data(self, data_path=None):
        """
        Load actual UCI HAR dataset
        If data_path is None, generates synthetic data for demonstration
        """
        if data_path:
            # Load actual UCI HAR data
            try:
                # Load features
                X_train = pd.read_csv(f"{data_path}/train/X_train.txt", delim_whitespace=True, header=None)
                X_test = pd.read_csv(f"{data_path}/test/X_test.txt", delim_whitespace=True, header=None)
                
                # Load labels
                y_train = pd.read_csv(f"{data_path}/train/y_train.txt", delim_whitespace=True, header=None)
                y_test = pd.read_csv(f"{data_path}/test/y_test.txt", delim_whitespace=True, header=None)
                
                # Load subject IDs
                subject_train = pd.read_csv(f"{data_path}/train/subject_train.txt", delim_whitespace=True, header=None)
                subject_test = pd.read_csv(f"{data_path}/test/subject_test.txt", delim_whitespace=True, header=None)
                
                # Combine train and test
                X = pd.concat([X_train, X_test], axis=0).values
                y = pd.concat([y_train, y_test], axis=0).values.ravel()
                subjects = pd.concat([subject_train, subject_test], axis=0).values.ravel()
                
                # Activity labels mapping
                activity_labels = {
                    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
                    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
                }
                
                return X, y, subjects, activity_labels
                
            except Exception as e:
                print(f"Error loading UCI HAR data: {e}")
                return None, None, None, None
        else:
            # Generate synthetic data for demonstration
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic HAR data for demonstration"""
        np.random.seed(42)
        n_samples = 10000
        n_features = 561  # UCI HAR has 561 features
        
        activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                     'SITTING', 'STANDING', 'LAYING']
        
        X = []
        y = []
        subjects = []
        
        for subject_id in range(1, 31):  # 30 subjects
            for activity_idx, activity in enumerate(activities, 1):
                n_activity_samples = n_samples // (30 * len(activities))
                
                if activity.startswith('WALKING'):
                    base_pattern = np.random.randn(n_features) * 0.5 + 1.0
                elif activity in ['SITTING', 'STANDING']:
                    base_pattern = np.random.randn(n_features) * 0.1 + 0.5
                else:  # LAYING
                    base_pattern = np.random.randn(n_features) * 0.05
                
                for _ in range(n_activity_samples):
                    noise = np.random.randn(n_features) * 0.1
                    sample = base_pattern + noise
                    X.append(sample)
                    y.append(activity_idx)
                    subjects.append(subject_id)
        
        return np.array(X), np.array(y), np.array(subjects), activities
    
    def extract_time_domain_features(self, window_data):
        """
        Extract time-domain features from sensor data window
        """
        features = {}
        
        # Statistical features
        features['mean'] = np.mean(window_data, axis=0)
        features['std'] = np.std(window_data, axis=0)
        features['var'] = np.var(window_data, axis=0)
        features['max'] = np.max(window_data, axis=0)
        features['min'] = np.min(window_data, axis=0)
        features['range'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(window_data**2, axis=0))
        features['mad'] = np.mean(np.abs(window_data - features['mean']), axis=0)
        features['skewness'] = skew(window_data, axis=0)
        features['kurtosis'] = kurtosis(window_data, axis=0)
        
        # Zero crossing rate
        features['zero_crossing'] = np.sum(np.diff(np.signbit(window_data)), axis=0)
        
        # Signal magnitude area
        features['sma'] = np.sum(np.abs(window_data), axis=0) / window_data.shape[0]
        
        return features
    
    def extract_frequency_domain_features(self, window_data):
        """
        Extract frequency-domain features using FFT
        """
        features = {}
        
        # Apply FFT
        fft_vals = np.fft.fft(window_data, axis=0)
        fft_mag = np.abs(fft_vals[:window_data.shape[0]//2])
        freqs = np.fft.fftfreq(window_data.shape[0], 1/self.sampling_rate)[:window_data.shape[0]//2]
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_mag, axis=0)
        features['dominant_freq'] = freqs[dominant_freq_idx]
        
        # Spectral energy
        features['spectral_energy'] = np.sum(fft_mag**2, axis=0)
        
        # Spectral entropy
        for i in range(fft_mag.shape[1]):
            col_fft = fft_mag[:, i]
            if np.sum(col_fft) > 0:
                prob_dist = col_fft / np.sum(col_fft)
                features[f'spectral_entropy_{i}'] = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
            else:
                features[f'spectral_entropy_{i}'] = 0
        
        return features
    
    def create_windows(self, data, labels, window_size=128, stride=64):
        """
        Create overlapping windows for time-series data
        """
        windows = []
        window_labels = []
        
        for i in range(0, len(data) - window_size, stride):
            window = data[i:i+window_size]
            windows.append(window)
            
            # Use majority vote for window label
            window_label = np.bincount(labels[i:i+window_size]).argmax()
            window_labels.append(window_label)
        
        return np.array(windows), np.array(window_labels)
    
    def apply_filters(self, data, lowcut=0.3, highcut=20.0):
        """
        Apply bandpass filter to sensor data
        """
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        
        return filtered_data
    
    def normalize_data(self, X_train, X_test=None):
        """
        Normalize data using StandardScaler
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def encode_labels(self, y_train, y_test=None):
        """
        Encode categorical labels
        """
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if y_test is not None:
            y_test_encoded = self.label_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        
        return y_train_encoded
    
    def segment_sensor_data(self, acc_data, gyro_data, segment_duration=2.56):
        """
        Segment continuous sensor data into fixed-duration segments
        """
        segment_size = int(segment_duration * self.sampling_rate)
        
        acc_segments = []
        gyro_segments = []
        
        for i in range(0, len(acc_data) - segment_size, segment_size):
            acc_segments.append(acc_data[i:i+segment_size])
            gyro_segments.append(gyro_data[i:i+segment_size])
        
        return np.array(acc_segments), np.array(gyro_segments)
    
    def extract_gravity_acceleration(self, acc_data, alpha=0.8):
        """
        Separate gravity and body acceleration using low-pass filter
        """
        gravity = np.zeros_like(acc_data)
        body_acc = np.zeros_like(acc_data)
        
        # Simple low-pass filter
        gravity[0] = acc_data[0]
        for i in range(1, len(acc_data)):
            gravity[i] = alpha * gravity[i-1] + (1 - alpha) * acc_data[i]
        
        body_acc = acc_data - gravity
        
        return gravity, body_acc
    
    def calculate_jerk_signals(self, data, time_delta=1/50):
        """
        Calculate jerk signals (derivative of acceleration)
        """
        jerk = np.gradient(data, time_delta, axis=0)
        return jerk
    
    def calculate_magnitude(self, data):
        """
        Calculate magnitude of 3D signals
        """
        if data.shape[1] >= 3:
            magnitude = np.sqrt(np.sum(data[:, :3]**2, axis=1))
            return magnitude
        return None

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def create_features(self, windows):
        """
        Create comprehensive feature set from windows
        """
        features_list = []
        
        for window in windows:
            features = {}
            
            # Time domain features
            features.update(self._time_domain_features(window))
            
            # Frequency domain features
            features.update(self._frequency_domain_features(window))
            
            # Statistical features
            features.update(self._statistical_features(window))
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        
        return features_df
    
    def _time_domain_features(self, window):
        """Extract time domain features"""
        features = {}
        
        for i in range(window.shape[1]):
            col_data = window[:, i]
            
            features[f'mean_{i}'] = np.mean(col_data)
            features[f'std_{i}'] = np.std(col_data)
            features[f'max_{i}'] = np.max(col_data)
            features[f'min_{i}'] = np.min(col_data)
            features[f'range_{i}'] = features[f'max_{i}'] - features[f'min_{i}']
            features[f'rms_{i}'] = np.sqrt(np.mean(col_data**2))
            features[f'mad_{i}'] = np.mean(np.abs(col_data - features[f'mean_{i}']))
            
        return features
    
    def _frequency_domain_features(self, window):
        """Extract frequency domain features"""
        features = {}
        
        for i in range(window.shape[1]):
            col_data = window[:, i]
            
            # FFT
            fft_vals = np.fft.fft(col_data)
            fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
            
            if len(fft_mag) > 0:
                features[f'dominant_freq_{i}'] = np.argmax(fft_mag)
                features[f'spectral_energy_{i}'] = np.sum(fft_mag**2)
                
                # Spectral centroid
                if np.sum(fft_mag) > 0:
                    indices = np.arange(len(fft_mag))
                    features[f'spectral_centroid_{i}'] = np.sum(indices * fft_mag) / np.sum(fft_mag)
                else:
                    features[f'spectral_centroid_{i}'] = 0
        
        return features
    
    def _statistical_features(self, window):
        """Extract statistical features"""
        features = {}
        
        for i in range(window.shape[1]):
            col_data = window[:, i]
            
            features[f'skew_{i}'] = skew(col_data)
            features[f'kurtosis_{i}'] = kurtosis(col_data)
            features[f'percentile_25_{i}'] = np.percentile(col_data, 25)
            features[f'percentile_75_{i}'] = np.percentile(col_data, 75)
            features[f'iqr_{i}'] = features[f'percentile_75_{i}'] - features[f'percentile_25_{i}']
            
        return features