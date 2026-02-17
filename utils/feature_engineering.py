"""
Advanced feature engineering for HAR system
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
import pywt
from collections import defaultdict

class AdvancedFeatureEngineering:
    def __init__(self, sampling_rate=50):
        self.sampling_rate = sampling_rate
        self.feature_cache = {}
        
    def extract_all_features(self, acc_data, gyro_data):
        """
        Extract comprehensive feature set from accelerometer and gyroscope data
        """
        features = {}
        
        # Basic statistical features
        features.update(self._extract_statistical_features(acc_data, 'acc'))
        features.update(self._extract_statistical_features(gyro_data, 'gyro'))
        
        # Time-domain features
        features.update(self._extract_time_domain_features(acc_data, 'acc'))
        features.update(self._extract_time_domain_features(gyro_data, 'gyro'))
        
        # Frequency-domain features
        features.update(self._extract_frequency_features(acc_data, 'acc'))
        features.update(self._extract_frequency_features(gyro_data, 'gyro'))
        
        # Wavelet features
        features.update(self._extract_wavelet_features(acc_data, 'acc'))
        features.update(self._extract_wavelet_features(gyro_data, 'gyro'))
        
        # Correlation features
        features.update(self._extract_correlation_features(acc_data, gyro_data))
        
        # Entropy features
        features.update(self._extract_entropy_features(acc_data, 'acc'))
        features.update(self._extract_entropy_features(gyro_data, 'gyro'))
        
        return features
    
    def _extract_statistical_features(self, data, prefix):
        """Extract statistical features"""
        features = {}
        
        for i in range(data.shape[1]):
            col_data = data[:, i]
            
            features[f'{prefix}_mean_{i}'] = np.mean(col_data)
            features[f'{prefix}_std_{i}'] = np.std(col_data)
            features[f'{prefix}_var_{i}'] = np.var(col_data)
            features[f'{prefix}_max_{i}'] = np.max(col_data)
            features[f'{prefix}_min_{i}'] = np.min(col_data)
            features[f'{prefix}_range_{i}'] = features[f'{prefix}_max_{i}'] - features[f'{prefix}_min_{i}']
            features[f'{prefix}_median_{i}'] = np.median(col_data)
            
            # Robust statistics
            q75, q25 = np.percentile(col_data, [75, 25])
            features[f'{prefix}_iqr_{i}'] = q75 - q25
            features[f'{prefix}_mad_{i}'] = np.mean(np.abs(col_data - features[f'{prefix}_mean_{i}']))
            
            # Shape statistics
            features[f'{prefix}_skew_{i}'] = self._skewness(col_data)
            features[f'{prefix}_kurtosis_{i}'] = self._kurtosis(col_data)
            
        return features
    
    def _extract_time_domain_features(self, data, prefix):
        """Extract time domain features"""
        features = {}
        
        for i in range(data.shape[1]):
            col_data = data[:, i]
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(col_data)))[0]
            features[f'{prefix}_zero_crossing_{i}'] = len(zero_crossings) / len(col_data)
            
            # Signal magnitude area
            features[f'{prefix}_sma_{i}'] = np.sum(np.abs(col_data)) / len(col_data)
            
            # Root mean square
            features[f'{prefix}_rms_{i}'] = np.sqrt(np.mean(col_data**2))
            
            # Autocorrelation
            autocorr = np.correlate(col_data - features[f'{prefix}_mean_{i}'], 
                                   col_data - features[f'{prefix}_mean_{i}'], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if len(autocorr) > 1:
                features[f'{prefix}_autocorr_peak_{i}'] = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] != 0 else 0
            
        return features
    
    def _extract_frequency_features(self, data, prefix):
        """Extract frequency domain features using FFT"""
        features = {}
        
        for i in range(data.shape[1]):
            col_data = data[:, i]
            
            # Compute FFT
            fft_vals = fft(col_data)
            fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
            freqs = fftfreq(len(col_data), 1/self.sampling_rate)[:len(fft_vals)//2]
            
            if len(fft_mag) > 0:
                # Dominant frequency
                dominant_idx = np.argmax(fft_mag)
                features[f'{prefix}_dominant_freq_{i}'] = freqs[dominant_idx]
                features[f'{prefix}_dominant_mag_{i}'] = fft_mag[dominant_idx]
                
                # Spectral energy
                features[f'{prefix}_spectral_energy_{i}'] = np.sum(fft_mag**2)
                
                # Spectral centroid
                if np.sum(fft_mag) > 0:
                    features[f'{prefix}_spectral_centroid_{i}'] = np.sum(freqs * fft_mag) / np.sum(fft_mag)
                else:
                    features[f'{prefix}_spectral_centroid_{i}'] = 0
                
                # Spectral spread
                if features[f'{prefix}_spectral_centroid_{i}'] > 0:
                    spread = np.sum(((freqs - features[f'{prefix}_spectral_centroid_{i}'])**2) * fft_mag)
                    features[f'{prefix}_spectral_spread_{i}'] = spread / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0
                else:
                    features[f'{prefix}_spectral_spread_{i}'] = 0
                
                # Spectral rolloff (frequency where 85% of energy is contained)
                cumsum_energy = np.cumsum(fft_mag)
                total_energy = cumsum_energy[-1]
                if total_energy > 0:
                    rolloff_idx = np.where(cumsum_energy >= 0.85 * total_energy)[0]
                    if len(rolloff_idx) > 0:
                        features[f'{prefix}_spectral_rolloff_{i}'] = freqs[rolloff_idx[0]]
                    else:
                        features[f'{prefix}_spectral_rolloff_{i}'] = 0
        
        return features
    
    def _extract_wavelet_features(self, data, prefix):
        """Extract wavelet-based features"""
        features = {}
        
        for i in range(data.shape[1]):
            col_data = data[:, i]
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(col_data, 'db4', level=4)
            
            # Extract features from each level
            for level, coeff in enumerate(coeffs):
                features[f'{prefix}_wavelet_level{level}_mean_{i}'] = np.mean(np.abs(coeff))
                features[f'{prefix}_wavelet_level{level}_std_{i}'] = np.std(coeff)
                features[f'{prefix}_wavelet_level{level}_energy_{i}'] = np.sum(coeff**2)
                
                # Energy ratio
                if level > 0:
                    total_energy = np.sum([np.sum(c**2) for c in coeffs])
                    if total_energy > 0:
                        features[f'{prefix}_wavelet_level{level}_energy_ratio_{i}'] = np.sum(coeff**2) / total_energy
        
        return features
    
    def _extract_correlation_features(self, acc_data, gyro_data):
        """Extract correlation features between sensors"""
        features = {}
        
        # Correlation between accelerometer axes
        for i in range(3):
            for j in range(i+1, 3):
                corr = np.corrcoef(acc_data[:, i], acc_data[:, j])[0, 1]
                features[f'acc_corr_{i}_{j}'] = corr if not np.isnan(corr) else 0
        
        # Correlation between gyroscope axes
        for i in range(3):
            for j in range(i+1, 3):
                corr = np.corrcoef(gyro_data[:, i], gyro_data[:, j])[0, 1]
                features[f'gyro_corr_{i}_{j}'] = corr if not np.isnan(corr) else 0
        
        # Cross-correlation between accelerometer and gyroscope
        for i in range(3):
            for j in range(3):
                corr = np.corrcoef(acc_data[:, i], gyro_data[:, j])[0, 1]
                features[f'cross_corr_acc{i}_gyro{j}'] = corr if not np.isnan(corr) else 0
        
        return features
    
    def _extract_entropy_features(self, data, prefix):
        """Extract entropy-based features"""
        features = {}
        
        for i in range(data.shape[1]):
            col_data = data[:, i]
            
            # Approximate entropy
            features[f'{prefix}_approx_entropy_{i}'] = self._approximate_entropy(col_data)
            
            # Sample entropy
            features[f'{prefix}_sample_entropy_{i}'] = self._sample_entropy(col_data)
            
            # Spectral entropy
            features[f'{prefix}_spectral_entropy_{i}'] = self._spectral_entropy(col_data)
        
        return features
    
    def _approximate_entropy(self, data, m=2, r=0.2):
        """Calculate approximate entropy"""
        def _maxdist(x_i, x_j):
            return np.max(np.abs(x_i - x_j))
        
        def _phi(m):
            N = len(data)
            x = np.array([data[i:i+m] for i in range(N-m+1)])
            C = np.zeros(len(x))
            
            for i in range(len(x)):
                for j in range(len(x)):
                    if _maxdist(x[i], x[j]) <= r:
                        C[i] += 1
                C[i] /= (N-m+1)
            
            return np.sum(np.log(C)) / (N-m+1)
        
        if len(data) > m+1:
            return abs(_phi(m) - _phi(m+1))
        return 0
    
    def _sample_entropy(self, data, m=2, r=0.2):
        """Calculate sample entropy"""
        def _maxdist(x_i, x_j):
            return np.max(np.abs(x_i - x_j))
        
        def _phi(m):
            N = len(data)
            x = np.array([data[i:i+m] for i in range(N-m+1)])
            C = 0
            
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j and _maxdist(x[i], x[j]) <= r:
                        C += 1
            
            return C / ((N-m+1)*(N-m))
        
        if len(data) > m+1:
            return -np.log(_phi(m+1) / _phi(m))
        return 0
    
    def _spectral_entropy(self, data):
        """Calculate spectral entropy"""
        # Compute power spectral density
        freqs, psd = signal.periodogram(data, self.sampling_rate)
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # Calculate entropy
        return entropy(psd_norm)
    
    def _skewness(self, data):
        """Calculate skewness"""
        n = len(data)
        if n < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum((data - mean)**3) / ((n-1) * std**3)
    
    def _kurtosis(self, data):
        """Calculate kurtosis"""
        n = len(data)
        if n < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum((data - mean)**4) / ((n-1) * std**4) - 3

class FeatureSelector:
    def __init__(self, n_features=50):
        self.n_features = n_features
        self.selected_features = None
        self.feature_importance = None
    
    def select_features_rf(self, X, y, random_state=42):
        """Select features using Random Forest importance"""
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Select top features
        indices = np.argsort(importance)[::-1][:self.n_features]
        self.selected_features = indices
        self.feature_importance = importance[indices]
        
        return X[:, indices], indices
    
    def select_features_pca(self, X, variance_ratio=0.95):
        """Select features using PCA"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=variance_ratio)
        X_pca = pca.fit_transform(X)
        
        return X_pca, pca
    
    def select_features_correlation(self, X, threshold=0.85):
        """Remove highly correlated features"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find features to keep
        keep_features = []
        for i in range(corr_matrix.shape[0]):
            if i not in keep_features:
                # Find highly correlated features
                correlated = np.where(np.abs(corr_matrix[i]) > threshold)[0]
                keep_features.extend(correlated[correlated > i])
        
        # Get indices of features to keep
        keep_indices = [i for i in range(X.shape[1]) if i not in keep_features]
        self.selected_features = keep_indices
        
        return X[:, keep_indices], keep_indices