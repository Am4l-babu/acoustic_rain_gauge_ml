import pandas as pd
import numpy as np
import librosa
import pywt  # pip install PyWavelets
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = Path(r"D:\arg_cleaned_dataset")
OUTPUT_DIR = Path(r"D:\advanced_features")
OUTPUT_DIR.mkdir(exist_ok=True)

# Processing settings
SAMPLE_MODE = True              # Set to False to process ALL files (takes hours)
SAMPLES_PER_FOLDER = 2000       # If SAMPLE_MODE=True, process this many per folder
PACKET_DURATION = 0.5           # Packet size for histogram analysis (seconds)
NUM_HIST_BINS = 10              # Number of histogram bins
SAMPLE_RATE = 16000             # Audio sample rate

# ==========================================
# FEATURE EXTRACTION FUNCTIONS
# ==========================================

def extract_histogram_packet_features(y, sr, packet_duration=0.5, num_bins=10):
    """
    HISTOGRAM PACKET ANALYSIS
    Captures temporal variability patterns - rain has spiky, irregular packets
    while wind/noise has steady, consistent packets.
    """
    samples_per_packet = int(packet_duration * sr)
    total_packets = len(y) // samples_per_packet
    
    if total_packets < 3:
        return {}
    
    # Split into packets and compute RMS and ZCR for each
    rms_packets = []
    zcr_packets = []
    
    for i in range(total_packets):
        start = i * samples_per_packet
        end = start + samples_per_packet
        packet = y[start:end]
        
        rms_packets.append(np.sqrt(np.mean(np.square(packet))))
        zcr_packets.append(np.mean(librosa.feature.zero_crossing_rate(packet)))
    
    rms_packets = np.array(rms_packets)
    zcr_packets = np.array(zcr_packets)
    
    features = {}
    
    # RMS histogram (10 bins)
    rms_hist, _ = np.histogram(rms_packets, bins=num_bins, density=True)
    for i, val in enumerate(rms_hist):
        features[f'rms_hist_bin_{i}'] = val
    
    # ZCR histogram (10 bins)
    zcr_hist, _ = np.histogram(zcr_packets, bins=num_bins, density=True)
    for i, val in enumerate(zcr_hist):
        features[f'zcr_hist_bin_{i}'] = val
    
    # Statistical features of packet distributions
    features['rms_packet_mean'] = np.mean(rms_packets)
    features['rms_packet_std'] = np.std(rms_packets)
    features['rms_packet_skew'] = float(pd.Series(rms_packets).skew()) if len(rms_packets) > 2 else 0
    features['rms_packet_kurtosis'] = float(pd.Series(rms_packets).kurtosis()) if len(rms_packets) > 3 else 0
    
    features['zcr_packet_mean'] = np.mean(zcr_packets)
    features['zcr_packet_std'] = np.std(zcr_packets)
    features['zcr_packet_skew'] = float(pd.Series(zcr_packets).skew()) if len(zcr_packets) > 2 else 0
    
    return features


def extract_wavelet_features(y, sr, wavelet='db4', level=5):
    """
    WAVELET TRANSFORM ANALYSIS
    Decomposes signal into multi-scale components. Different raindrop sizes
    create energy at different scales (large drops = low freq, small drops = high freq).
    """
    try:
        coeffs = pywt.wavedec(y, wavelet, level=level)
        
        features = {}
        
        # Energy at each decomposition level
        for i, c in enumerate(coeffs):
            energy = np.sum(c**2)
            features[f'wavelet_level_{i}_energy'] = energy
        
        # Total energy
        total_energy = sum(np.sum(c**2) for c in coeffs)
        features['wavelet_total_energy'] = total_energy
        
        # Energy ratios (normalized)
        if total_energy > 0:
            for i, c in enumerate(coeffs):
                features[f'wavelet_level_{i}_ratio'] = np.sum(c**2) / total_energy
        
        # Wavelet entropy (signal complexity at each scale)
        for i, c in enumerate(coeffs):
            p = c**2 / (np.sum(c**2) + 1e-10)
            features[f'wavelet_level_{i}_entropy'] = -np.sum(p * np.log(p + 1e-10))
        
        return features
        
    except Exception as e:
        return {}


def extract_spectral_flux_features(y, sr):
    """
    SPECTRAL FLUX
    Measures how rapidly the frequency spectrum changes over time.
    Rain creates rapid spectral changes; wind/noise is more stable.
    """
    # Compute spectrogram
    hop_length = 512
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    
    # Spectral flux: frame-to-frame spectral difference
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    
    features = {
        'spectral_flux_mean': np.mean(flux),
        'spectral_flux_std': np.std(flux),
        'spectral_flux_max': np.max(flux),
        'spectral_flux_median': np.median(flux),
        'spectral_flux_skew': float(pd.Series(flux).skew()) if len(flux) > 2 else 0,
    }
    
    return features


def extract_teager_energy_features(y, sr):
    """
    TEAGER ENERGY OPERATOR (TEO)
    Tracks instantaneous energy of the signal. Extremely sensitive to
    transient impacts like raindrop strikes.
    Formula: Ψ(x[n]) = x[n]² - x[n-1] * x[n+1]
    """
    if len(y) < 3:
        return {}
    
    # Compute TEO
    teo = y[1:-1]**2 - y[:-2] * y[2:]
    
    features = {
        'teo_mean': np.mean(teo),
        'teo_std': np.std(teo),
        'teo_max': np.max(teo),
        'teo_median': np.median(teo),
        'teo_peak_to_mean': np.max(teo) / (np.mean(teo) + 1e-10),
        'teo_energy_ratio': np.sum(teo > np.mean(teo)) / len(teo),  # % of high-energy samples
    }
    
    return features


def extract_entropy_features(y, sr):
    """
    ENTROPY FEATURES
    Measures signal complexity and randomness. Rain is more complex/random
    than steady wind or silence.
    """
    features = {}
    
    # 1. Spectral Entropy
    S = np.abs(librosa.stft(y))
    S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
    spectral_entropy_per_frame = scipy_entropy(S_norm + 1e-10, axis=0)
    features['spectral_entropy_mean'] = np.mean(spectral_entropy_per_frame)
    features['spectral_entropy_std'] = np.std(spectral_entropy_per_frame)
    
    # 2. Shannon Entropy of amplitude distribution
    hist, _ = np.histogram(np.abs(y), bins=50, density=True)
    hist = hist / (hist.sum() + 1e-10)
    features['shannon_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
    
    # 3. Sample Entropy (simplified approximation)
    # Measures signal irregularity
    diff_signal = np.diff(y)
    features['sample_entropy_approx'] = np.std(diff_signal) / (np.std(y) + 1e-10)
    
    return features


def extract_all_advanced_features(audio_path):
    """
    Master function that extracts ALL advanced features from a single audio file.
    Returns a flat dictionary of features.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=None)
        
        if len(y) < SAMPLE_RATE:  # Less than 1 second
            return None
        
        # Extract all feature categories
        features = {}
        features.update(extract_histogram_packet_features(y, sr, PACKET_DURATION, NUM_HIST_BINS))
        features.update(extract_wavelet_features(y, sr))
        features.update(extract_spectral_flux_features(y, sr))
        features.update(extract_teager_energy_features(y, sr))
        features.update(extract_entropy_features(y, sr))
        
        # Add basic metadata
        features['audio_duration'] = len(y) / sr
        features['sample_rate'] = sr
        
        return features
        
    except Exception as e:
        return None


# ==========================================
# MAIN PROCESSING PIPELINE
# ==========================================

def process_dataset():
    """Process all (or sampled) audio files and extract advanced features"""
    
    print("="*70)
    print("ADVANCED FEATURE EXTRACTION PIPELINE")
    print("="*70)
    print(f"Mode: {'SAMPLE (testing)' if SAMPLE_MODE else 'FULL PROCESSING'}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    # Find all CSV files
    all_csvs = list(DATA_DIR.glob("**/cleaned_aligned_data.csv"))
    print(f"\n📁 Found {len(all_csvs)} data folders")
    
    all_records = []
    total_processed = 0
    total_errors = 0
    
    for csv_file in tqdm(all_csvs, desc="Processing folders"):
        try:
            df = pd.read_csv(csv_file)
            
            # Sample if in sample mode
            if SAMPLE_MODE:
                # Stratified sample: keep ratio of rainy/dry
                rainy = df[df['rainfall_mm'] > 0]
                dry = df[df['rainfall_mm'] == 0]
                
                rainy_sample = rainy.sample(
                    min(SAMPLES_PER_FOLDER // 2, len(rainy)), 
                    random_state=42
                )
                dry_sample = dry.sample(
                    min(SAMPLES_PER_FOLDER // 2, len(dry)), 
                    random_state=42
                )
                df = pd.concat([rainy_sample, dry_sample])
            
            # Process each audio file
            for idx, row in df.iterrows():
                audio_path = Path(row['audio_full_path'])
                
                if not audio_path.exists():
                    total_errors += 1
                    continue
                
                # Extract features
                features = extract_all_advanced_features(audio_path)
                
                if features is None:
                    total_errors += 1
                    continue
                
                # Add metadata
                features['timestamp'] = row['timestamp']
                features['rainfall_mm'] = row['rainfall_mm']
                features['is_rainy'] = 1 if row['rainfall_mm'] > 0 else 0
                features['source_folder'] = csv_file.parent.name
                features['audio_filename'] = row['audio_filename']
                
                all_records.append(features)
                total_processed += 1
        
        except Exception as e:
            print(f"\n❌ Error processing {csv_file}: {e}")
            continue
    
    # Create DataFrame
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {total_processed} files")
    print(f"❌ Errors: {total_errors}")
    print(f"📊 Total features extracted: {len(all_records[0]) - 5 if all_records else 0}")
    
    # Save to CSV
    advanced_df = pd.DataFrame(all_records)
    output_path = OUTPUT_DIR / 'advanced_features_dataset.csv'
    advanced_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    # Save feature list
    feature_cols = [c for c in advanced_df.columns if c not in 
                   ['timestamp', 'rainfall_mm', 'is_rainy', 'source_folder', 'audio_filename']]
    
    with open(OUTPUT_DIR / 'feature_list.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print(f"📝 Feature list saved: {OUTPUT_DIR / 'feature_list.txt'}")
    
    # Print summary statistics
    print(f"\n📊 DATASET SUMMARY:")
    print(f"  Total records: {len(advanced_df)}")
    print(f"  Rainy samples: {(advanced_df['is_rainy'] == 1).sum()}")
    print(f"  Dry samples: {(advanced_df['is_rainy'] == 0).sum()}")
    print(f"  Total features: {len(feature_cols)}")
    
    # Print feature categories
    print(f"\n🎯 FEATURE CATEGORIES:")
    hist_features = [c for c in feature_cols if 'hist' in c or 'packet' in c]
    wavelet_features = [c for c in feature_cols if 'wavelet' in c]
    flux_features = [c for c in feature_cols if 'flux' in c]
    teo_features = [c for c in feature_cols if 'teo' in c]
    entropy_features = [c for c in feature_cols if 'entropy' in c]
    
    print(f"  Histogram Packet: {len(hist_features)} features")
    print(f"  Wavelet: {len(wavelet_features)} features")
    print(f"  Spectral Flux: {len(flux_features)} features")
    print(f"  Teager Energy: {len(teo_features)} features")
    print(f"  Entropy: {len(entropy_features)} features")
    
    return advanced_df


# ==========================================
# RUN IT
# ==========================================
if __name__ == "__main__":
    advanced_df = process_dataset()
    
    print("\n" + "="*70)
    print("🎉 EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Check the output CSV: {OUTPUT_DIR / 'advanced_features_dataset.csv'}")
    print(f"2. Review feature_list.txt to see all extracted features")
    print(f"3. Merge with original features and retrain XGBoost model")
    print(f"4. Run feature selection to find the most important ones")