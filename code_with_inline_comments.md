# -*- coding: utf-8 -*-
"""
Parkinson’s Disease Detection from Voice Recordings
Feature Extraction + eGeMAPS + Save to CSV

বাংলা ব্যাখ্যা:
এই স্ক্রিপ্টে আপনি প্রতিটা .wav অডিও ফাইল থেকে অনেক ধরনের acoustic feature বের করছেন
(যেমন MFCC, pitch, jitter, shimmer, HNR, formants, spectral features, rhythm ইত্যাদি)
এরপর OpenSMILE থেকে eGeMAPS feature বের করে সব একসাথে করে CSV ফাইলে সেভ করছেন।
"""

import sys

# -------------------- (A) Colab only: library install --------------------
# কেন লাগে:
# Colab এ praat-parselmouth এবং opensmile default এ থাকে না, তাই install করতে হয়।
if 'google.colab' in sys.modules:
    # !pip install --upgrade pip
    # !pip install praat-parselmouth opensmile
    pass

# -------------------- (B) Mount Google Drive --------------------
# কেন লাগে:
# আপনার dataset এবং output drive এ আছে, তাই drive mount দরকার।
# Colab না হলে এই অংশটি remove করবেন।
# from google.colab import drive
# drive.mount('/content/drive')

# -------------------- (C) Imports --------------------
import os
import numpy as np
import pandas as pd
import librosa                 # MFCC, spectral features, chroma, mel, pyin etc.
import parselmouth             # Praat-based features: jitter, shimmer, formants, pitch, HNR
import opensmile               # eGeMAPS features
from scipy.stats import skew, kurtosis
from parselmouth.praat import call
import warnings
warnings.filterwarnings("ignore")

# -------------------- (D) Project paths --------------------
# কেন লাগে:
# ফাইল পাথ এক জায়গায় define করলে পরে maintain করা সহজ হয়।
PROJECT_DIR = "/content/drive/MyDrive/Colab Notebooks/Thesis/Parkinson’s Disease Detection from Voice Recordings"
DATA_ROOT = os.path.join(PROJECT_DIR, "raw_data", "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "parkinsons_audio_features_combined.csv")


# ============================================================
# (1) Helper Function: summarize_feature
# ============================================================
def summarize_feature(arr, prefix):
    """
    English:
    Convert a time-series feature array into fixed-length statistics.
    Returns mean/std/min/max/range/skew/kurtosis for that feature.

    বাংলা কেন:
    অডিও ফিচারগুলো (MFCC, energy, pitch ইত্যাদি) সাধারণত ফ্রেম অনুযায়ী পরিবর্তন হয়।
    কিন্তু ML মডেলকে fixed-length feature vector দিতে হয়।
    তাই আমরা time-series → summary stats (mean, std...) বানাই।
    """
    stats = {}
    arr = np.asarray(arr)

    # কেন:
    # NaN/Inf থাকলে mean/std নষ্ট হয়ে যাবে, তাই বাদ দিচ্ছি।
    arr = arr[np.isfinite(arr)]

    # কেন:
    # কিছু ফিচারে negative/0 মান আসতে পারে। আপনি positive অংশ প্রাধান্য দিচ্ছেন।
    # (যদিও সব feature-এর জন্য এই rule সবসময় perfect না—কিন্তু আপনি safe হিসেবে করেছেন)
    arr = arr[arr > 0] if np.any(arr > 0) else arr

    # mean/std/min/max = basic statistics
    stats[f"{prefix}_mean"] = np.mean(arr) if arr.size else 0
    stats[f"{prefix}_std"] = np.std(arr) if arr.size else 0
    stats[f"{prefix}_min"] = np.min(arr) if arr.size else 0
    stats[f"{prefix}_max"] = np.max(arr) if arr.size else 0

    # range = max-min, spread বোঝায়
    stats[f"{prefix}_range"] = stats[f"{prefix}_max"] - stats[f"{prefix}_min"]

    # skew/kurtosis = distribution shape বোঝায় (optional কিন্তু useful)
    stats[f"{prefix}_skew"] = skew(arr) if arr.size > 2 else 0
    stats[f"{prefix}_kurtosis"] = kurtosis(arr) if arr.size > 3 else 0
    return stats


# ============================================================
# (2) Rhythm Features
# ============================================================
def calculate_rhythm_features(y, sr):
    """
    English:
    Compute simple rhythm features using RMS thresholding:
    - VarcoV: variability of voiced-segment durations
    - PercentV: proportion of voiced time
    - nPVI: normalized Pairwise Variability Index for durations

    বাংলা কেন:
    Parkinson’s রোগীদের speech timing/ rhythm অনেক সময় irregular হয়।
    তাই voiced segment duration ধরে এই ধরনের rhythm features helpful হতে পারে।
    """
    features = {}

    # frame_length/hop_length কেন:
    # অডিওকে ছোট ছোট frame এ ভাগ করে RMS হিসাব করা হয়
    frame_length, hop_length = 2048, 512

    # RMS_THR কেন:
    # RMS energy বেশি হলে ধরে নিচ্ছি সেখানে speech/voiced আছে
    RMS_THR, MIN_DUR = 0.25, 0.05

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # normalize কেন:
    # thresholding সহজ করতে RMS কে 0-1 স্কেলে আনা
    rms = rms / np.max(rms) if np.max(rms) > 1e-6 else np.zeros_like(rms)

    voiced = rms > RMS_THR
    frame_dur = hop_length / sr

    # voiced segment duration বের করা
    durations, start, active = [], 0, False
    for i in range(len(voiced)):
        if voiced[i] and not active:
            start, active = i, True
        if (not voiced[i] or i == len(voiced) - 1) and active:
            dur = (i - start) * frame_dur
            if dur >= MIN_DUR:
                durations.append(dur)
            active = False

    V = np.array(durations)
    total_time = len(y) / sr

    # VarcoV: std/mean (irregularity measure)
    features["VarcoV"] = np.std(V) / np.mean(V) if V.size and np.mean(V) != 0 else 0

    # PercentV: মোট সময়ের কত অংশ voiced
    features["PercentV"] = np.sum(V) / total_time if V.size else 0

    # nPVI: consecutive durations কতটা পরিবর্তিত হচ্ছে
    features["nPVI"] = (
        (np.sum(np.abs(np.diff(V))) / np.sum(V[:-1] + V[1:])) *
        (len(V) / (len(V) - 1)) * 100
        if V.size > 1 and np.sum(V[:-1] + V[1:]) != 0 else 0
    )
    return features


# ============================================================
# (3) Main Feature Extraction
# ============================================================
def extract_all_features(audio_path):
    """
    English:
    Extract a large set of custom acoustic features from one audio file.

    বাংলা:
    ১টা wav ফাইল থেকে আপনার custom features (MFCC, jitter, shimmer, HNR, formants, spectral, rhythm)
    সবকিছু বের করে dictionary হিসেবে return করবে।
    """
    # librosa.load কেন:
    # waveform (y) এবং sampling rate (sr) দরকার MFCC/spectral feature বের করতে
    y, sr = librosa.load(audio_path, sr=None)

    results = {}

    # Praat/parselmouth কেন:
    # jitter/shimmer/formants/pitch/HNR বের করতে praat algorithms ব্যবহার হয়
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    harmonicity = snd.to_harmonicity()

    # PointProcess কেন:
    # jitter/shimmer এর জন্য periodic point process (glottal pulses) লাগে
    point = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    # ----------------- JITTER -----------------
    # jp tuple মানে praat function এর parameters
    jp = (0, 0, 0.0001, 0.02, 1.3)
    results["Jitter_local"] = call(point, "Get jitter (local)", *jp)
    results["Jitter_local_abs"] = call(point, "Get jitter (local, absolute)", *jp)
    results["Jitter_rap"] = call(point, "Get jitter (rap)", *jp)
    results["Jitter_ppq5"] = call(point, "Get jitter (ppq5)", *jp)
    results["Jitter_ddp"] = call(point, "Get jitter (ddp)", *jp)

    # ----------------- SHIMMER -----------------
    sp = (0, 0, 0.0001, 0.02, 1.3, 1.6)
    results["Shimmer_local"] = call([snd, point], "Get shimmer (local)", *sp)
    results["Shimmer_local_dB"] = call([snd, point], "Get shimmer (local_dB)", *sp)
    results["Shimmer_apq3"] = call([snd, point], "Get shimmer (apq3)", *sp)
    results["Shimmer_apq5"] = call([snd, point], "Get shimmer (apq5)", *sp)
    results["Shimmer_apq11"] = call([snd, point], "Get shimmer (apq11)", *sp)
    results["Shimmer_dda"] = call([snd, point], "Get shimmer (dda)", *sp)

    # ----------------- HNR -----------------
    # harmonicity.values থেকে অস্বাভাবিক মান বাদ দিয়ে summary
    h_vals = harmonicity.values[harmonicity.values > -200]
    results.update(summarize_feature(h_vals, "HNR"))

    # ----------------- F0 (Pitch) -----------------
    # pitch.selected_array['frequency'] -> frame-wise pitch
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  # unvoiced frames বাদ
    results.update(summarize_feature(f0, "F0"))

    # slope কেন:
    # pitch trend ধরতে (সময় অনুযায়ী pitch উঠছে/নামছে?)
    results["F0_slope"] = np.polyfit(np.arange(len(f0)), f0, 1)[0] if len(f0) > 1 else 0

    # ----------------- FORMANTS (F1-F3) -----------------
    formant = snd.to_formant_burg()
    duration = snd.get_total_duration()

    # t_vals কেন:
    # অডিও duration জুড়ে অনেক timepoint এ formant sample নিচ্ছি
    t_vals = np.linspace(0, duration, max(int(duration * 100), 1))

    for i in range(1, 4):
        vals = np.array([formant.get_value_at_time(i, t) for t in t_vals])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        results.update(summarize_feature(vals, f"F{i}"))

    # ----------------- MFCC + DELTAS -----------------
    # MFCC speech analysis এ সবচেয়ে popular
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Delta/Delta2/Delta3 কেন:
    # speech dynamics ধরতে (feature পরিবর্তনের হার)
    for name, arr in [
        ("MFCC", mfcc),
        ("DeltaMFCC", librosa.feature.delta(mfcc)),
        ("Delta2MFCC", librosa.feature.delta(mfcc, order=2)),
        ("Delta3MFCC", librosa.feature.delta(mfcc, order=3)),
    ]:
        for i in range(13):
            results.update(summarize_feature(arr[i], f"{name}{i+1}"))

    # ----------------- SPECTRAL FEATURES -----------------
    spectral = {
        "ZCR": librosa.feature.zero_crossing_rate(y)[0],
        "Energy": librosa.feature.rms(y=y)[0],
        "Centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
        "Rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
        "Bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
        "Flatness": librosa.feature.spectral_flatness(y=y)[0],
    }

    # কেন delta/delta2:
    # spectral feature এর temporal change ধরতে
    for k, v in spectral.items():
        results.update(summarize_feature(v, k))
        results.update(summarize_feature(librosa.feature.delta(v), f"Delta{k}"))
        results.update(summarize_feature(librosa.feature.delta(v, order=2), f"Delta2{k}"))

    # ----------------- SPECTRAL CONTRAST -----------------
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        results.update(summarize_feature(contrast[i], f"SpecContrast{i+1}"))

    # ----------------- CHROMA -----------------
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        results.update(summarize_feature(chroma[i], f"Chroma{i+1}"))

    # ----------------- MEL BANDS -----------------
    # Mel bands = perceptual frequency bands
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=25)
    for i, val in enumerate(np.mean(mel, axis=1)):
        results[f"MelBand_{i+1}"] = float(val)

    # ----------------- PITCH CONTOUR (pyin) -----------------
    # pyin কেন:
    # pitch contour বের করার robust method
    f0c, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    f0c = f0c[~np.isnan(f0c)]
    results.update(summarize_feature(f0c, "PitchContour"))
    results["PitchContour_slope"] = np.polyfit(np.arange(len(f0c)), f0c, 1)[0] if len(f0c) > 1 else 0

    # ----------------- RHYTHM -----------------
    results.update(calculate_rhythm_features(y, sr))

    return results


# ============================================================
# (4) eGeMAPS feature extractor (OpenSMILE)
# ============================================================
# কেন:
# eGeMAPS হলো literature-friendly standard feature set, thesis এ লিখলে ভালো লাগে
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv01a,
    feature_level=opensmile.FeatureLevel.Functionals
)

def extract_egemaps_features(audio_path):
    """
    English: Extract eGeMAPS functionals from one audio file.
    বাংলা: OpenSMILE দিয়ে স্ট্যান্ডার্ড eGeMAPS features dict আকারে return করে।
    """
    return smile.process_file(audio_path).iloc[0].to_dict()


# ============================================================
# (5) Batch processing: Loop over dataset and save one CSV
# ============================================================
languages = ["Italian", "english"]  # আপনার folder নাম অনুযায়ী ঠিক করবেন
folders = ["HC", "PD"]              # HC=healthy, PD=parkinson
label_map = {"HC": 0, "PD": 1}

all_rows = []

for language in languages:
    for folder in folders:
        data_dir = os.path.join(DATA_ROOT, language, folder)

        print(f"\n--- Processing: {language}/{folder} ---")

        # কেন:
        # directory না থাকলে crash না করে skip
        if not os.path.exists(data_dir):
            print(f"Skipping {language}/{folder}: Directory not found at {data_dir}")
            continue

        for fname in sorted(os.listdir(data_dir)):
            # কেন:
            # শুধুমাত্র .wav প্রসেস করবো
            if not fname.lower().endswith(".wav"):
                continue

            file_path = os.path.join(data_dir, fname)
            print(f"Processing: {fname}")

            try:
                # custom features (~many)
                custom = extract_all_features(file_path)

                # eGeMAPS features (standard set)
                egemaps = extract_egemaps_features(file_path)

                # merge features
                features = {**custom, **egemaps}

                # Patient_ID কেন:
                # speaker-wise split করার জন্য same patient কে group করতে হবে
                base_name = os.path.splitext(fname)[0]
                patient_id = base_name.split("-")[0]

                # metadata যোগ
                features["Patient_ID"] = patient_id
                features["Class"] = label_map[folder]   # 0/1 label
                features["File"] = fname
                features["Folder"] = folder
                features["Language"] = language

                all_rows.append(features)

                print(f"✔ Features extracted: {len(features)}")

            except Exception as e:
                print(f"❌ Failed on {fname}: {e}")

# Save final CSV
if len(all_rows) > 0:
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ DONE — Combined CSV saved to:\n{OUTPUT_CSV}")
    print(f"Shape: {final_df.shape}")
else:
    print("\n❌ No data was processed. CSV not created.")
