import os
import glob
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

SIG_DIR = "BioVid-DL/Part_3_Multimodal_Recognition/results/signal_analysis"
os.makedirs(SIG_DIR, exist_ok=True)

class BioSignalProcessor:
    def __init__(self, fs=512):
        self.fs = fs
        # 1. GSR filter: 4th order lowpass Butterworth at 5 Hz
        b_gsr, a_gsr = signal.butter(4, 5.0 / (fs / 2.0), btype='low')
        self.b_gsr, self.a_gsr = b_gsr, a_gsr

        # 2. ECG filter: 4th order bandpass Butterworth 0.5 - 45 Hz
        b_ecg, a_ecg = signal.butter(4, [0.5 / (fs / 2.0), 45.0 / (fs / 2.0)], btype='bandpass')
        self.b_ecg, self.a_ecg = b_ecg, a_ecg

        # 3. EMG filter: 4th order bandpass Butterworth 20 - 250 Hz + 50Hz notch
        b_emg, a_emg = signal.butter(4, [20.0 / (fs / 2.0), 240.0 / (fs / 2.0)], btype='bandpass')
        self.b_emg, self.a_emg = b_emg, a_emg
        b_notch, a_notch = signal.iirnotch(50.0 / (fs / 2.0), 30.0)
        self.b_notch, self.a_notch = b_notch, a_notch

    def filter_gsr(self, data):
        return signal.filtfilt(self.b_gsr, self.a_gsr, data)

    def filter_ecg(self, data):
        return signal.filtfilt(self.b_ecg, self.a_ecg, data)

    def filter_emg(self, data):
        bp = signal.filtfilt(self.b_emg, self.a_emg, data)
        notched = signal.filtfilt(self.b_notch, self.a_notch, bp)
        # Rectify and compute envelope
        rectified = np.abs(notched)
        return rectified

    def process_trial(self, df_bio):
        """
        Process a 5.5-second biosignal trial dataframe.
        Expected columns: gsr, ecg, emg_trapezius (and optional corrugator, zygomaticus).
        Returns: numpy array of shape (num_channels, time_samples) normalized.
        """
        channels = []
        if 'gsr' in df_bio.columns or 'scl ' in df_bio.columns:
            col = 'gsr' if 'gsr' in df_bio.columns else 'scl '
            gsr_f = self.filter_gsr(df_bio[col].values)
            channels.append(gsr_f)

        if 'ecg' in df_bio.columns or ' ecg ' in df_bio.columns:
            col = 'ecg' if 'ecg' in df_bio.columns else ' ecg '
            ecg_f = self.filter_ecg(df_bio[col].values)
            channels.append(ecg_f)

        if 'emg_trapezius' in df_bio.columns or ' emg_trap' in df_bio.columns:
            col = 'emg_trapezius' if 'emg_trapezius' in df_bio.columns else ' emg_trap'
            emg_f = self.filter_emg(df_bio[col].values)
            channels.append(emg_f)

        # Optional facial EMGs
        for c in ['emg_corrugator', 'emg_zygomaticus']:
            if c in df_bio.columns:
                channels.append(self.filter_emg(df_bio[c].values))

        sig_arr = np.stack(channels, axis=0) # [C, L]
        # Standardize per channel
        mean = np.mean(sig_arr, axis=-1, keepdims=True)
        std = np.std(sig_arr, axis=-1, keepdims=True) + 1e-6
        return (sig_arr - mean) / std

def generate_signal_plots():
    print("Generating publication-quality physiological waveform figures...")
    sample_file = "/run/media/acernitro/Ventoy/DB/PartA/biosignals_raw/071309_w_21/071309_w_21-PA4-001_bio.csv"
    if not os.path.exists(sample_file):
        sample_file = glob.glob("/run/media/acernitro/Ventoy/DB/PartA/biosignals_raw/*/*-PA4-*.csv")[0]

    df = pd.read_csv(sample_file, sep=r'\s*;\s*|\t|,', engine='python')
    proc = BioSignalProcessor(fs=512)
    time_s = np.arange(len(df)) / 512.0

    # 1. Raw vs Filtered GSR Waveform
    raw_gsr = df['gsr'].values
    filt_gsr = proc.filter_gsr(raw_gsr)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s, raw_gsr, label="Raw GSR (Electrodermal)", color="gray", alpha=0.6, linewidth=1)
    ax.plot(time_s, filt_gsr, label="Filtered GSR (Butterworth Lowpass 5Hz)", color="#1f77b4", linewidth=2)
    ax.set_title("Electrodermal Activity (GSR): Raw vs Low-Pass Filtered Waveform (PA4 Trial)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Amplitude (µS / a.u.)", fontsize=11)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(SIG_DIR, "gsr_raw_vs_filtered.png"), dpi=300)
    plt.close(fig)

    # 2. Filtered ECG Waveform
    raw_ecg = df['ecg'].values
    filt_ecg = proc.filter_ecg(raw_ecg)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s[:1024], filt_ecg[:1024], color="#d62728", linewidth=1.5)
    ax.set_title("Electrocardiogram (ECG): Bandpass Filtered (0.5–45 Hz) Showing QRS Complexes", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("ECG Voltage (mV)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(SIG_DIR, "ecg_filtered_waveform.png"), dpi=300)
    plt.close(fig)

    # 3. Filtered EMG Waveform
    raw_emg = df['emg_trapezius'].values
    filt_emg = proc.filter_emg(raw_emg)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s, filt_emg, color="#2ca02c", linewidth=1.2)
    ax.set_title("Electromyography (EMG Trapezius): Bandpass (20–250 Hz) + 50Hz Notch + Rectified", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Rectified EMG Amplitude (µV)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(SIG_DIR, "emg_trapezius_waveform.png"), dpi=300)
    plt.close(fig)

    # 4. Multi-channel Comparison Across Pain Classes
    fig, axes = plt.subplots(5, 3, figsize=(14, 12), sharex=True)
    subj_dir = "/run/media/acernitro/Ventoy/DB/PartA/biosignals_raw/071309_w_21"
    classes = [("BL1", "Baseline (No Pain)"), ("PA1", "Level 1"), ("PA2", "Level 2"), ("PA3", "Level 3"), ("PA4", "Level 4 (Tolerance)")]

    for row_idx, (cname, cdesc) in enumerate(classes):
        c_files = glob.glob(f"{subj_dir}/*-{cname}-*_bio.csv")
        if c_files:
            c_df = pd.read_csv(c_files[0], sep=r'\s*;\s*|\t|,', engine='python')
            t_ax = np.arange(len(c_df)) / 512.0
            g_f = proc.filter_gsr(c_df['gsr'].values)
            e_f = proc.filter_ecg(c_df['ecg'].values)
            m_f = proc.filter_emg(c_df['emg_trapezius'].values)

            axes[row_idx, 0].plot(t_ax, g_f, color="#1f77b4")
            axes[row_idx, 0].set_ylabel(f"{cname}\n({cdesc})", fontsize=10, fontweight="bold")
            axes[row_idx, 1].plot(t_ax[:1024], e_f[:1024], color="#d62728")
            axes[row_idx, 2].plot(t_ax, m_f, color="#2ca02c")

    axes[0, 0].set_title("GSR (Electrodermal)", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("ECG (Cardiac, 2-sec zoom)", fontsize=12, fontweight="bold")
    axes[0, 2].set_title("EMG Trapezius (Muscular)", fontsize=12, fontweight="bold")
    for j in range(3):
        axes[-1, j].set_xlabel("Time (seconds)", fontsize=11)

    plt.suptitle("Physiological Waveform Comparisons Across Increasing Pain Intensities (BL1 → PA4)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(SIG_DIR, "signals_across_pain_classes.png"), dpi=300)
    plt.close(fig)
    print("Signal waveform figures successfully created and saved in results/signal_analysis.")

if __name__ == "__main__":
    generate_signal_plots()
