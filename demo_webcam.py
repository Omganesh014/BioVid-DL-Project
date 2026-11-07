import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIOVID_ROOT = BASE_DIR if os.path.exists(os.path.join(BASE_DIR, "Part_1_Pain_Recognition")) else os.path.join(BASE_DIR, "BioVid-DL")

sys.path.append(os.path.join(BIOVID_ROOT, "Part_1_Pain_Recognition/models/cnn_lstm"))
sys.path.append(os.path.join(BIOVID_ROOT, "Part_2_Emotion_Recognition/models/3d_cnn"))

from cnn_lstm import VideoCNNLSTM
from r3d import VideoR3D18

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

EMOTION_CLASSES = ['Amusement', 'Anger', 'Disgust', 'Fear', 'Sadness']
EMOTION_COLORS = [
    (0, 200, 100),   # Amusement: Emerald Green
    (0, 50, 230),    # Anger: Vibrant Red
    (0, 140, 255),   # Disgust: Vivid Orange
    (220, 20, 180),  # Fear: Magenta / Purple
    (180, 150, 50)   # Sadness: Muted Slate Cyan
]

PAIN_CLASSES = [
    'BL1: Baseline (No Pain)',
    'PA1: Mild Pain',
    'PA2: Moderate Pain',
    'PA3: Severe Pain',
    'PA4: Pain Tolerance'
]
PAIN_COLORS = [
    (0, 220, 50),    # BL1: Bright Green
    (0, 230, 230),   # PA1: Warm Yellow
    (0, 150, 255),   # PA2: Orange
    (0, 80, 240),    # PA3: Crimson Red
    (20, 0, 200)     # PA4: Deep Blood Red
]

def load_models(device):
    print(f"Loading deep learning checkpoints onto {device}...", flush=True)
    
    # 1. Emotion 3D-CNN (Part E, 5 classes)
    emo_model = VideoR3D18(num_classes=5, pretrained=False)
    emo_path = os.path.join(BIOVID_ROOT, "Part_2_Emotion_Recognition/results/checkpoints/PartE_3D_CNN_R3D18_best.pt")
    if os.path.exists(emo_path):
        ckpt = torch.load(emo_path, map_location=device)
        sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
        emo_model.load_state_dict(sd)
        print("✓ Emotion 3D CNN (R3D-18) checkpoint restored.", flush=True)
    emo_model.to(device).eval()

    # 2. Pain CNN-LSTM (Part 1, 5 classes)
    pain_model = VideoCNNLSTM(num_classes=5, pretrained=False)
    pain_path = os.path.join(BIOVID_ROOT, "Part_1_Pain_Recognition/results/checkpoints/CNN_LSTM_best.pt")
    if os.path.exists(pain_path):
        ckpt = torch.load(pain_path, map_location=device)
        sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
        pain_model.load_state_dict(sd)
        print("✓ Pain CNN-LSTM checkpoint restored.", flush=True)
    pain_model.to(device).eval()

    return emo_model, pain_model

def preprocess_buffer(buffer):
    """Convert deque of 16 BGR frames [224, 224, 3] to normalized PyTorch tensor [1, 16, 3, 224, 224]."""
    frames_np = np.array(buffer) # [16, 224, 224, 3]
    # BGR to RGB
    frames_rgb = frames_np[..., ::-1]
    # Transpose to [16, 3, 224, 224]
    transposed = frames_rgb.transpose(0, 3, 1, 2)
    # Normalize with ImageNet mean and std
    norm = (transposed.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm).unsqueeze(0).float() # [1, 16, 3, 224, 224]
    return tensor

def draw_hud(frame, mode, classes, colors, probs, latency_ms, fps):
    h, w, _ = frame.shape
    top_pred_idx = int(np.argmax(probs))
    top_label = classes[top_pred_idx]
    top_conf = probs[top_pred_idx] * 100.0
    theme_color = colors[top_pred_idx]

    # Semi-transparent overlay for HUD sidebar
    sidebar_w = 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (sidebar_w, h), (18, 20, 24), -1)
    # Top banner overlay
    cv2.rectangle(overlay, (sidebar_w, 0), (w, 80), (18, 20, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Top Banner Text
    mode_title = "AFFECTIVE RECOGNITION (EMOTION)" if mode == "emotion" else "CLINICAL PAIN ASSESSMENT (BIOVID)"
    cv2.putText(frame, mode_title, (sidebar_w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 190, 205), 1, cv2.LINE_AA)
    
    # Large Highlighted Prediction
    pred_str = f"PREDICTED: {top_label.upper()} ({top_conf:.1f}%)"
    cv2.putText(frame, pred_str, (sidebar_w + 20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme_color, 2, cv2.LINE_AA)

    # Sidebar Header
    cv2.putText(frame, "CLASS PROBABILITIES", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, 48), (sidebar_w - 20, 48), (60, 70, 85), 1)

    # Probability Bars
    start_y = 85
    bar_max_w = 190
    for idx, (label, prob) in enumerate(zip(classes, probs)):
        y = start_y + idx * 55
        is_top = (idx == top_pred_idx)
        bar_color = colors[idx] if is_top else (100, 110, 125)
        text_color = (255, 255, 255) if is_top else (180, 190, 205)

        # Label & Percentage
        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1 if not is_top else 2, cv2.LINE_AA)
        cv2.putText(frame, f"{prob*100.0:4.1f}%", (sidebar_w - 70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, bar_color, 1 if not is_top else 2, cv2.LINE_AA)

        # Bar background
        cv2.rectangle(frame, (20, y + 8), (20 + bar_max_w, y + 18), (35, 40, 48), -1)
        # Bar fill
        fill_w = int(bar_max_w * prob)
        if fill_w > 0:
            cv2.rectangle(frame, (20, y + 8), (20 + fill_w, y + 18), colors[idx], -1)

    # Center Face Target Box
    box_size = int(min(h, w - sidebar_w) * 0.62)
    cx = sidebar_w + (w - sidebar_w) // 2
    cy = h // 2 + 25
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2

    corner_len = 28
    corner_col = theme_color
    t = 3
    # Top-Left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), corner_col, t)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), corner_col, t)
    # Top-Right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), corner_col, t)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), corner_col, t)
    # Bottom-Left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), corner_col, t)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), corner_col, t)
    # Bottom-Right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), corner_col, t)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), corner_col, t)

    cv2.putText(frame, "ALIGN FACE INSIDE BOX", (cx - 100, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

    # Footer Telemetry
    foot_y = h - 25
    telemetry = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f} ms | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    cv2.putText(frame, telemetry, (20, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 135, 155), 1, cv2.LINE_AA)

    instructions = "Keys: [M] Toggle Mode (Pain / Emotion)  |  [Q] Exit"
    cv2.putText(frame, instructions, (sidebar_w + 20, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)

    return frame

def run_webcam_demo(camera_idx=0, initial_mode="emotion"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emo_model, pain_model = load_models(device)

    print(f"\nOpening camera device {camera_idx}...", flush=True)
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"\n[ERROR] Unable to open camera at index {camera_idx}.")
        print("Tip: If you are running on a remote headless server or VM, a physical camera cannot open directly.")
        print("Tip: You can pass a video file path instead, e.g.:")
        print("     python demo_webcam.py --video path/to/sample.mp4")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = initial_mode # 'emotion' or 'pain'
    frame_buffer = deque(maxlen=16)
    smoothed_probs = np.ones(5, dtype=np.float32) / 5.0

    print("\n" + "=" * 60)
    print("  BioVid Live Camera Affective & Pain Recognition Active")
    print(f"  Current Mode: {mode.upper()}")
    print("  Press 'm' to toggle between Emotion and Pain modes.")
    print("  Press 'q' or 'ESC' to exit.")
    print("=" * 60 + "\n", flush=True)

    prev_time = time.time()
    latency_ms = 18.0
    fps = 30.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            h, w, _ = frame.shape
            # Center-crop face region to square
            min_dim = min(h, w)
            cx, cy = w // 2, h // 2
            crop = frame[cy - min_dim // 2: cy + min_dim // 2, cx - min_dim // 2: cx + min_dim // 2]
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

            frame_buffer.append(resized)

            if len(frame_buffer) == 16:
                t0 = time.time()
                input_tensor = preprocess_buffer(frame_buffer).to(device)

                with torch.no_grad():
                    if mode == "emotion":
                        logits = emo_model(input_tensor) # [1, 5]
                    else:
                        logits = pain_model(input_tensor) # [1, 5]

                    raw_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                latency_ms = (time.time() - t0) * 1000.0
                # Exponential moving average filter for smooth on-screen display
                smoothed_probs = 0.65 * smoothed_probs + 0.35 * raw_probs

            # FPS calculation
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
            prev_time = curr_time

            # Draw HUD
            curr_classes = EMOTION_CLASSES if mode == "emotion" else PAIN_CLASSES
            curr_colors = EMOTION_COLORS if mode == "emotion" else PAIN_COLORS
            display_frame = draw_hud(frame, mode, curr_classes, curr_colors, smoothed_probs, latency_ms, fps)

            cv2.imshow("BioVid Deep Learning Real-Time Recognition", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC
                break
            elif key == ord('m') or key == ord('M'):
                mode = "pain" if mode == "emotion" else "emotion"
                smoothed_probs = np.ones(5, dtype=np.float32) / 5.0
                print(f"[*] Switched mode to: {mode.upper()}", flush=True)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam demo terminated cleanly.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioVid Live Real-Time Facial Affective & Pain Recognition")
    parser.add_argument("--camera_idx", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--mode", type=str, choices=["emotion", "pain"], default="emotion", help="Initial mode: 'emotion' or 'pain'")
    args = parser.parse_args()

    run_webcam_demo(camera_idx=args.camera_idx, initial_mode=args.mode)
