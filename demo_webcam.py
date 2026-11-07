import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Set Qt environment defaults for Wayland / X11 compatibility
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

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

# Shared global frame for browser streaming
GLOBAL_STREAM_FRAME = None
FRAME_LOCK = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global GLOBAL_STREAM_FRAME, FRAME_LOCK
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>BioVid Live Deep Learning Recognition</title>
                <style>
                    body { background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; text-align: center; margin: 0; padding: 20px; }
                    h1 { color: #38bdf8; margin-bottom: 6px; }
                    p { color: #94a3b8; font-size: 14px; margin-top: 0; }
                    .stream-box { display: inline-block; border: 2px solid #334155; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
                    img { display: block; max-width: 90vw; height: auto; }
                </style>
            </head>
            <body>
                <h1>BioVid Live Facial Affective & Pain Recognition</h1>
                <p>Real-Time Deep Neural Network Inference on NVIDIA GPU</p>
                <div class="stream-box">
                    <img src="/stream.mjpg" />
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with FRAME_LOCK:
                        if GLOBAL_STREAM_FRAME is None:
                            time.sleep(0.03)
                            continue
                        ret, jpeg = cv2.imencode('.jpg', GLOBAL_STREAM_FRAME, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        continue
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception:
                pass
        else:
            self.send_error(404)

def start_web_server(port=5000):
    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"\n[✓] Browser Live Stream active at: http://localhost:{port} (or http://127.0.0.1:{port})", flush=True)

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
    frames_rgb = frames_np[..., ::-1]
    transposed = frames_rgb.transpose(0, 3, 1, 2)
    norm = (transposed.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm).unsqueeze(0).float()
    return tensor

def draw_hud(frame, mode, classes, colors, probs, latency_ms, fps):
    h, w, _ = frame.shape
    top_pred_idx = int(np.argmax(probs))
    top_label = classes[top_pred_idx]
    top_conf = probs[top_pred_idx] * 100.0
    theme_color = colors[top_pred_idx]

    sidebar_w = 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (sidebar_w, h), (18, 20, 24), -1)
    cv2.rectangle(overlay, (sidebar_w, 0), (w, 80), (18, 20, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    mode_title = "AFFECTIVE RECOGNITION (EMOTION)" if mode == "emotion" else "CLINICAL PAIN ASSESSMENT (BIOVID)"
    cv2.putText(frame, mode_title, (sidebar_w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 190, 205), 1, cv2.LINE_AA)
    
    pred_str = f"PREDICTED: {top_label.upper()} ({top_conf:.1f}%)"
    cv2.putText(frame, pred_str, (sidebar_w + 20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme_color, 2, cv2.LINE_AA)

    cv2.putText(frame, "CLASS PROBABILITIES", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, 48), (sidebar_w - 20, 48), (60, 70, 85), 1)

    start_y = 85
    bar_max_w = 190
    for idx, (label, prob) in enumerate(zip(classes, probs)):
        y = start_y + idx * 55
        is_top = (idx == top_pred_idx)
        bar_color = colors[idx] if is_top else (100, 110, 125)
        text_color = (255, 255, 255) if is_top else (180, 190, 205)

        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1 if not is_top else 2, cv2.LINE_AA)
        cv2.putText(frame, f"{prob*100.0:4.1f}%", (sidebar_w - 70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, bar_color, 1 if not is_top else 2, cv2.LINE_AA)

        cv2.rectangle(frame, (20, y + 8), (20 + bar_max_w, y + 18), (35, 40, 48), -1)
        fill_w = int(bar_max_w * prob)
        if fill_w > 0:
            cv2.rectangle(frame, (20, y + 8), (20 + fill_w, y + 18), colors[idx], -1)

    box_size = int(min(h, w - sidebar_w) * 0.62)
    cx = sidebar_w + (w - sidebar_w) // 2
    cy = h // 2 + 25
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2

    corner_len = 28
    corner_col = theme_color
    t = 3
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), corner_col, t)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), corner_col, t)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), corner_col, t)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), corner_col, t)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), corner_col, t)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), corner_col, t)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), corner_col, t)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), corner_col, t)

    cv2.putText(frame, "ALIGN FACE INSIDE BOX", (cx - 100, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

    foot_y = h - 25
    telemetry = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f} ms | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    cv2.putText(frame, telemetry, (20, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 135, 155), 1, cv2.LINE_AA)

    instructions = "Keys: [M] Toggle Mode (Pain / Emotion)  |  [Q] Exit"
    cv2.putText(frame, instructions, (sidebar_w + 20, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)

    return frame

def print_terminal_dashboard(mode, classes, probs, latency_ms, fps):
    top_pred_idx = int(np.argmax(probs))
    top_label = classes[top_pred_idx]
    top_conf = probs[top_pred_idx] * 100.0

    # Clear line and print dashboard
    sys.stdout.write("\033[H\033[J") # ANSI clear screen
    sys.stdout.write("==============================================================\n")
    sys.stdout.write(f"  BioVid Live Recognition [{mode.upper()}] | {fps:4.1f} FPS | {latency_ms:4.1f} ms\n")
    sys.stdout.write("==============================================================\n")
    sys.stdout.write(f"  >>> TOP PREDICTION: {top_label.upper()} ({top_conf:.1f}%) <<<\n\n")
    for idx, (lbl, p) in enumerate(zip(classes, probs)):
        bar = "█" * int(p * 30)
        marker = "==>" if idx == top_pred_idx else "   "
        sys.stdout.write(f"  {marker} {lbl:26s} [{p*100.0:5.1f}%] |{bar:<30s}|\n")
    sys.stdout.write("--------------------------------------------------------------\n")
    sys.stdout.write("  Browser View: http://localhost:5000  |  Press Ctrl+C to stop\n")
    sys.stdout.flush()

def run_webcam_demo(camera_idx=0, initial_mode="emotion", port=5000):
    global GLOBAL_STREAM_FRAME, FRAME_LOCK

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emo_model, pain_model = load_models(device)

    # Start browser streaming server in background
    start_web_server(port=port)

    print(f"\nOpening camera device {camera_idx}...", flush=True)
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"\n[ERROR] Unable to open camera at index {camera_idx}.")
        print("Tip: Check if another application (like Zoom or browser) is using your webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = initial_mode
    frame_buffer = deque(maxlen=16)
    smoothed_probs = np.ones(5, dtype=np.float32) / 5.0

    print("\n" + "=" * 60)
    print("  BioVid Live Camera Affective & Pain Recognition Active")
    print(f"  Current Mode: {mode.upper()}")
    print(f"  Web Live Stream: http://localhost:{port}")
    print("  Desktop Key: Press 'm' to toggle mode | 'q' to exit")
    print("=" * 60 + "\n", flush=True)

    prev_time = time.time()
    last_term_print = time.time()
    latency_ms = 18.0
    fps = 30.0
    gui_window_enabled = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            h, w, _ = frame.shape
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
                        logits = emo_model(input_tensor)
                    else:
                        logits = pain_model(input_tensor)

                    raw_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                latency_ms = (time.time() - t0) * 1000.0
                smoothed_probs = 0.65 * smoothed_probs + 0.35 * raw_probs

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
            prev_time = curr_time

            curr_classes = EMOTION_CLASSES if mode == "emotion" else PAIN_CLASSES
            curr_colors = EMOTION_COLORS if mode == "emotion" else PAIN_COLORS
            display_frame = draw_hud(frame, mode, curr_classes, curr_colors, smoothed_probs, latency_ms, fps)

            # Update global frame for browser stream
            with FRAME_LOCK:
                GLOBAL_STREAM_FRAME = display_frame.copy()

            # Attempt desktop GUI window display
            if gui_window_enabled:
                try:
                    cv2.imshow("BioVid Deep Learning Real-Time Recognition", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord('m') or key == ord('M'):
                        mode = "pain" if mode == "emotion" else "emotion"
                        smoothed_probs = np.ones(5, dtype=np.float32) / 5.0
                except Exception as gui_err:
                    print(f"\n[Note] Desktop window failed ({gui_err}). Switching smoothly to Browser & Terminal streaming mode...")
                    gui_window_enabled = False

            # Terminal dashboard print (every 0.25 seconds if GUI not active)
            if not gui_window_enabled and (curr_time - last_term_print > 0.25):
                print_terminal_dashboard(mode, curr_classes, smoothed_probs, latency_ms, fps)
                last_term_print = curr_time

    except KeyboardInterrupt:
        print("\nExiting by user request...")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("\nWebcam demo terminated cleanly.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioVid Live Real-Time Facial Affective & Pain Recognition")
    parser.add_argument("--camera_idx", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--mode", type=str, choices=["emotion", "pain"], default="emotion", help="Initial mode: 'emotion' or 'pain'")
    parser.add_argument("--port", type=int, default=5000, help="Web stream port (default: 5000)")
    args = parser.parse_args()

    run_webcam_demo(camera_idx=args.camera_idx, initial_mode=args.mode, port=args.port)
