"""
=============================================================================
HỆ THỐNG CAMERA CẢNH BÁO CHÁY - GIAO DIỆN TKINTER
YOLOv11 + CNN (EfficientNet-B0) + LSTM + Self-Attention
=============================================================================
Cài thư viện:
    pip install ultralytics torch torchvision opencv-python pillow
Chạy:
    python fire_detection_app.py
=============================================================================
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from ultralytics import YOLO
from collections import deque
import threading
import time
import os
import queue
import subprocess
import sys

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ─────────────────────────────────────────────
# BEEP (Windows-only fallback)
# ─────────────────────────────────────────────
try:
    import winsound
    def _do_beep():
        winsound.Beep(1000, 350)
except ImportError:
    def _do_beep():
        print("\a", end="", flush=True)

# ─────────────────────────────────────────────
# DEFAULT CONFIG
# ─────────────────────────────────────────────
DEFAULT = dict(
    camera_source       = 1,
    yolo_model_path     = "yolo_best.pt",
    lstm_model_path     = "lstm_best.pt",
    yolo_conf           = 0.40,
    lstm_conf           = 0.55,
    sequence_len        = 16,
    stop_record_delay   = 20.0,
    reset_fire_delay    = 3.0,   # giây thấy lửa liên tục thì bắt đầu quay
    miss_tolerance      = 1.5,   # giây bỏ qua miss (gió/nhiễu) trước khi reset đếm
    record_dir          = "fire_records",
    lstm_call_every     = 6,
    display_width       = 800,
    display_height      = 480,
)

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class CNNBackbone(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", feature_dim=512):
        super().__init__()
        if backbone_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=None)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        elif backbone_name == "resnet50":
            base = models.resnet50(weights=None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone_name == "mobilenet_v3":
            base = models.mobilenet_v3_small(weights=None)
            in_features = base.classifier[3].in_features
            base.classifier = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Backbone không hợp lệ: {backbone_name}")
        self.projection = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean([-2, -1])
        features = self.projection(features)
        return features.view(B, T, -1)


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out)), None


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cnn = CNNBackbone(
            backbone_name=config["cnn_backbone"],
            feature_dim=config["feature_dim"]
        )
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config["num_frames"], config["feature_dim"]) * 0.02
        )
        self.lstm = nn.LSTM(
            input_size=config["feature_dim"],
            hidden_size=config["lstm_hidden"],
            num_layers=config["lstm_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=config["dropout"] if config["lstm_layers"] > 1 else 0,
        )
        lstm_out_dim = config["lstm_hidden"] * 2
        self.attention = SelfAttention(d_model=lstm_out_dim,
                                       num_heads=config["num_heads"],
                                       dropout=config["dropout"])
        self.aggregate_norm = nn.LayerNorm(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(config["dropout"] * 0.5),
            nn.Linear(64, config["num_classes"]),
        )

    def forward(self, x):
        features = self.cnn(x)
        features = features + self.pos_encoding
        lstm_out, _ = self.lstm(features)
        attn_out, _ = self.attention(lstm_out)
        attn_out = self.aggregate_norm(attn_out)
        mean_pool = attn_out.mean(dim=1)
        max_pool = attn_out.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.classifier(pooled)


FRAME_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return FRAME_TRANSFORM(rgb)


# ─────────────────────────────────────────────
# DETECTION THREAD
# ─────────────────────────────────────────────

class DetectionThread(threading.Thread):
    def __init__(self, cfg, result_queue, stop_event):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = result_queue
        self.stop_event = stop_event

        # shared mutable config (can be changed live from GUI)
        self.yolo_conf        = cfg["yolo_conf"]
        self.lstm_conf        = cfg["lstm_conf"]
        self.stop_rec_delay   = cfg["stop_record_delay"]
        self.reset_fire_delay = cfg["reset_fire_delay"]
        self.miss_tolerance   = cfg["miss_tolerance"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

        os.makedirs(cfg["record_dir"], exist_ok=True)

        self._beep_active = False
        self._beep_thread = None

        # recorder state
        self.writer            = None
        self.is_recording      = False
        self.fire_first_seen   = None   # lần đầu thấy lửa (đếm delay trước khi quay)
        self.last_fire_time    = None   # lần cuối thấy lửa (đếm dừng quay)
        self.miss_since        = None   # lần đầu mất lửa (để bỏ qua miss ngắn)
        self._cam_fps          = 30.0

        # writer thread riêng để không block detection loop
        self._write_queue  = queue.Queue(maxsize=60)
        self._write_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._write_thread.start()

    def _load_models(self):
        self.yolo = YOLO(self.cfg["yolo_model_path"])
        ckpt = torch.load(self.cfg["lstm_model_path"], map_location=self.device)
        config = ckpt["config"]
        self.seq_len = config["num_frames"]
        self.lstm_model = LSTMAttentionClassifier(config)
        self.lstm_model.load_state_dict(ckpt["model_state_dict"])
        self.lstm_model.eval()
        self.lstm_model.to(self.device)

    def run(self):
        # ── Thread 1: capture ─────────────────────────────────────────
        # Chỉ đọc frame từ camera, không làm gì khác → không bao giờ bị block bởi inference
        self._cap_queue = queue.Queue(maxsize=2)   # luôn giữ frame mới nhất

        def _capture_loop():
            cap = cv2.VideoCapture(self.cfg["camera_source"])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg["display_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["display_height"])
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    cap.release()
                    cap = cv2.VideoCapture(self.cfg["camera_source"])
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                # luôn giữ frame mới nhất, bỏ frame cũ nếu queue đầy
                if self._cap_queue.full():
                    try:
                        self._cap_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._cap_queue.put(frame)
            cap.release()

        cap_thread = threading.Thread(target=_capture_loop, daemon=True)
        cap_thread.start()

        # ── Thread 2 (main run thread): inference ─────────────────────
        frame_buffer = deque(maxlen=self.seq_len)
        lstm_fire    = False
        lstm_conf    = 0.0
        frame_count  = 0
        self.call_every = self.cfg["lstm_call_every"]

        _fps_t0     = time.time()
        _fps_count  = 0

        while not self.stop_event.is_set():
            try:
                frame = self._cap_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_count += 1
            _fps_count  += 1

            # đo FPS thật mỗi 2 giây → dùng cho VideoWriter
            _now = time.time()
            if _now - _fps_t0 >= 2.0:
                self._cam_fps = max(_fps_count / (_now - _fps_t0), 5.0)
                _fps_count = 0
                _fps_t0    = _now

            # YOLO
            results  = self.yolo(frame, conf=self.yolo_conf, verbose=False)
            yolo_det = any(len(r.boxes) > 0 for r in results)

            # buffer + LSTM
            frame_buffer.append(preprocess_frame(frame))
            if len(frame_buffer) == self.seq_len and frame_count % self.call_every == 0:
                with torch.no_grad():
                    seq    = torch.stack(list(frame_buffer)).unsqueeze(0).to(self.device)
                    logits = self.lstm_model(seq)
                    probs  = torch.softmax(logits, dim=1)[0]
                    lstm_conf = float(probs[1])
                    lstm_fire = lstm_conf >= self.lstm_conf

            confirmed = yolo_det and lstm_fire

            # beep
            if confirmed:
                self._start_beep()
            else:
                self._stop_beep()

            # ── RECORD LOGIC ──────────────────────────────────────────
            now = time.time()
            if confirmed:
                self.last_fire_time = now
                self.miss_since     = None
                if not self.is_recording:
                    if self.fire_first_seen is None:
                        self.fire_first_seen = now
                    if (now - self.fire_first_seen) >= self.reset_fire_delay:
                        self._start_recording(frame)
            else:
                if not self.is_recording and self.fire_first_seen is not None:
                    if self.miss_since is None:
                        self.miss_since = now
                    elif (now - self.miss_since) > self.miss_tolerance:
                        self.fire_first_seen = None
                        self.miss_since      = None

            if self.is_recording:
                try:
                    self._write_queue.put_nowait(frame)
                except queue.Full:
                    pass
                if self.last_fire_time is not None and (now - self.last_fire_time) > self.stop_rec_delay:
                    self._stop_recording()
                    self.last_fire_time  = None
                    self.fire_first_seen = None
                    self.miss_since      = None
            # ──────────────────────────────────────────────────────────

            # draw
            display = frame.copy()
            if yolo_det:
                display = self._draw_boxes(display, results, lstm_fire)

            if confirmed:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], 50), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)
                cv2.putText(display, "CANH BAO CHAY!", (12, 36),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 60, 255), 2)
            else:
                cv2.putText(display, "An toan", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 220, 80), 2)

            cv2.putText(display, f"LSTM: {lstm_conf:.2f}", (12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (80, 80, 255) if lstm_fire else (160, 160, 160), 1)
            cv2.putText(display, f"YOLO: {'DETECT' if yolo_det else 'clear'}", (12, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 165, 255) if yolo_det else (160, 160, 160), 1)

            if self.is_recording:
                cv2.circle(display, (display.shape[1] - 28, 22), 9, (0, 0, 255), -1)
                cv2.putText(display, "REC", (display.shape[1] - 70, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if self.q.qsize() < 2:
                self.q.put({
                    "frame":     display,
                    "confirmed": confirmed,
                    "lstm_conf": lstm_conf,
                    "yolo_det":  yolo_det,
                    "recording": self.is_recording,
                })

        self._stop_recording()
        self._stop_beep()

    def _writer_loop(self):
        """Thread riêng ghi frame vào disk, không block detection loop."""
        while True:
            frame = self._write_queue.get()
            if frame is None:
                break
            if self.writer:
                self.writer.write(frame)

    def _start_recording(self, frame):
        h, w = frame.shape[:2]
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.cfg["record_dir"], f"fire_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = max(self._cam_fps, 20.0)
        self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        self.is_recording = True

    def _stop_recording(self):
        if self.writer:
            while not self._write_queue.empty():
                try:
                    self.writer.write(self._write_queue.get_nowait())
                except queue.Empty:
                    break
            self.writer.release()
            self.writer = None
            print("[INFO] Video đã lưu xong.")
        self.is_recording = False

    def _draw_boxes(self, frame, results, lstm_fire):
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = r.names.get(cls_id, str(cls_id))
                # fire=đỏ, smoke=xanh dương; nếu LSTM chưa xác nhận thì nhạt hơn
                if label == "fire":
                    color = (0, 0, 255) if lstm_fire else (0, 80, 180)
                else:  # smoke
                    color = (255, 120, 0) if lstm_fire else (180, 80, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def _beep_loop(self):
        while self._beep_active:
            _do_beep()
            time.sleep(0.15)

    def _start_beep(self):
        if not self._beep_active:
            self._beep_active = True
            self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
            self._beep_thread.start()

    def _stop_beep(self):
        self._beep_active = False


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────

DARK_BG   = "#0f1117"
PANEL_BG  = "#16181f"
CARD_BG   = "#1e2130"
ACCENT    = "#ff4d1c"
ACCENT2   = "#ff8c42"
TEXT_W    = "#e8eaf0"
TEXT_DIM  = "#6b7280"
SUCCESS   = "#22c55e"
WARNING   = "#f59e0b"
FONT_HEAD = ("Courier New", 13, "bold")
FONT_BODY = ("Courier New", 10)
FONT_MONO = ("Courier New", 9)


class FireApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔥 FIRE DETECTION SYSTEM")
        self.configure(bg=DARK_BG)
        self.geometry("1100x680")
        self.minsize(900, 560)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.cfg = dict(DEFAULT)
        self.det_thread  = None
        self.stop_event  = threading.Event()
        self.frame_queue = queue.Queue(maxsize=3)
        self.running     = False
        self._shared_models = None

        self._build_ui()
        self._poll_frame()

    # ── UI BUILD ──────────────────────────────

    def _build_ui(self):
        # ── header bar
        hdr = tk.Frame(self, bg=DARK_BG)
        hdr.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(hdr, text="🔥 FIRE DETECTION", font=("Courier New", 18, "bold"),
                 bg=DARK_BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text="YOLOv11 + LSTM + ATTENTION",
                 font=FONT_MONO, bg=DARK_BG, fg=TEXT_DIM).pack(side="left", padx=12)

        # ── notebook tabs
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Fire.TNotebook",
                        background=DARK_BG, borderwidth=0)
        style.configure("Fire.TNotebook.Tab",
                        background=CARD_BG, foreground=TEXT_DIM,
                        font=FONT_HEAD, padding=[16, 8],
                        borderwidth=0)
        style.map("Fire.TNotebook.Tab",
                  background=[("selected", PANEL_BG)],
                  foreground=[("selected", ACCENT)])

        nb = ttk.Notebook(self, style="Fire.TNotebook")
        nb.pack(fill="both", expand=True, padx=16, pady=8)

        self.tab_cam  = tk.Frame(nb, bg=PANEL_BG)
        self.tab_img  = tk.Frame(nb, bg=PANEL_BG)
        self.tab_file = tk.Frame(nb, bg=PANEL_BG)
        self.tab_vid  = tk.Frame(nb, bg=PANEL_BG)
        nb.add(self.tab_cam,  text="  📷  CAMERA  ")
        nb.add(self.tab_img,  text="  🖼  ẢNH  ")
        nb.add(self.tab_file, text="  🎬  VIDEO FILE  ")
        nb.add(self.tab_vid,  text="  🎥  VIDEO LƯU  ")
        nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self._build_cam_tab()
        self._build_img_tab()
        self._build_file_tab()
        self._build_vid_tab()

    # ── CAMERA TAB ────────────────────────────

    def _build_cam_tab(self):
        root = self.tab_cam
        # left: video feed
        left = tk.Frame(root, bg=PANEL_BG)
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)

        self.video_label = tk.Label(left, bg="#000000")
        self.video_label.pack(fill="both", expand=True)
        self.video_label.config(width=80, height=30)

        # status bar under video
        self.status_var = tk.StringVar(value="⬤  Chưa khởi động")
        tk.Label(left, textvariable=self.status_var,
                 font=FONT_BODY, bg=DARK_BG, fg=TEXT_DIM,
                 anchor="w").pack(fill="x", pady=(4, 0))

        # right: control panel
        right = tk.Frame(root, bg=CARD_BG, width=260)
        right.pack(side="right", fill="y", padx=(0, 12), pady=12)
        right.pack_propagate(False)

        self._build_controls(right)

    def _build_controls(self, parent):
        pad = dict(padx=14, pady=5)

        def section(text):
            f = tk.Frame(parent, bg=ACCENT, height=2)
            f.pack(fill="x", padx=14, pady=(14, 2))
            tk.Label(parent, text=text, font=("Courier New", 10, "bold"),
                     bg=CARD_BG, fg=ACCENT2).pack(anchor="w", **pad)

        # ── Camera source
        section("NGUỒN CAMERA")
        tk.Label(parent, text="Index / URL:", font=FONT_MONO,
                 bg=CARD_BG, fg=TEXT_DIM).pack(anchor="w", padx=14)
        self.cam_src_var = tk.StringVar(value=str(self.cfg["camera_source"]))
        cam_entry = tk.Entry(parent, textvariable=self.cam_src_var,
                             font=FONT_BODY, bg="#252836", fg=TEXT_W,
                             insertbackground=TEXT_W, relief="flat", bd=0)
        cam_entry.pack(fill="x", padx=14, pady=(0, 4), ipady=5)

        # ── Thresholds
        section("NGƯỠNG PHÁT HIỆN")

        self.yolo_conf_var = tk.DoubleVar(value=self.cfg["yolo_conf"])
        self.lstm_conf_var = tk.DoubleVar(value=self.cfg["lstm_conf"])

        self._slider_row(parent, "YOLO conf", self.yolo_conf_var, 0.1, 0.95)
        self._slider_row(parent, "LSTM conf", self.lstm_conf_var, 0.1, 0.99)

        tk.Label(parent, text="LSTM N frame (tang=nhanh hon, it chinh xac hon)",
                 font=("Courier New", 8), bg=CARD_BG, fg=TEXT_DIM,
                 justify="left").pack(anchor="w", padx=14)
        self.lstm_every_var = tk.IntVar(value=self.cfg["lstm_call_every"])
        self._slider_row(parent, "LSTM moi N frame", self.lstm_every_var, 1, 30, resolution=1)

        # ── Video recording
        section("CÀI ĐẶT QUAY VIDEO")

        self.reset_delay_var   = tk.DoubleVar(value=self.cfg["reset_fire_delay"])
        self.stop_delay_var    = tk.DoubleVar(value=self.cfg["stop_record_delay"])
        self.miss_tol_var      = tk.DoubleVar(value=self.cfg["miss_tolerance"])

        self._slider_row(parent, "Bat dau quay sau (s)", self.reset_delay_var, 1, 30, resolution=0.5)
        self._slider_row(parent, "Bo qua miss (s)",      self.miss_tol_var,    0.5, 10, resolution=0.5)
        self._slider_row(parent, "Dung quay (s)",        self.stop_delay_var,  5, 60, resolution=1)

        # ── Start / Stop
        section("ĐIỀU KHIỂN")
        self.start_btn = tk.Button(
            parent, text="▶  BẮT ĐẦU",
            font=("Courier New", 11, "bold"),
            bg=ACCENT, fg="white", activebackground="#cc3a14",
            relief="flat", bd=0, cursor="hand2",
            command=self._toggle_detection
        )
        self.start_btn.pack(fill="x", padx=14, pady=(4, 4), ipady=8)

        # alert indicator
        self.alert_label = tk.Label(parent, text="",
                                     font=("Courier New", 11, "bold"),
                                     bg=CARD_BG, fg=ACCENT)
        self.alert_label.pack(pady=4)

        # live stats
        section("THỐNG KÊ LIVE")
        self.stats_var = tk.StringVar(value="LSTM: —\nYOLO: —\nREC:  —")
        tk.Label(parent, textvariable=self.stats_var,
                 font=FONT_MONO, bg=CARD_BG, fg=TEXT_DIM,
                 justify="left").pack(anchor="w", padx=14, pady=4)

    def _slider_row(self, parent, label, var, from_, to, resolution=0.01):
        tk.Label(parent, text=label, font=FONT_MONO,
                 bg=CARD_BG, fg=TEXT_DIM).pack(anchor="w", padx=14)
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill="x", padx=14, pady=(0, 6))

        val_lbl = tk.Label(row, text=f"{var.get():.2f}",
                           font=FONT_MONO, bg=CARD_BG, fg=TEXT_W, width=5)
        val_lbl.pack(side="right")

        slider = tk.Scale(row, variable=var, from_=from_, to=to,
                          resolution=resolution, orient="horizontal",
                          showvalue=False, sliderlength=14,
                          bg=CARD_BG, fg=TEXT_DIM,
                          troughcolor="#252836", activebackground=ACCENT,
                          highlightthickness=0, bd=0,
                          command=lambda v, lbl=val_lbl: lbl.config(text=f"{float(v):.2f}"))
        slider.pack(side="left", fill="x", expand=True)

    # ── HELPER ────────────────────────────────

    def _resize_fit(self, frame, max_w, max_h):
        """Resize frame giữ tỉ lệ, không vượt quá max_w x max_h."""
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(frame, (nw, nh)) if scale < 1.0 else frame

    # ── IMAGE TAB ─────────────────────────────

    def _build_img_tab(self):
        root = self.tab_img

        left = tk.Frame(root, bg="#000000")
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)

        self.img_preview = tk.Label(left, bg="#000000",
                                     text="Chưa chọn ảnh", fg=TEXT_DIM, font=FONT_BODY)
        self.img_preview.pack(fill="both", expand=True)

        self.img_status = tk.Label(left, text="", font=FONT_BODY,
                                    bg=DARK_BG, fg=TEXT_DIM, anchor="w")
        self.img_status.pack(fill="x", pady=(4, 0))

        right = tk.Frame(root, bg=CARD_BG, width=260)
        right.pack(side="right", fill="y", padx=(0, 12), pady=12)
        right.pack_propagate(False)

        def sec(t):
            tk.Frame(right, bg=ACCENT, height=2).pack(fill="x", padx=14, pady=(14, 2))
            tk.Label(right, text=t, font=("Courier New", 10, "bold"),
                     bg=CARD_BG, fg=ACCENT2).pack(anchor="w", padx=14, pady=5)

        sec("CHỌN ẢNH")
        tk.Button(right, text="📂  Chọn file ảnh", font=FONT_BODY,
                  bg=CARD_BG, fg=TEXT_W, activebackground=DARK_BG,
                  relief="flat", bd=0, cursor="hand2",
                  command=self._pick_image).pack(fill="x", padx=14, ipady=6)
        self.img_path_var = tk.StringVar(value="Chưa chọn file")
        tk.Label(right, textvariable=self.img_path_var, font=("Courier New", 8),
                 bg=CARD_BG, fg=TEXT_DIM, wraplength=220,
                 justify="left").pack(anchor="w", padx=14, pady=(4, 0))

        sec("NGƯỠNG")
        self.img_yolo_var = tk.DoubleVar(value=self.cfg["yolo_conf"])
        self._slider_row(right, "YOLO conf", self.img_yolo_var, 0.1, 0.95)

        sec("PHÂN TÍCH")
        tk.Button(right, text="🔍  Phân tích ảnh",
                  font=("Courier New", 11, "bold"),
                  bg=ACCENT, fg="white", activebackground="#cc3a14",
                  relief="flat", bd=0, cursor="hand2",
                  command=self._analyze_image).pack(fill="x", padx=14, ipady=8)

        sec("KẾT QUẢ")
        self.img_result_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.img_result_var,
                 font=("Courier New", 12, "bold"),
                 bg=CARD_BG, fg=TEXT_W, justify="left",
                 wraplength=220).pack(anchor="w", padx=14, pady=6)

        self._img_path = None

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        self._img_path = path
        self.img_path_var.set(os.path.basename(path))
        img = Image.open(path).convert("RGB")
        img.thumbnail((700, 460), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.img_preview.config(image=photo, text="")
        self.img_preview._img = photo
        self.img_result_var.set("—")
        self.img_status.config(text="Ảnh đã tải. Nhấn Phân tích.", fg=TEXT_DIM)

    def _analyze_image(self):
        if not self._img_path:
            messagebox.showinfo("Thông báo", "Chọn ảnh trước.")
            return
        if not self._shared_models:
            messagebox.showwarning("Chưa sẵn sàng", "Bấm BẮT ĐẦU ở tab CAMERA trước để load model.")
            return

        frame = cv2.imread(self._img_path)
        if frame is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return

        yolo, lstm_model, device, seq_len = self._shared_models
        results  = yolo(frame, conf=self.img_yolo_var.get(), verbose=False)
        yolo_det = any(len(r.boxes) > 0 for r in results)

        tensor = preprocess_frame(frame)
        seq    = torch.stack([tensor] * seq_len).unsqueeze(0).to(device)
        with torch.no_grad():
            probs     = torch.softmax(lstm_model(seq), dim=1)[0]
            lstm_c    = float(probs[1])
            lstm_fire = lstm_c >= self.cfg["lstm_conf"]

        confirmed = yolo_det and lstm_fire

        display = frame.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                c   = float(box.conf[0])
                cid = int(box.cls[0])
                lbl = r.names.get(cid, str(cid))
                color = (0, 0, 255) if lbl == "fire" else (255, 120, 0)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{lbl} {c:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        txt = "CANH BAO CHAY!" if confirmed else "An toan"
        col = (0, 0, 255) if confirmed else (60, 220, 80)
        cv2.putText(display, txt, (12, 36), cv2.FONT_HERSHEY_DUPLEX, 1.1, col, 2)
        cv2.putText(display, f"LSTM: {lstm_c:.2f}", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 255), 1)

        # resize giữ tỉ lệ để hiện preview
        lw = max(self.img_preview.winfo_width(), 100)
        lh = max(self.img_preview.winfo_height(), 100)
        display = self._resize_fit(display, lw, lh)
        rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.img_preview.config(image=photo, text="")
        self.img_preview._img = photo

        if confirmed:
            self.img_result_var.set("🔥 PHÁT HIỆN CHÁY!")
            self.img_status.config(
                text=f"YOLO: detect | LSTM: {lstm_c:.2f} | KẾT LUẬN: CHÁY", fg=ACCENT)
        else:
            self.img_result_var.set("✔  An toàn")
            self.img_status.config(
                text=f"YOLO: {'detect' if yolo_det else 'clear'} | LSTM: {lstm_c:.2f} | AN TOÀN",
                fg=SUCCESS)

    # ── VIDEO FILE TAB ────────────────────────

    def _build_file_tab(self):
        root = self.tab_file

        left = tk.Frame(root, bg="#000000")
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)

        self.file_preview = tk.Label(left, bg="#000000",
                                      text="Chưa chọn video", fg=TEXT_DIM, font=FONT_BODY)
        self.file_preview.pack(fill="both", expand=True)

        self.file_progress = ttk.Progressbar(left, orient="horizontal",
                                               mode="determinate")
        self.file_progress.pack(fill="x", pady=(4, 0))

        self.file_status = tk.Label(left, text="", font=FONT_BODY,
                                     bg=DARK_BG, fg=TEXT_DIM, anchor="w")
        self.file_status.pack(fill="x", pady=(2, 0))

        right = tk.Frame(root, bg=CARD_BG, width=260)
        right.pack(side="right", fill="y", padx=(0, 12), pady=12)
        right.pack_propagate(False)

        def sec(t):
            tk.Frame(right, bg=ACCENT, height=2).pack(fill="x", padx=14, pady=(14, 2))
            tk.Label(right, text=t, font=("Courier New", 10, "bold"),
                     bg=CARD_BG, fg=ACCENT2).pack(anchor="w", padx=14, pady=5)

        sec("CHỌN VIDEO")
        tk.Button(right, text="📂  Chọn file video", font=FONT_BODY,
                  bg=CARD_BG, fg=TEXT_W, activebackground=DARK_BG,
                  relief="flat", bd=0, cursor="hand2",
                  command=self._pick_video_file).pack(fill="x", padx=14, ipady=6)
        self.file_path_var = tk.StringVar(value="Chưa chọn file")
        tk.Label(right, textvariable=self.file_path_var, font=("Courier New", 8),
                 bg=CARD_BG, fg=TEXT_DIM, wraplength=220,
                 justify="left").pack(anchor="w", padx=14, pady=(4, 0))

        sec("NGƯỠNG")
        self.file_yolo_var  = tk.DoubleVar(value=self.cfg["yolo_conf"])
        self.file_lstm_var  = tk.DoubleVar(value=self.cfg["lstm_conf"])
        self.file_every_var = tk.IntVar(value=self.cfg["lstm_call_every"])
        self._slider_row(right, "YOLO conf",        self.file_yolo_var,  0.1, 0.95)
        self._slider_row(right, "LSTM conf",        self.file_lstm_var,  0.1, 0.99)
        self._slider_row(right, "LSTM moi N frame", self.file_every_var, 1,   30, resolution=1)

        sec("PHÂN TÍCH")
        self.file_btn = tk.Button(right, text="▶  Phân tích video",
                                   font=("Courier New", 11, "bold"),
                                   bg=ACCENT, fg="white", activebackground="#cc3a14",
                                   relief="flat", bd=0, cursor="hand2",
                                   command=self._toggle_file_analysis)
        self.file_btn.pack(fill="x", padx=14, ipady=8)

        sec("KẾT QUẢ")
        self.file_result_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.file_result_var,
                 font=("Courier New", 10, "bold"),
                 bg=CARD_BG, fg=TEXT_W, justify="left",
                 wraplength=220).pack(anchor="w", padx=14, pady=6)

        self._file_path    = None
        self._file_running = False
        self._file_stop    = threading.Event()
        self._file_queue   = queue.Queue(maxsize=3)

    def _pick_video_file(self):
        path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")]
        )
        if not path:
            return
        self._file_path = path
        self.file_path_var.set(os.path.basename(path))
        self.file_result_var.set("—")
        self.file_status.config(text="Video đã tải. Nhấn Phân tích.", fg=TEXT_DIM)
        self.file_progress["value"] = 0

    def _toggle_file_analysis(self):
        if not self._file_running:
            self._start_file_analysis()
        else:
            self._stop_file_analysis()

    def _start_file_analysis(self):
        if not self._file_path:
            messagebox.showinfo("Thông báo", "Chọn video trước.")
            return
        if not self._shared_models:
            messagebox.showwarning("Chưa sẵn sàng", "Bấm BẮT ĐẦU ở tab CAMERA trước để load model.")
            return
        self._file_running = True
        self._file_stop.clear()
        self.file_btn.config(text="■  Dừng", bg="#333")
        self.file_result_var.set("Đang phân tích...")
        self.file_progress["value"] = 0
        threading.Thread(target=self._file_analysis_thread, daemon=True).start()
        self._poll_file_frame()

    def _stop_file_analysis(self):
        self._file_stop.set()
        self._file_running = False
        self.file_btn.config(text="▶  Phân tích video", bg=ACCENT)

    def _file_analysis_thread(self):
        yolo, lstm_model, device, seq_len = self._shared_models
        yolo_conf  = self.file_yolo_var.get()
        lstm_conf  = self.file_lstm_var.get()
        call_every = int(self.file_every_var.get())

        cap         = cv2.VideoCapture(self._file_path)
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        frame_buf   = deque(maxlen=seq_len)
        frame_count = 0
        lstm_fire   = False
        lstm_c      = 0.0
        fire_frames = 0

        while not self._file_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = (frame_count / total) * 100

            results  = yolo(frame, conf=yolo_conf, verbose=False)
            yolo_det = any(len(r.boxes) > 0 for r in results)

            frame_buf.append(preprocess_frame(frame))
            if len(frame_buf) == seq_len and frame_count % call_every == 0:
                with torch.no_grad():
                    seq    = torch.stack(list(frame_buf)).unsqueeze(0).to(device)
                    probs  = torch.softmax(lstm_model(seq), dim=1)[0]
                    lstm_c    = float(probs[1])
                    lstm_fire = lstm_c >= lstm_conf

            confirmed = yolo_det and lstm_fire
            if confirmed:
                fire_frames += 1

            display = frame.copy()
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    c   = float(box.conf[0])
                    cid = int(box.cls[0])
                    lbl = r.names.get(cid, str(cid))
                    color = (0, 0, 255) if lbl == "fire" else (255, 120, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, f"{lbl} {c:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            txt = "CANH BAO CHAY!" if confirmed else "An toan"
            col = (0, 0, 255) if confirmed else (60, 220, 80)
            cv2.putText(display, txt, (12, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
            cv2.putText(display, f"LSTM:{lstm_c:.2f}  {frame_count}/{total}",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 255), 1)

            if self._file_queue.qsize() < 2:
                self._file_queue.put({
                    "frame":       display,
                    "confirmed":   confirmed,
                    "progress":    progress,
                    "frame_count": frame_count,
                    "total":       total,
                    "fire_frames": fire_frames,
                    "lstm_conf":   lstm_c,
                })

        cap.release()
        fire_pct = (fire_frames / max(frame_count, 1)) * 100
        self._file_queue.put({
            "done":         True,
            "fire_frames":  fire_frames,
            "total_frames": frame_count,
            "fire_pct":     fire_pct,
        })

    def _poll_file_frame(self):
        try:
            data = self._file_queue.get_nowait()
        except queue.Empty:
            data = None

        if data:
            if data.get("done"):
                self._file_running = False
                self.file_btn.config(text="▶  Phân tích video", bg=ACCENT)
                self.file_progress["value"] = 100
                pct = data["fire_pct"]
                verdict = "🔥 PHÁT HIỆN CHÁY!" if pct > 5 else "✔  An toàn"
                self.file_result_var.set(
                    f"{verdict}\nFrame cháy: {data['fire_frames']}/{data['total_frames']}\nTỉ lệ: {pct:.1f}%"
                )
                self.file_status.config(
                    text=f"Hoàn thành | Tỉ lệ cháy: {pct:.1f}%",
                    fg=ACCENT if pct > 5 else SUCCESS)
                return

            frame = data["frame"]
            # resize giữ tỉ lệ, không phóng to bất thường
            lw = max(self.file_preview.winfo_width(), 100)
            lh = max(self.file_preview.winfo_height(), 100)
            frame = self._resize_fit(frame, lw, lh)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.file_preview.config(image=photo, text="")
            self.file_preview._img = photo
            self.file_progress["value"] = data["progress"]
            self.file_status.config(
                text=f"Frame {data['frame_count']}/{data['total']} | "
                     f"Cháy: {data['fire_frames']} | LSTM: {data['lstm_conf']:.2f}",
                fg=ACCENT if data["confirmed"] else TEXT_DIM)

        if self._file_running:
            self.after(30, self._poll_file_frame)

    # ── VIDEO LIBRARY TAB ─────────────────────

    def _build_vid_tab(self):
        root = self.tab_vid

        top = tk.Frame(root, bg=PANEL_BG)
        top.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(top, text="Video đã lưu", font=FONT_HEAD,
                 bg=PANEL_BG, fg=TEXT_W).pack(side="left")
        tk.Button(top, text="↺  Làm mới", font=FONT_MONO,
                  bg=CARD_BG, fg=ACCENT2, activebackground=CARD_BG,
                  relief="flat", bd=0, cursor="hand2",
                  command=self._refresh_videos).pack(side="right", padx=4, ipady=4, ipadx=8)

        cols = ("Tên file", "Kích thước", "Thời gian")
        style = ttk.Style()
        style.configure("Fire.Treeview",
                        background=CARD_BG, fieldbackground=CARD_BG,
                        foreground=TEXT_W, rowheight=26,
                        font=FONT_MONO, borderwidth=0)
        style.configure("Fire.Treeview.Heading",
                        background=DARK_BG, foreground=ACCENT2,
                        font=("Courier New", 10, "bold"),
                        relief="flat")
        style.map("Fire.Treeview",
                  background=[("selected", ACCENT)])

        frame_tree = tk.Frame(root, bg=PANEL_BG)
        frame_tree.pack(fill="both", expand=True, padx=16, pady=4)

        self.tree = ttk.Treeview(frame_tree, columns=cols,
                                  show="headings", style="Fire.Treeview",
                                  selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("Tên file",    width=340)
        self.tree.column("Kích thước",  width=100, anchor="center")
        self.tree.column("Thời gian",   width=180, anchor="center")
        sb = ttk.Scrollbar(frame_tree, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        btn_row = tk.Frame(root, bg=PANEL_BG)
        btn_row.pack(fill="x", padx=16, pady=(4, 14))
        tk.Button(btn_row, text="▶  Mở video",
                  font=FONT_BODY, bg=ACCENT, fg="white",
                  activebackground="#cc3a14", relief="flat", bd=0,
                  cursor="hand2", command=self._open_selected_video,
                  ).pack(side="left", ipady=6, ipadx=14, padx=(0, 8))
        tk.Button(btn_row, text="🗑  Xoá",
                  font=FONT_BODY, bg=CARD_BG, fg=WARNING,
                  activebackground=DARK_BG, relief="flat", bd=0,
                  cursor="hand2", command=self._delete_selected_video,
                  ).pack(side="left", ipady=6, ipadx=14)

        self._refresh_videos()

    def _refresh_videos(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        d = self.cfg["record_dir"]
        if not os.path.isdir(d):
            return
        files = sorted(
            [f for f in os.listdir(d) if f.endswith(".avi")],
            reverse=True
        )
        for f in files:
            path = os.path.join(d, f)
            size = os.path.getsize(path) / 1024 / 1024
            mtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(os.path.getmtime(path)))
            self.tree.insert("", "end", values=(f, f"{size:.1f} MB", mtime))

    def _open_selected_video(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Thông báo", "Chọn 1 video trước.")
            return
        fname = self.tree.item(sel[0])["values"][0]
        path  = os.path.abspath(os.path.join(self.cfg["record_dir"], fname))
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _delete_selected_video(self):
        sel = self.tree.selection()
        if not sel:
            return
        fname = self.tree.item(sel[0])["values"][0]
        if messagebox.askyesno("Xác nhận", f"Xoá file {fname}?"):
            os.remove(os.path.join(self.cfg["record_dir"], fname))
            self._refresh_videos()

    # ── DETECTION CONTROL ─────────────────────

    def _toggle_detection(self):
        if not self.running:
            self._start_detection()
        else:
            self._stop_detection()

    def _start_detection(self):
        # apply current slider values
        src = self.cam_src_var.get()
        try:
            self.cfg["camera_source"] = int(src)
        except ValueError:
            self.cfg["camera_source"] = src

        self.cfg["yolo_conf"]           = round(self.yolo_conf_var.get(), 2)
        self.cfg["lstm_conf"]           = round(self.lstm_conf_var.get(), 2)
        self.cfg["reset_fire_delay"]    = round(self.reset_delay_var.get(), 1)
        self.cfg["stop_record_delay"]   = round(self.stop_delay_var.get(), 1)
        self.cfg["miss_tolerance"]      = round(self.miss_tol_var.get(), 1)

        self.stop_event.clear()
        self.frame_queue = queue.Queue(maxsize=3)
        self.det_thread  = DetectionThread(self.cfg, self.frame_queue, self.stop_event)
        self.det_thread.start()
        self.after(2000, self._grab_shared_models)
        self.running = True
        self.start_btn.config(text="■  DỪNG", bg="#333")
        self.status_var.set("⬤  Đang chạy...")

    def _grab_shared_models(self):
        if self.det_thread and hasattr(self.det_thread, 'yolo'):
            self._shared_models = (
                self.det_thread.yolo,
                self.det_thread.lstm_model,
                self.det_thread.device,
                self.det_thread.seq_len,
            )

    def _stop_detection(self):
        self.stop_event.set()
        self.running = False
        self.start_btn.config(text="▶  BẮT ĐẦU", bg=ACCENT)
        self.status_var.set("⬤  Đã dừng")
        self.video_label.config(image="", bg="#000000")
        self.alert_label.config(text="")

    # ── FRAME POLL ────────────────────────────

    def _poll_frame(self):
        if self.running:
            # sync sliders live (allow adjusting while running)
            if self.det_thread:
                self.det_thread.yolo_conf      = self.yolo_conf_var.get()
                self.det_thread.lstm_conf      = self.lstm_conf_var.get()
                self.det_thread.stop_rec_delay   = self.stop_delay_var.get()
                self.det_thread.reset_fire_delay = self.reset_delay_var.get()
                self.det_thread.miss_tolerance   = self.miss_tol_var.get()
                self.det_thread.call_every     = int(self.lstm_every_var.get())

            try:
                data = self.frame_queue.get_nowait()
            except queue.Empty:
                data = None

            if data:
                frame = data["frame"]
                # resize to fit label
                lw = self.video_label.winfo_width()
                lh = self.video_label.winfo_height()
                if lw > 10 and lh > 10:
                    frame = cv2.resize(frame, (lw, lh))
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.config(image=img)
                self.video_label._img = img   # keep reference

                # alert indicator
                if data["confirmed"]:
                    self.alert_label.config(text="🔥 CẢNH BÁO CHÁY!", fg=ACCENT)
                    self.status_var.set("⬤  CHÁY được phát hiện!")
                else:
                    self.alert_label.config(text="✔  An toàn", fg=SUCCESS)
                    self.status_var.set("⬤  Đang chạy...")

                rec_txt = "● ĐANG QUAY" if data["recording"] else "—"
                self.stats_var.set(
                    f"LSTM: {data['lstm_conf']:.2f}\n"
                    f"YOLO: {'DETECT' if data['yolo_det'] else 'clear'}\n"
                    f"REC:  {rec_txt}"
                )

        self.after(25, self._poll_frame)   # ~40 fps UI refresh

    # ── TAB CHANGE ────────────────────────────

    def _on_tab_change(self, event):
        tab = event.widget.tab("current", "text")
        if "VIDEO LƯU" in tab:
            self._refresh_videos()

    # ── CLOSE ─────────────────────────────────

    def _on_close(self):
        self._stop_detection()
        time.sleep(0.3)
        self.destroy()


# ─────────────────────────────────────────────
# AUTO OPEN iVCAM
# ─────────────────────────────────────────────
def auto_open_ivcam():
    ivcam_exe = None

    # Cách 1: Tìm trong registry
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SOFTWARE\e2eSoft\iVCam")
        install_path = winreg.QueryValueEx(key, "InstallPath")[0]
        winreg.CloseKey(key)
        candidate = os.path.join(install_path, "iVCam.exe")
        if os.path.exists(candidate):
            ivcam_exe = candidate
    except Exception:
        pass

    # Cách 2: Thử các đường dẫn phổ biến
    if not ivcam_exe:
        common_paths = [
            r"C:\Program Files\e2eSoft\iVCam\iVCam.exe",
            r"C:\Program Files (x86)\e2eSoft\iVCam\iVCam.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                ivcam_exe = p
                break

    # Mở iVCam nếu tìm thấy
    if ivcam_exe:
        subprocess.Popen(ivcam_exe)
        print(f"✅ Đã mở iVCam: {ivcam_exe}")
        time.sleep(3)  # đợi iVCam khởi động xong
    else:
        from tkinter import messagebox as _mb
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        _mb.showwarning(
            "iVCam chưa được cài",
            "Không tìm thấy iVCam trên máy này!\n"
            "Vui lòng cài iVCam trước khi sử dụng.\n\n"
            "Tải tại: https://www.e2esoft.com/ivcam/"
        )
        _root.destroy()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    auto_open_ivcam()
    app = FireApp()
    app.mainloop()