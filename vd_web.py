import streamlit as st
import cv2
import tempfile
import torch
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from collections import Counter
import edge_tts
import asyncio
import threading
import pygame
import uuid
import time
import os

# Set Page Config MUST be the first Streamlit command
st.set_page_config(
    page_title="Vision AI Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Beauty UI"
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    .css-1v0mbdj.e115fcil1 {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
#  CONFIGURATION & MODELS (README Strategy)
# ============================================
class Config:
    YOLO_MODEL = "yolov8m.pt"
    SEG_MODEL_ID = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Accuracy & Smoothing
    CONFIDENCE_THRESHOLD = 0.50
    ROAD_THRESHOLD = 0.3
    AUDIO_INTERVAL = 6 
    SKIP_FRAMES = 4 # FPS smooth ဖြစ်စေရန် frames ကျော်ဖတ်မည်
    # Performance tuning (keeps features but reduces per-frame cost)
    TARGET_W = 480
    TARGET_H = 360
    SEG_EVERY_N_FRAMES = 2
    
    # Distance Logic
    STEP_LENGTH_M = 0.65
    MPP_BOTTOM = 0.025
    MPP_TOP = 0.12
    CORRIDOR_WIDTH_PX = 40
    CLASS_FILTER = [0, 2, 3, 5, 7] 

MM_LABELS = {"person": "လူ", "car": "ကား", "motorcycle": "မော်တော်ဆိုင်ကယ်", "bus": "ဘတ်စ်ကား", "truck": "ကုန်တင်ကား"}
MM_COUNTER = {"လူ": "ယောက်", "ကား": "စီး", "မော်တော်ဆိုင်ကယ်": "စီး", "ဘတ်စ်ကား": "စီး", "ကုန်တင်ကား": "စီး"}
MM_NUM = {i: n for i, n in enumerate(["သုည", "တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ရှစ်", "ကိုး", "တစ်ဆယ်"])}

@st.cache_resource
def load_models():
    yolo = YOLO(Config.YOLO_MODEL).to(Config.DEVICE)
    processor = SegformerImageProcessor.from_pretrained(Config.SEG_MODEL_ID)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(Config.SEG_MODEL_ID).to(Config.DEVICE)
    yolo.model.eval()
    seg_model.eval()
    return yolo, processor, seg_model

yolo_model, seg_processor, seg_model = load_models()

# ============================================
#  FIXED AUDIO LOGIC (From Photo Version)
# ============================================

# Initialize mixer once to avoid overhead
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Audio init error: {e}")

def play_tts_sync(text):
    """photo_web.py ထဲကအတိုင်း pygame သုံးပြီး အသံဖွင့်ပေးခြင်း"""
    file = f"tts_{uuid.uuid4().hex}.mp3"
    async def generate():
        communicate = edge_tts.Communicate(text=text, voice="my-MM-NilarNeural")
        await communicate.save(file)
    
    try:
        # Create a new event loop for this thread to avoid conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate())
        loop.close()
        
        # Ensure mixer is ready
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): 
            time.sleep(0.1)
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        # Do not quit mixer, just cleanup file
        if os.path.exists(file): 
            try:
                os.remove(file)
            except:
                pass

def trigger_audio(text):
    threading.Thread(target=play_tts_sync, args=(text,), daemon=True).start()

# ============================================
#  MAIN LOOP (Optimized for Video)
# ============================================
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🎬 AI-Based Road Analysis and Object Detection with Myanmar Audio Feedback</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>Real-time object detection and road segmentation for visually impaired assistance</p>", unsafe_allow_html=True)
st.divider()

# Layout: Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    with st.expander("🛠️ Model Settings", expanded=True):
        Config.CONFIDENCE_THRESHOLD = st.slider("Detection Confidence", 0.1, 1.0, Config.CONFIDENCE_THRESHOLD, 0.05)
        Config.ROAD_THRESHOLD = st.slider("Road Threshold", 0.1, 1.0, Config.ROAD_THRESHOLD, 0.05)
    
    with st.expander("🔊 Audio Settings", expanded=True):
        Config.AUDIO_INTERVAL = st.slider("Audio Interval (s)", 2, 10, Config.AUDIO_INTERVAL)
    
    st.markdown("---")
    st.caption("Powered by YOLOv8 & SegFormer")

# Layout: Main Content
col1, col2 = st.columns([0.7, 0.3], gap="large")

with col2:
    st.markdown("### 📥 Upload & Status")
    uploaded_video = st.file_uploader("စစ်ဆေးမည့် ဗီဒီယိုကို ရွေးပါ...", type=["mp4", "mov"], help="Select a video file to analyze")
    
    st.markdown("### 📊 Live Analysis")
    status_container = st.container()
    
    stop_button = st.button("🛑 Stop Processing", type="primary")

with col1:
    st.markdown("### 📹 Video Feed")
    video_placeholder = st.empty()
    # Default placeholder image or text
    video_placeholder.info("စစ်ဆေးမှု စတင်နိုင်ရန် ဗီဒီယိုကို တင်ပါ")

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    vf = cv2.VideoCapture(tfile.name)
    
    last_audio_time = 0
    frame_count = 0
    last_road_mask = None
    
    with status_container:
        st.info("🔄 Initializing AI models...")

    while vf.isOpened():
        if stop_button:
            with status_container:
                st.warning("⚠️ Processing stopped by user.")
            break
            
        ret, frame = vf.read()
        if not ret: break

        frame_count += 1
        if frame_count % Config.SKIP_FRAMES != 0: continue

        # Image Enhancement (lighter)
        proc_frame = cv2.resize(frame, (Config.TARGET_W, Config.TARGET_H))
        img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Inference (use inference_mode for less overhead)
        with torch.inference_mode():
            res = yolo_model(pil_img, device=Config.DEVICE, conf=Config.CONFIDENCE_THRESHOLD, classes=Config.CLASS_FILTER, verbose=False)[0]
            if frame_count % Config.SEG_EVERY_N_FRAMES == 0 or last_road_mask is None:
                inputs = seg_processor(images=pil_img, return_tensors="pt").to(Config.DEVICE)
                logits = seg_model(**inputs).logits
                logits = F.interpolate(logits, size=(Config.TARGET_H, Config.TARGET_W), mode="bilinear", align_corners=False)
                road_mask = (torch.softmax(logits, dim=1)[0, 0] >= Config.ROAD_THRESHOLD).cpu().numpy()
                last_road_mask = road_mask
            else:
                road_mask = last_road_mask

        # Detection logic
        width = Config.TARGET_W
        mid_x = width // 2
        corridor_half = int(width * 0.3 // 2) # အလယ် 30% ကို corridor အဖြစ်ယူ
        left_bound = mid_x - corridor_half
        right_bound = mid_x + corridor_half
        
        is_blocked_ahead = False
        current_dets = [] # Detection list ကို loop အစမှာ initialize လုပ်ခြင်း

        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name_en = yolo_model.names[int(box.cls[0])]
            current_dets.append({"box": (x1, y1, x2, y2), "name_en": name_en})
            
            # အရှေ့တည့်တည့်မှာ ရှိမရှိ စစ်ဆေးခြင်း
            obj_center_x = (x1 + x2) / 2
            if left_bound < obj_center_x < right_bound:
                is_blocked_ahead = True

            # Draw (Original display logic)
            cv2.rectangle(proc_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(proc_frame, name_en, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- AUDIO GUIDANCE (Original MM logic + New Direction) ---
        if time.time() - last_audio_time > Config.AUDIO_INTERVAL:
            # မူလပါပြီးသား MM Labeling Logic
            counts = Counter([MM_LABELS.get(d['name_en'], d['name_en']) for d in current_dets])
            summary = "၊ ".join([f"{n} {MM_NUM.get(c, c)} {MM_COUNTER.get(n, 'ခု')}" for n, c in counts.items()])
            
            # လမ်းကြောင်း ခွဲခြမ်းစိတ်ဖြာခြင်း
            if road_mask is not None:
                left_path = np.sum(road_mask[:, :left_bound])
                right_path = np.sum(road_mask[:, right_bound:])
                clear_side = "ဘယ်ဘက်က" if left_path > right_path else "ညာဘက်က"
            else:
                clear_side = ""

            # Final Voice Message Logic
            if is_blocked_ahead:
                final_text = f"ရှေ့တည့်တည့်မှာ {summary if summary else 'အတားအဆီး'} ရှိနေလို့ ခဏရပ်ပါ။ {clear_side} ရှောင်သွားပါ။"
            elif summary:
                final_text = f"{summary} တွေ့ရှိပါတယ်။ လမ်းဆက်လျှောက်နိုင်ပါတယ်။"
            else:
                final_text = "လမ်းရှင်းပါတယ်။ တည့်တည့် ဆက်သွားနိုင်ပါတယ်။"

            # Update Status & Trigger Audio
            with status_container:
                if is_blocked_ahead: st.warning(f"**🔊 AI Says:** {final_text}")
                else: st.success(f"**🔊 AI Says:** {final_text}")

            trigger_audio(final_text)
            last_audio_time = time.time()

        # Visual Overlay (Original Road display)
        overlay = proc_frame.copy()
        if road_mask is not None:
            overlay[road_mask] = [255, 0, 0]
        proc_frame = cv2.addWeighted(proc_frame, 0.7, overlay, 0.3, 0)
        
        # Corridor Guide Lines (DEBUG အတွက် မျဉ်းနီလေး ၂ ချောင်း ထည့်ပေးထားပါတယ်)
        cv2.line(proc_frame, (left_bound, 0), (left_bound, Config.TARGET_H), (0, 0, 255), 1)
        cv2.line(proc_frame, (right_bound, 0), (right_bound, Config.TARGET_H), (0, 0, 255), 1)

        # Display Video
        video_placeholder.image(proc_frame, channels="BGR", use_container_width=True)

    vf.release()
    if os.path.exists(tfile.name): os.remove(tfile.name)