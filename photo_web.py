import streamlit as st
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from collections import Counter
import edge_tts
import asyncio
import threading
import pygame
import uuid
import time
from ultralytics import YOLO

# ============================================
#  STREAMLIT SETUP & CLEAN UI STYLING
# ============================================
st.set_page_config(page_title="Vision AI Assistant", layout="wide")

st.markdown("""
    <style>
    /* Background */
    .main { background-color: #f5f7f9; }
    
    /* Remove Button and style File Uploader */
    .stButton { display: none; }
    
    /* Result Card Styling */
    .status-box {
        padding: 30px;
        border-radius: 20px;
        background: #ffffff;
        border-top: 10px solid #2e7d32;
        box-shadow: 0 15px 35px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    
    .result-header {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 15px;
        letter-spacing: 0.5px;
    }
    
    .result-text {
        font-size: 1.3rem;
        line-height: 1.8;
        color: #2c3e50;
        font-family: 'Pyidaungsu', sans-serif;
    }

    /* Container Spacing */
    [data-testid="column"] {
        padding: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
#  CONFIGURATION (Logic မူရင်းအတိုင်း)
# ============================================
class Config:
    YOLO_MODEL_SIZE = "l"
    CONFIDENCE_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.45
    CLASS_FILTER = [0, 2, 3, 29, 40, 56, 58, 60, 62, 63, 74, 77]
    ROAD_THRESHOLD = 0.3
    ROAD_ALPHA = 0.4
    ROAD_COLOR = [255, 0, 0]
    USE_VERTICAL_WEIGHTING = True
    VERTICAL_WEIGHT_RANGE = (0.4, 1.0)
    METERS_PER_PIXEL = 0.02

    # မြန်မာလူမျိုးများ၏ ပျမ်းမျှခြေလှမ်း
    STEP_LENGTH_M = 0.65

    BOX_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    FONT_SCALE = 0.6
    BOX_THICKNESS = 2
    SHOW_CONFIDENCE = True
    ENHANCE_IMAGE = True
    CONTRAST_FACTOR = 1.2
    BRIGHTNESS_FACTOR = 1.1
    SHARPNESS_FACTOR = 1.1

    # မျက်မှန်အမြင့် (၅ ပေ ကျော်) အတွက် ညှိထားသော တန်ဖိုး
    MPP_BOTTOM = 0.025
    
    MPP_TOP = 0.12
    CORRIDOR_WIDTH_PX = 40
    OBSTACLE_LABELS = {"person", "car", "motorcycle", "bicycle", "bus", "truck"}
    OBSTACLE_BUFFER = 20

config = Config()

MM_LABELS = {"person": "လူ", "car": "ကား", "motorcycle": "မော်တော်ဆိုင်ကယ်", "suitcase": "ခရီးဆောင်အိတ်", "bottle": "ဘူး", "chair": "ထိုင်ခုံ", "potted plant": "အိုးစိုက်အပင်", "dining table": "စားပွဲ", "tv": "တီဗီ", "laptop": "လက်တော့ပ်", "book": "စာအုပ်", "cell phone": "ဖုန်း"}
MM_COUNTER = {"လူ": "ယောက်", "ကား": "စီး", "မော်တော်ဆိုင်ကယ်": "စီး", "ခရီးဆောင်အိတ်": "လုံး", "ဘူး": "ခု", "ထိုင်ခုံ": "လုံး", "အိုးစိုက်အပင်": "ပင်", "စားပွဲ": "လုံး", "တီဗီ": "လုံး", "လက်တော့ပ်": "လုံး", "စာအုပ်": "အုပ်", "ဖုန်း": "လုံး"}
MM_NUM = {1: "တစ်", 2: "နှစ်", 3: "သုံး", 4: "လေး", 5: "ငါး", 6: "ခြောက်", 7: "ခုနစ်", 8: "ရှစ်", 9: "ကိုး", 10: "တစ်ဆယ်"}

# (Original Logic Functions)
def preprocess_image(image):
    if config.ENHANCE_IMAGE:
        image = ImageEnhance.Contrast(image).enhance(config.CONTRAST_FACTOR)
        image = ImageEnhance.Brightness(image).enhance(config.BRIGHTNESS_FACTOR)
        image = ImageEnhance.Sharpness(image).enhance(config.SHARPNESS_FACTOR)
    return image

@st.cache_resource
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    yolo = YOLO(f"yolov8{config.YOLO_MODEL_SIZE}.pt")
    model_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_id)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    return yolo, processor, seg_model, device

def adaptive_mpp(y, H):
    t = 1.0 - (y / max(H - 1, 1))
    return config.MPP_BOTTOM * (1 - t) + config.MPP_TOP * t

def estimate_walk_distance(road_mask, detections):
    H, W = road_mask.shape
    center_col = W // 2
    half_w = config.CORRIDOR_WIDTH_PX // 2
    corridor_l, corridor_r = max(0, center_col - half_w), min(W, center_col + half_w)
    blocking = [d["box"] for d in detections if d["name_en"] in config.OBSTACLE_LABELS]
    meters = 0.0
    for y in range(H - 1, -1, -1):
        obstructed = any(y >= y1 and y <= y2 and not (x2 < corridor_l or x1 > corridor_r) for x1, y1, x2, y2 in blocking)
        if obstructed or road_mask[y, corridor_l:corridor_r].mean() < 0.6: break
        meters += adaptive_mpp(y, H)
    return int(meters / config.STEP_LENGTH_M)

def calculate_guidance(road_mask, detections):
    H, W = road_mask.shape
    center_x = W // 2
    safe_mask = road_mask.copy()
    b = config.OBSTACLE_BUFFER
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        safe_mask[max(0, y1-b):min(H, y2+b), max(0, x1-b):min(W, x2+b)] = False
    
    l, r = max(0, center_x-40), min(W, center_x+40)
    central_strip = safe_mask[int(H*0.5):, l:r]
    if central_strip.sum() > (central_strip.size * 0.5):
        return "path is clear, you can go straight forward", center_x
    
    left_space = safe_mask[int(H*0.5):, :center_x].sum()
    right_space = safe_mask[int(H*0.5):, center_x:].sum()
    if left_space > right_space and left_space > 5000:
        return "there is something at front, turn a little left", center_x - 100
    elif right_space > left_space and right_space > 5000:
        return "there is something at front, turn a little right", center_x + 100
    else: return "please stop", center_x

def create_roadmap(image_np, road_mask, detections, guidance, cx):
    output = image_np.copy()
    overlay = np.zeros_like(output)
    overlay[road_mask > 0] = config.ROAD_COLOR
    output = cv2.addWeighted(output, 1.0, overlay, config.ROAD_ALPHA, 0)
    H, W, _ = output.shape
    cv2.line(output, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
    cx_clamped = max(0, min(W - 1, cx))
    cv2.line(output, (cx_clamped, 0), (cx_clamped, H), (0, 255, 255), 2)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), config.BOX_COLORS[i % 4], 2)
        cv2.putText(output, det["label"], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(output, guidance, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return output

async def play_tts(text):
    file = f"tts_{uuid.uuid4().hex}.mp3"
    communicate = edge_tts.Communicate(text=text, voice="my-MM-NilarNeural")
    await communicate.save(file)
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): time.sleep(0.1)
    pygame.mixer.quit()
    if os.path.exists(file): os.remove(file)

def guidance_mm(text_en):
    return {"path is clear, you can go straight forward": "လမ်းရှင်းနေပါတယ်၊ တိုက်ရိုက်ရှေ့ကို သွားနိုင်ပါတယ်",
            "there is something at front, turn a little left": "ရှေ့တွင် တားဆီးမှုရှိနေပြီး၊ အနည်းငယ် ဘယ်ဘက်ကို လှည့်ပါ",
            "there is something at front, turn a little right": "ရှေ့တွင် တားဆီးမှုရှိနေပြီး၊ အနည်းငယ် ညာဘက်ကို လှည့်ပါ",
            "please stop": "ကျေးဇူးပြု၍ ရပ်ပါ"}.get(text_en, text_en)

# ============================================
#  MAIN INTERFACE (AUTO-PROCESS)
# ============================================
st.title(" AI-Based Road Analysis and Object Detection with Myanmar Audio Feedback")
yolo_model, seg_processor, seg_model, device = load_models()
st.markdown("#### Object Accuracy 52.9% | Road Segmentation mIoU 70% | Powered by YOLOv8 & SegFormer")
uploaded_file = st.file_uploader("စစ်ဆေးမည့် ဓာတ်ပုံကို ရွေးပါ...", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "ico"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.markdown("###  Original Photo")
        st.image(image_pil, use_container_width=True)

    # ဓာတ်ပုံရှိတာနဲ့ အလိုအလျောက် အောက်က code တွေ run မည်
    with st.spinner('AI က ခွဲခြမ်းစိတ်ဖြာနေသည်...'):
        processed_pil = preprocess_image(image_pil)
        image_np = np.array(processed_pil)
        
        # YOLO Detection
        res = yolo_model(processed_pil, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD, classes=config.CLASS_FILTER, verbose=False)[0]
        detections = []
        if res.boxes:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                name_en = yolo_model.names[int(box.cls[0])]
                detections.append({"box": (int(x1), int(y1), int(x2), int(y2)), "name_en": name_en, "name_mm": MM_LABELS.get(name_en, name_en), "label": name_en})

        # Segmentation
        inputs = seg_processor(images=processed_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = seg_model(**inputs).logits
        logits = F.interpolate(logits, size=processed_pil.size[::-1], mode="bilinear", align_corners=False)
        road_mask = (torch.softmax(logits, dim=1)[0, 0] >= config.ROAD_THRESHOLD).cpu().numpy()

        # Results
        guidance_en, cx = calculate_guidance(road_mask, detections)
        steps = estimate_walk_distance(road_mask, detections)
        roadmap = create_roadmap(image_np, road_mask, detections, guidance_en, cx)
        
        with col2:
            st.markdown("###  Analysis Result")
            st.image(roadmap, use_container_width=True)
        
        # Result Summary Box တွက်ချက်သည့်အပိုင်း
        counts = Counter([d["name_mm"] for d in detections])
        summary_parts = [f"{n} {MM_NUM.get(c, c)} {MM_COUNTER.get(n, 'ခု')}" for n, c in counts.items()]

        # --- Logic အသစ် စတင်ရန် ---
        if summary_parts:
            # Object ရှိလျှင် (ဥပမာ - လူ တစ်ယောက် တွေ့ရှိပါတယ်)
            object_msg = f"{'၊ '.join(summary_parts)} တွေ့ရှိပါတယ်။ "
        else:
            # Object မရှိလျှင်
            object_msg = "ရှေ့တွင် အတားအဆီး မရှိပါ။ "

        # ခြေလှမ်းတွက်ချက်မှု စာသား
        if steps > 0:
            step_msg = f"လမ်းကို ခန့်မှန်းခြေ ခြေလှမ်း {MM_NUM.get(steps, steps)} လှမ်းခန့် ဆက်လျှောက်နိုင်ပါတယ်။ "
        else:
            step_msg = "ရှေ့တည့်တည့်တွင် လမ်းပိတ်နေသဖြင့် ဆက်မသွားပါနှင့်။ "

        # Guidance (ဘယ်ကွေ့၊ ညာကွေ့)
        direction_msg = guidance_mm(guidance_en)

        # အားလုံးကို ပေါင်းလိုက်ခြင်း
        final_text = f"{object_msg}{step_msg}{direction_msg}"

        # --- Logic အသစ် ပြီးဆုံး ---
        st.markdown(f"""
            <div class="status-box">
                <div class="result-header">🔍 Analysis Summary</div>
                <p class="result-text">{final_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Voice (Run once)
        if 'last_processed' not in st.session_state or st.session_state.last_processed != uploaded_file.name:
            threading.Thread(target=lambda: asyncio.run(play_tts(final_text))).start()
            st.session_state.last_processed = uploaded_file.name
else:
    st.info("ဓာတ်ပုံတင်လိုက်တာနဲ့ AI က အလိုအလျောက် စစ်ဆေးပေးမှာဖြစ်ပါတယ်။")