AI-Based Road Analysis and Object Detection with Myanmar Audio Feedback
 An advanced assistive technology developed to empower visually impaired individuals through real-time environmental analysis. This project utilizes SOTA (State-of-the-Art) Deep Learning models to perform semantic road segmentation and object detection, providing intuitive navigation guidance via Myanmar voice feedback.

1. Unified Model Strategy & Core Architecture
 To maximize efficiency, we have transitioned from a multi-tier system to a Unified Engine Strategy. This approach ensures that the system maintains high performance across all video-based inputs while providing professional-grade accuracy for static images.

    Feature Module, Model Complexity, Processing Focus, Inference Speed
    Static Image Mode, SegFormer-b5 + YOLOv8l, Maximum Precision, High Accuracy
    Video & Live Mode, SegFormer-b3 + YOLOv8m, Unified Engine, Smooth real-time navigation

2. Dynamic Performance Tuning (Adaptive Control)
 We understand that hardware capabilities vary between users. To address this, we have implemented a Customizable Performance Slider in the sidebar. This allows users to manually balance the "Accuracy vs. Speed" trade-off:

 - Adjustment Range: The slider operates on a scale from 0.1 to 1.00.
 - The 0.20 Confidence Threshold: We have optimized the system to report objects with at least 20% certainty. This ensures high mAP (mean  
   Average Precision) by balancing detection sensitivity and processing speed.
 - Optimization Logic:
        Higher Values (Up to 1.00): These settings increase the density of the analysis, leading to higher detection stability and precision. It is recommended for users with dedicated GPUs.
        Lower Values (Down to 0.1): These settings reduce computational overhead, ensuring zero latency and high frame rates on older or mobile-based hardware.

3. Comprehensive Project Methodology
Our technical pipeline is designed with a "Safety First" approach, consisting of three critical stages:

    Phase A: Image Acquisition & Preprocessing
        To handle unpredictable outdoor lighting, the system enhances every frame before analysis:
            Dynamic Contrast Adjustment: Ensures the AI can "see" clearly in low-light shadows or overexposed sunlight.
            Edge Sharpening: Refines the boundaries of objects, making it easier for the models to distinguish between a pedestrian and the background.

    Phase B: Dual-Engine Simultaneous Analysis
        The system utilizes two powerful AI engines working in parallel:
            Object Detection (YOLOv8m): Specifically trained to identify and track people, cars, and motorcycles. It features a Class Filter to ignore static, non-threatening objects, focusing only on what matters for navigation.
            Semantic Road Segmentation (SegFormer-b3): This model classifies every pixel to identify "Walkable Surface" versus "Obstacles," creating a safety map for the user.

    Phase C: Post-Processing & Navigation Intelligence
        Raw spatial data is transformed into intuitive navigation cues using:       
            Morphological Noise Reduction: Cleans up the AI's output to ensure the road path is solid and reliable.
            Vertical Weighting Logic: The system prioritizes objects at the bottom of the screen, as these represent the most immediate threats to the user.
            Adaptive MPP (Meters Per Pixel): A smart logic that converts digital pixels into "Real-World Human Steps," providing a realistic sense of distance.

4. Intelligent Myanmar Voice Guidance
 Navigation instructions are delivered through a customized Myanmar Text-to-Speech (Edge-TTS) engine. The system provides structured audio summaries to keep the user informed without causing sensory overload:
 - Detection Summary: "လူ တစ်ယောက် နှင့် ကား နှစ်စီး တွေ့ရှိပါတယ်။" (One person and two cars detected.)
 - Distance Estimate: "ခြေလှမ်း တစ်ဆယ် လှမ်းခန့် ဆက်လျှောက်နိုင်ပါတယ်။" (You can walk approximately 10 steps.)
 - Directional Cues: Precise instructions such as "Go Straight," "Turn Slightly Left," or "Stop."

5. System Limitations
 - Extreme Environmental Conditions: Our AI models depend on clear visual data to work effectively. In situations like heavy rain, thick fog, or near-total darkness, the system's accuracy may decrease because the camera cannot capture the necessary details for YOLOv8 and SegFormer to process correctly.
 - Navigation Scope (Roads & Flat Paths Only): The current system is specifically designed and optimized to identify flat road surfaces and outdoor pedestrian paths.
 - Staircase Detection: In its current version, the AI does not have the capability to distinguish stairs, steep inclines, or significant changes in elevation.
 - User Safety Caution: Because the models cannot yet identify vertical obstacles like steps or high curbs, users must exercise extra caution when approaching staircases or platform edges. 

6. Technology Specifications
Programming: Python 3.8+
Deep Learning Frameworks: PyTorch, Ultralytics, Hugging Face
Deep Learning Models: YOLOv8 (Medium & Large), SegFormer (B3 & B5 finetuned on Cityscapes)
UI Framework: Streamlit (Web-based Interface)
Computer Vision: OpenCV (Video processing & Mask rendering)
TTS Engine: Microsoft Edge-TTS (Myanmar Neural Voice)

7. How to run project files?
python -m streamlit run photo_web.py
python -m streamlit run vd_web.py
python -m streamlit run live_web.py

Academic Institution: University of Computer Studies (Thaton)
Academic Year: 2025 – 2026 (Fourth Year, Semester VII)
Developed by: Mg Thet Min Ko Ko & Group Members