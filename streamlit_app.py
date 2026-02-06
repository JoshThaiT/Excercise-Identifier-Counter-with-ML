import streamlit as st
import tempfile
from utils import features_to_extract, count_pullup, count_pushup, count_situp, rescale_frame,smooth_prediction
from ultralytics import YOLO
import numpy as np
from xgboost import XGBClassifier
import pickle
import os
import imageio
from PIL import Image, ImageDraw, ImageFont
from collections import deque, Counter

st.title("Exercise Tracker")
st.write("Upload either pull up, push up, or sit up and this will identify it.")

uploaded_file = st.file_uploader("Upload video please", type=["mp4"])

@st.cache_resource
def load_models():
    # Load XGB model
    with open("exercise_identifier_xgb.pkl", "rb") as f:
        xgb = pickle.load(f)

    # Load YOLO safely
    yolo_path = os.path.join(os.getcwd(), "yolo26n-pose.pt")
    yolo = YOLO(yolo_path)

    return xgb, yolo

loaded_xgb, model = load_models()

states = {"push_up": "UP", "pull_up": "UP", "sit_up": "UP"}
counters = {"push_up": 0, "pull_up": 0, "sit_up": 0}

buffer = []
window_size = 30
prediction = "None Detected"
history = deque(maxlen=5)

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


if uploaded_file:
    st.write("Processing video...")

    # Save uploaded video
    input_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_tfile.write(uploaded_file.read())
    input_tfile.close()

    # Output temp file
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # Read video with imageio
    reader = imageio.get_reader(input_tfile.name, 'ffmpeg')
    meta = reader.get_meta_data()
    fps = meta['fps']
    frame_w = meta['size'][0]
    frame_h = meta['size'][1]
    base_dim = min(frame_w, frame_h)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    label_font_size = max(20, base_dim // 20)
    reps_font_size = max(15, base_dim // 20)
    font_label = ImageFont.truetype(font_path, label_font_size)
    font_reps = ImageFont.truetype(font_path, reps_font_size)

    writer = imageio.get_writer(output_path, fps=fps)

    frame_idx = 0
    total_frames = meta.get("nframes", 300)
    progress = st.progress(0)

    for frame in reader:
        frame_idx += 1
        frame_np = np.array(frame)

        # YOLO inference
        results = model(frame_np, conf=0.2)
        r = results[0]

        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            kpts = r.keypoints.xy[0].cpu().numpy()
            h_, w_ = frame.shape[:2]
            kpts[:, 0] /= w_
            kpts[:, 1] /= h_

            feat = features_to_extract(kpts)
            buffer.append(feat)

            if len(buffer) > window_size:
                buffer.pop(0)

            if len(buffer) == window_size:
                X_input = np.array(buffer).reshape(1, -1)
                raw_pred = loaded_xgb.predict(X_input)[0]
                prediction = smooth_prediction(raw_pred, history)

                if prediction == 1:
                    angle = buffer[-1][1]
                    states["push_up"], counters["push_up"] = count_pushup(
                        angle, states["push_up"], counters["push_up"]
                    )
                elif prediction == 0:
                    angle = buffer[-1][1]
                    states["pull_up"], counters["pull_up"] = count_pullup(
                        angle, states["pull_up"], counters["pull_up"]
                    )
                elif prediction == 2:
                    angle = buffer[-1][5]
                    states["sit_up"], counters["sit_up"] = count_situp(
                        angle, states["sit_up"], counters["sit_up"]
                    )

        # Annotate frame using PIL (no libGL needed)
        pil_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_frame)

        label_map = {0: "Pull Up", 1: "Push Up", 2: "Sit Up"}
        label = label_map.get(prediction, "Detecting...")
        reps = counters.get(label.lower().replace(" ", "_"), 0)

        draw.text((20, 20), f"{label}", font=font_label, fill=(255, 255, 255))
        draw.text((20, 80), f"Reps: {reps}", font=font_reps, fill=(0, 255, 0))

        # Convert back to numpy array for writing
        writer.append_data(np.array(pil_frame))

        progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

    reader.close()
    writer.close()

    st.success("Done!")

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes)