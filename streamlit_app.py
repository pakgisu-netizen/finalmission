import os
import io
import json
import tempfile
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import streamlit as st
from torchvision import transforms

# ConvAutoencoder must be defined in train_autoencoder.py at repo root
from train_autoencoder import ConvAutoencoder


@st.cache_resource
def load_model(model_path: str, threshold_json: str, device: str):
    with open(threshold_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = meta["threshold"]

    device_t = torch.device(device)
    model = ConvAutoencoder(latent_dim=int(meta.get("latent_dim", 256))).to(device_t)
    model.load_state_dict(torch.load(model_path, map_location=device_t))
    model.eval()
    return model, float(threshold), device_t, int(meta.get("img_size", 256))


def infer_image(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        x = image_tensor.to(device).unsqueeze(0)
        recon = model(x)
        residual = (recon - x).pow(2).squeeze(0).cpu().numpy()
        residual_map = residual.mean(axis=0)
        mse_score = float(np.mean(residual_map))
    return mse_score, residual_map


def find_bboxes_from_residual(residual_map: np.ndarray, min_area: int = 200, threshold_ratio: float = 0.6) -> List[Tuple[int, int, int, int]]:
    """
    residual_map: 2D numpy array (H, W)
    returns list of (x1,y1,x2,y2) in residual_map coordinates
    """
    r = residual_map.copy()
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    mask = (r > threshold_ratio).astype("uint8") * 255
    if mask.sum() == 0:
        return []

    # find contours via OpenCV
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    # sort by area desc
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes


def draw_boxes_on_pil(img: Image.Image, boxes: List[Tuple[int, int, int, int]], scores: List[float] = None):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = "bbox"
        if scores is not None:
            label = f"{scores[i]:.4f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)
    return img


st.set_page_config(page_title="Autoencoder 이상치 검사", layout="wide")
st.title("Autoencoder 이상치 검사 (Streamlit)")
st.write("여러 이미지를 업로드하면 모델이 정상/불량을 판정하고, 불량이면 이미지에 박스를 표시합니다.")

# Sidebar settings
st.sidebar.header("설정")
model_path = st.sidebar.text_input("모델 파일 경로", value="model_out/ae_final.pth")
threshold_json = st.sidebar.text_input("threshold json 경로", value="model_out/threshold.json")
img_size = int(st.sidebar.number_input("이미지 입력 크기 (픽셀)", min_value=32, value=256, step=32))
device_options = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
device = st.sidebar.selectbox("디바이스", options=device_options)
min_area = st.sidebar.number_input("최소 bbox 면적", value=50)
thresh_ratio = st.sidebar.slider("Residual threshold ratio", min_value=0.0, max_value=1.0, value=0.7)

# Allow model/threshold upload via UI
model_upload = st.sidebar.file_uploader("모델 파일 업로드 (.pt/.pth)", type=["pt", "pth"], key="model_upload")
threshold_upload = st.sidebar.file_uploader("threshold.json 업로드 (선택)", type=["json"], key="thresh_upload")

if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

col1, col2 = st.columns([1, 2])
with col1:
    uploaded_files = st.file_uploader("이미지 선택 (여러 개 가능)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("모델 로드/재로딩"):
        try:
            device_t = torch.device(device)
            # If a model file uploaded via UI, use it
            if model_upload is not None:
                model_bytes = model_upload.read()
                # Try torch.load from bytes (state_dict or whole model)
                try:
                    buffer = io.BytesIO(model_bytes)
                    state = torch.load(buffer, map_location=device_t)
                    if isinstance(state, dict):
                        latent_dim = int(state["fc_mu.weight"].shape[0]) if "fc_mu.weight" in state else 256
                        model = ConvAutoencoder(latent_dim=latent_dim).to(device_t)
                        model.load_state_dict(state)
                    else:
                        # saved full model
                        model = state
                except Exception:
                    # Fallback: try torch.jit.load via temp file
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
                    try:
                        tmp.write(model_bytes)
                        tmp.flush()
                        tmp.close()
                        model = torch.jit.load(tmp.name, map_location=device_t)
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass

                # threshold: uploaded JSON takes precedence if provided
                if threshold_upload is not None:
                    thresh_meta = json.load(io.StringIO(threshold_upload.getvalue().decode('utf-8')))
                    threshold_val = float(thresh_meta.get("threshold", 0.0))
                    st.session_state["model_img_size"] = int(thresh_meta.get("img_size", img_size))
                else:
                    try:
                        with open(threshold_json, "r", encoding="utf-8") as f:
                            thresh_meta = json.load(f)
                        threshold_val = float(thresh_meta["threshold"])
                        st.session_state["model_img_size"] = int(thresh_meta.get("img_size", img_size))
                    except Exception as exc:
                        raise FileNotFoundError("threshold JSON을 업로드하거나 올바른 경로를 지정하세요.") from exc

            else:
                model, threshold_val, device_t, metadata_img_size = load_model(model_path, threshold_json, device)
                st.session_state["model_img_size"] = metadata_img_size

            st.session_state['model'] = model
            st.session_state['threshold'] = threshold_val
            st.session_state['device'] = device_t
            st.session_state['model_loaded'] = True
            st.success(f"모델 로드 완료. threshold={threshold_val}")
        except Exception as e:
            st.session_state['model_loaded'] = False
            st.error(f"모델 로드 실패: {e}")

    if not st.session_state['model_loaded']:
        st.info("먼저 모델을 로드한 후 이미지를 업로드하세요.")
    st.markdown("---")
    st.write("설정 요약:")
    st.write({
        "model_path": model_path,
        "threshold_json": threshold_json,
        "img_size": img_size,
        "device": device,
        "min_area": min_area,
        "thresh_ratio": thresh_ratio,
        "model_upload": bool(model_upload),
        "threshold_upload": bool(threshold_upload)
    })

with col2:
    if uploaded_files is None or len(uploaded_files) == 0:
        st.info("이미지를 업로드하면 결과가 이곳에 표시됩니다.")
    else:
        if not st.session_state['model_loaded']:
            st.warning("모델이 로드되어 있지 않습니다. 먼저 '모델 로드/재로딩' 버튼을 눌러주세요.")
        else:
            model = st.session_state['model']
            threshold_val = st.session_state['threshold']
            device_t = st.session_state['device']
            active_img_size = int(st.session_state.get("model_img_size", img_size))

            results = []
            for up in uploaded_files:
                try:
                    img = Image.open(up).convert("RGB")
                except Exception as e:
                    st.error(f"{up.name} 파일을 열 수 없습니다: {e}")
                    continue

                original_w, original_h = img.size
                transform = transforms.Compose([
                    transforms.Resize((active_img_size, active_img_size)),
                    transforms.ToTensor()
                ])
                x = transform(img)

                score, residual_map = infer_image(model, x, device_t)
                label = "defect" if score > threshold_val else "normal"
                boxes = []
                if label == "defect":
                    boxes_res = find_bboxes_from_residual(residual_map, min_area=min_area, threshold_ratio=thresh_ratio)
                    # scale boxes back to original image size
                    sx = original_w / float(active_img_size)
                    sy = original_h / float(active_img_size)
                    boxes = [[int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)] for (x1, y1, x2, y2) in boxes_res]

                # draw boxes
                img_copy = img.copy()
                if boxes:
                    img_copy = draw_boxes_on_pil(img_copy, boxes, scores=[score]*len(boxes))

                # display
                st.subheader(f"{up.name} — {label.upper()}")
                st.write(f"score: {score:.6f} — threshold: {threshold_val}")
                st.image(img_copy, use_column_width=True)

                results.append({
                    "filename": up.name,
                    "score": score,
                    "threshold": threshold_val,
                    "label": label,
                    "bboxes": boxes
                })

            # summary table
            if results:
                st.markdown("---")
                st.subheader("결과 요약")
                import pandas as pd
                df = pd.DataFrame([{
                    'filename': r['filename'], 'score': r['score'], 'threshold': r['threshold'], 'label': r['label'], 'bboxes': r['bboxes']
                } for r in results])
                st.dataframe(df)
