import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "crop_disease_model.h5")))
LABEL_MAP_PATH = Path(os.getenv("LABEL_MAP_PATH", str(BASE_DIR / "label_map.json")))
HISTORY_PATH = BASE_DIR / "prediction_history.json"


@st.cache_resource
def load_model_and_labels():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Train a model first with `python train_model.py` or set MODEL_PATH."
        )
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Label map not found: {LABEL_MAP_PATH}. Ensure the generated label map is available."
        )

    model = tf.keras.models.load_model(str(MODEL_PATH))
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
        raw_label_map = json.load(file)
    label_map = {int(v): k for k, v in raw_label_map.items()}
    return model, label_map


def summarize_history(history):
    if not history:
        return {"disease_counts": {}, "severity_counts": {"Healthy": 0, "Low risk": 0, "Moderate risk": 0, "Severe risk": 0}}

    disease_counts = {}
    severity_counts = {"Healthy": 0, "Low risk": 0, "Moderate risk": 0, "Severe risk": 0}

    for item in history:
        disease = item["predicted_class"]
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        severity_counts[item["severity"]] = severity_counts.get(item["severity"], 0) + 1

    return {"disease_counts": disease_counts, "severity_counts": severity_counts}


def load_history():
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        try:
            history = json.load(file)
        except json.JSONDecodeError:
            return []
    return history if isinstance(history, list) else []


def save_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def get_severity(predicted_class, confidence):
    class_label = predicted_class.lower().replace("__", "_").replace("-", "_")
    if "healthy" in class_label:
        return "Healthy"

    if any(token in class_label for token in ["late_blight", "yellowleaf", "mosaic", "spider_mites", "target_spot"]):
        if confidence >= 70:
            return "Severe risk"
        return "Moderate risk"

    if any(token in class_label for token in ["early_blight", "bacterial", "leaf_mold", "septoria"]):
        if confidence >= 60:
            return "Moderate risk"
        return "Low risk"

    if confidence >= 65:
        return "Moderate risk"
    return "Low risk"


def prepare_prediction_result(filename, predicted_class, confidence):
    risk_level = get_severity(predicted_class, float(confidence))
    return {
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 2),
        "severity": risk_level,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def export_history_csv(history):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "filename", "predicted_class", "confidence", "severity"])
    writer.writeheader()
    writer.writerows(history)
    return output.getvalue()


def parse_history_timestamp(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return datetime.min


def filter_history(history, selected_disease, selected_severity, start_date=None, end_date=None):
    filtered = history
    if selected_disease != "All":
        filtered = [item for item in filtered if item["predicted_class"] == selected_disease]
    if selected_severity != "All":
        filtered = [item for item in filtered if item["severity"] == selected_severity]
    if start_date is not None:
        filtered = [item for item in filtered if parse_history_timestamp(item["timestamp"]).date() >= start_date]
    if end_date is not None:
        filtered = [item for item in filtered if parse_history_timestamp(item["timestamp"]).date() <= end_date]
    return filtered


def get_unique_values(history, key):
    values = sorted({item[key] for item in history})
    return ["All", *values]


st.set_page_config(page_title="Crop Disease Detector", layout="centered")
st.title("🌾 Crop Leaf Disease Detection")
st.write("Upload one or more leaf images or use your camera for a quick field check.")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = load_history()

try:
    model, label_map = load_model_and_labels()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Once you have trained the model, run the app again or mount the model file into the deployment environment.")
    st.stop()

filters = st.sidebar.expander("Filter history")
all_diseases = get_unique_values(st.session_state.prediction_history, "predicted_class")
all_severities = get_unique_values(st.session_state.prediction_history, "severity")
selected_disease = filters.selectbox("Disease", all_diseases)
selected_severity = filters.selectbox("Severity", all_severities)
if st.session_state.prediction_history:
    available_dates = [parse_history_timestamp(item["timestamp"]).date() for item in st.session_state.prediction_history]
    min_date = min(available_dates)
    max_date = max(available_dates)
    start_date, end_date = filters.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(start_date, tuple):
        start_date, end_date = start_date
else:
    start_date = end_date = None

filtered_history = filter_history(
    st.session_state.prediction_history,
    selected_disease,
    selected_severity,
    start_date,
    end_date,
)

summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("Total scans", len(filtered_history))
with summary_col2:
    if filtered_history:
        avg_conf = sum(item["confidence"] for item in filtered_history) / len(filtered_history)
    else:
        avg_conf = 0
    st.metric("Avg confidence", f"{avg_conf:.2f}%")
with summary_col3:
    if filtered_history:
        disease_summary = summarize_history(filtered_history)
        top_label = max(disease_summary["disease_counts"], key=disease_summary["disease_counts"].get)
    else:
        top_label = "None"
    st.metric("Most common", top_label)

history_summary = summarize_history(filtered_history)
if filtered_history:
    st.subheader("Prediction dashboard")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        disease_df = {"Disease": list(history_summary["disease_counts"].keys()), "Count": list(history_summary["disease_counts"].values())}
        st.bar_chart(disease_df, x="Disease", y="Count")
    with chart_col2:
        severity_df = {"Severity": list(history_summary["severity_counts"].keys()), "Count": list(history_summary["severity_counts"].values())}
        st.bar_chart(severity_df, x="Severity", y="Count")

    st.subheader("Filtered history")
    st.dataframe(
        [{
            "Timestamp": item["timestamp"],
            "Filename": item["filename"],
            "Disease": item["predicted_class"],
            "Confidence": f"{item['confidence']:.2f}%",
            "Severity": item["severity"],
        } for item in filtered_history],
        use_container_width=True,
    )

left_col, right_col = st.columns(2)
with left_col:
    uploaded_files = st.file_uploader(
        "Choose one or more leaf images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
with right_col:
    camera_photo = st.camera_input("Or take a photo with your camera")

input_images = []
if uploaded_files:
    input_images.extend(uploaded_files)
if camera_photo is not None:
    input_images.append(camera_photo)

if input_images:
    from treatments import treatment_suggestions

    for uploaded_file in input_images:
        image = Image.open(uploaded_file).convert("RGB")
        image_for_display = image.copy()
        source_name = getattr(uploaded_file, "name", "camera_capture.png")

        image = image.resize((256, 256))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_class = label_map.get(predicted_index, str(predicted_index))
        confidence = float(np.max(prediction)) * 100
        severity = get_severity(predicted_class, confidence)

        result = prepare_prediction_result(source_name, predicted_class, confidence)
        st.session_state.prediction_history.insert(0, result)
        st.session_state.prediction_history = st.session_state.prediction_history[:20]
        save_history(st.session_state.prediction_history)

        suggestion = treatment_suggestions.get(predicted_class)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_for_display, caption=result["filename"], use_column_width=True)
        with col2:
            st.success(f"🧠 Predicted: **{predicted_class}** ({confidence:.2f}% confidence)")
            st.caption(f"Assessment: {severity}")
            if suggestion:
                st.info(f"💊 Treatment Suggestion:\n{suggestion}")
            else:
                st.info("💊 No treatment guidance is available for this class yet.")

st.sidebar.subheader("Recent predictions")
if st.session_state.prediction_history:
    for item in st.session_state.prediction_history[:10]:
        st.sidebar.markdown(
            f"- **{item['filename']}**: {item['predicted_class']} ({item['confidence']:.2f}%, {item['severity']})"
        )
    csv_data = export_history_csv(filtered_history or st.session_state.prediction_history)
    st.sidebar.download_button(
        label="Download CSV report",
        data=csv_data,
        file_name="crop_disease_predictions.csv",
        mime="text/csv",
    )
    if st.sidebar.button("Clear history"):
        st.session_state.prediction_history = []
        save_history([])
        st.rerun()
else:
    st.sidebar.write("No predictions yet.")
