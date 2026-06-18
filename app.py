import streamlit as st
import cv2
import time
import json
import os
from ultralytics import YOLO
from google import genai
from dotenv import load_dotenv
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Video Auditor", page_icon="👁️", layout="wide")

# --- INITIALIZATION & CACHING ---
@st.cache_resource
def load_yolo_model():
    """Load the YOLO model once and cache it in memory."""
    return YOLO("yolov8n.pt")

def init_gemini(api_key):
    """Initialize the Gemini client using the provided API key."""
    if not api_key:
        return None
    os.environ["GEMINI_API_KEY"] = api_key
    return genai.Client()

# Load environment variables
load_dotenv()
default_api_key = os.environ.get("GEMINI_API_KEY", "")

# --- SIDEBAR: SETTINGS ---
st.sidebar.title("⚙️ Auditor Settings")
api_key_input = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
video_source = st.sidebar.selectbox("Video Source", ["Webcam (0)", "Upload Video File"])
reporting_interval = st.sidebar.slider("Reporting Interval (Seconds)", min_value=5, max_value=60, value=10)

# Session state controls
if "run_auditor" not in st.session_state:
    st.session_state.run_auditor = False

def start_auditor():
    st.session_state.run_auditor = True

def stop_auditor():
    st.session_state.run_auditor = False

col1, col2 = st.sidebar.columns(2)
col1.button("▶️ Start", on_click=start_auditor, use_container_width=True)
col2.button("⏹️ Stop", on_click=stop_auditor, use_container_width=True)

# --- GEMINI REPORTING FUNCTION ---
def generate_gemini_audit(client, log_data):
    """Generates the audit report and returns the text."""
    prompt = f"""
    You are an AI Security and Operations Auditor.
    Review the following JSON logs spanning the last {reporting_interval} seconds.
    
    Tasks:
    1. Summarize overall activity and crowd trends.
    2. Flag operational anomalies or security risks.
    3. Keep it brief, professional, and actionable.

    Telemetry Log Data:
    {json.dumps(log_data, indent=2)}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"**Error generating report:** {e}"

# --- MAIN DASHBOARD LAYOUT ---
st.title("👁️ Multi-Stream Multimodal Video Auditor")
st.markdown("Real-time local Edge AI object detection paired with high-level cloud LLM reasoning.")

# Create two columns: Left for Video, Right for AI Reports
video_col, report_col = st.columns([0.6, 0.4])

with video_col:
    st.subheader("Live Edge AI Stream")
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()

with report_col:
    st.subheader("Cloud Intelligence Reports")
    report_placeholder = st.empty()

# --- MAIN EXECUTION LOOP ---
if st.session_state.run_auditor:
    # Validation
    if not api_key_input:
        st.warning("⚠️ Please enter a Gemini API Key in the sidebar to start.")
        st.stop()

    gemini_client = init_gemini(api_key_input)
    yolo_model = load_yolo_model()

    # Handle Video Source
    source = 0 if video_source == "Webcam (0)" else None

    if video_source == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload an MP4", type=['mp4', 'mov', 'avi'])

        if uploaded_file is not None:
            # Use tempfile to create a robust, system-safe temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            source = tfile.name # Pass the safe temp file path to OpenCV
        else:
            st.warning("⚠️ Please upload a video file in the sidebar to continue.")
            st.stop()

    cap = cv2.VideoCapture(source)

    event_logs = []
    last_report_time = time.time()

    # Pre-populate the report column so it isn't blank
    report_placeholder.info("System initializing... waiting for first data batch.")

    while cap.isOpened() and st.session_state.run_auditor:
        success, frame = cap.read()
        if not success:
            st.warning("Video stream ended.")
            break

        # 1. Local YOLO Inference
        results = yolo_model.track(frame, classes=[0], persist=True, verbose=False)
        current_count = len(results[0].boxes) if results[0].boxes is not None else 0

        # 2. Update Video UI
        # OpenCV uses BGR, Streamlit expects RGB
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
        metrics_placeholder.metric(label="Current People Count", value=current_count)

        # 3. Accumulate Logs
        current_time = time.time()
        event_logs.append({
            "timestamp": time.strftime('%H:%M:%S', time.localtime(current_time)),
            "detected_count": current_count
        })

        # 4. Trigger Gemini Report
        if current_time - last_report_time >= reporting_interval:
            with report_col:
                with st.spinner("Gemini is analyzing the telemetry..."):
                    report_text = generate_gemini_audit(gemini_client, event_logs)

                    # Update the report UI
                    report_placeholder.markdown(f"### Audit generated at {time.strftime('%H:%M:%S')}")
                    report_placeholder.info(report_text)

            # Reset buffers
            event_logs = []
            last_report_time = current_time

    cap.release()