"""
SafeCCTV - Real-Time CCTV Fall Monitoring & Instant Alarm Dashboard
Supports Laptop Webcams (Device 0, 1), Network CCTV Cameras (RTSP Streams),
Instant Siren Alarms (Windows Beep + Browser Audio), Acknowledge/Mute/Reset controls,
Incident Evidence Archiving, and an optional Uploaded Video Analysis Mode.
"""

import os
import io
import cv2
import wave
import math
import struct
import base64
import tempfile
import pandas as pd
import streamlit as st
from datetime import datetime
from camera_stream import CameraStreamManager, sanitize_camera_source
from mock_detector import get_detector, AnalysisResult

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeCCTV - Real-Time Fall Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generate in-memory WAV audio siren for browser playback
@st.cache_data
def get_siren_audio_base64() -> str:
    """Generates a 1.2-second alternating two-tone siren (950Hz & 650Hz) as a base64 WAV."""
    sample_rate = 22050
    duration = 1.2
    n_samples = int(sample_rate * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            freq = 950 if (int(t * 4) % 2 == 0) else 650
            sample = int(18000 * math.sin(2 * math.pi * freq * t))
            frames.extend(struct.pack("<h", sample))
        wav.writeframes(frames)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_browser_siren(active: bool):
    """Renders HTML5 audio tag for browser alarm playback."""
    if not active:
        return
    siren_b64 = get_siren_audio_base64()
    audio_html = f"""
    <audio autoplay loop id="cctv-siren">
        <source src="data:audio/wav;base64,{siren_b64}" type="audio/wav">
    </audio>
    <div style="font-size:0.8rem; color:#fca5a5; margin-top:4px;">
        🔊 <i>Browser siren active. If sound is muted by browser autoplay policy, click anywhere on the page to unmute.</i>
    </div>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-lbl {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Flashing Alarm Banner */
    .alarm-banner {
        background: linear-gradient(90deg, #991b1b, #dc2626);
        border: 2px solid #f87171;
        border-radius: 8px;
        padding: 18px;
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.7);
        animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
        from { opacity: 0.85; }
        to { opacity: 1.0; }
    }
    .status-live {
        background: #064e3b;
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        font-weight: 700;
        margin-bottom: 16px;
    }
    .status-standby {
        background: #1f2937;
        color: #9ca3af;
        border: 1px solid #374151;
        padding: 10px 14px;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SINGLETON INSTANCE
# -----------------------------------------------------------------------------
stream_manager = CameraStreamManager.get_instance()

# -----------------------------------------------------------------------------
# MAIN DASHBOARD TABS
# -----------------------------------------------------------------------------
tab_live, tab_upload = st.tabs([
    "🔴 Live CCTV & Webcam Monitoring",
    "📁 Uploaded Video Analysis Mode (Testing)"
])

# =============================================================================
# TAB 1: LIVE CCTV & WEBCAM MONITORING
# =============================================================================
with tab_live:
    # Sidebar Controls for Tab 1
    with st.sidebar:
        st.markdown("### 🛡️ **SafeCCTV Controls**")
        st.caption("Real-Time CCTV Fall Monitoring & Instant Alarm")
        st.divider()

        st.markdown("#### 📹 **Camera Input Source**")
        st.info("💻 **Laptop Webcam (Device 0)**")
        camera_source = 0
        is_prerecorded = False
        source_display_name = "Laptop Webcam"

        st.markdown("#### 🔍 **Feed Display Mode**")
        display_mode = st.radio(
            "Display Mode",
            options=["Annotated (MobileNetV2-LSTM HUD)", "Original (Unannotated Frame)"],
            index=0,
            help="Toggle between real-time MobileNetV2-LSTM fall detection view and the clean raw webcam frame."
        )
        show_raw = (display_mode == "Original (Unannotated Frame)")

        st.divider()
        st.markdown("#### 🧠 **Fall Detection Model**")
        st.info("🧠 **MobileNetV2-LSTM** (15 frames @ 3 FPS)")
        live_thresh = st.slider(
            "Live Fall Threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(stream_manager.lstm_threshold),
            step=0.05,
            help="Prediction probability threshold for live MobileNetV2-LSTM fall detection."
        )
        stream_manager.lstm_threshold = live_thresh

        st.caption("• **Architecture**: `MobileNetV2 + LSTM`")
        st.caption("• **Input**: `15 frames @ 3 FPS (224x224)`")
        st.caption("• **Decision**: `P(Fall) >= Threshold`")

        st.divider()
        st.markdown("#### 🔔 **Alarm Siren Status**")
        if stream_manager.alarm_active:
            if stream_manager.alarm_muted:
                st.warning("🔕 **ALARM MUTED** (Visual Alert Active)")
            else:
                st.error("🚨 **AUDIBLE ALARM ACTIVE**")
        else:
            st.success("🔇 **Silent (Normal Activity)**")

        st.divider()
        st.markdown("#### 🧹 **Data & Incident Management**")
        if "confirm_clear_history" not in st.session_state:
            st.session_state["confirm_clear_history"] = False

        if not st.session_state["confirm_clear_history"]:
            if st.button("🗑️ Clear History & Reset", use_container_width=True, help="Safely stops monitoring/alarm, deletes saved incident snapshots, and resets the CSV log."):
                st.session_state["confirm_clear_history"] = True
                st.rerun()
        else:
            st.warning("⚠️ **Confirm Reset?** All incident records, snapshots, and counters will be cleared.")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                if st.button("✔️ Confirm", type="primary", use_container_width=True):
                    # Safely clear runtime data
                    res = stream_manager.clear_runtime_data()
                    st.session_state["confirm_clear_history"] = False
                    if "offline_result" in st.session_state:
                        del st.session_state["offline_result"]
                    st.toast(f"🧹 History cleared! {res.get('deleted_snapshots', 0)} snapshots removed and CSV log reset.")
                    st.rerun()
            with c_col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state["confirm_clear_history"] = False
                    st.rerun()

    # Header & Status Banner
    st.title("🛡️ SafeCCTV: Real-Time Human Fall Detection")

    if stream_manager.alarm_active:
        alarm_status_txt = "MUTED" if stream_manager.alarm_muted else "SIREN SOUNDING"
        st.markdown(f"""
        <div class="alarm-banner">
            🚨 CRITICAL ALERT: FALL DETECTED! 🚨<br>
            <span style="font-size:1.0rem; font-weight:600;">
                Camera: {stream_manager.source_label} | Status: {alarm_status_txt}
            </span>
        </div>
        """, unsafe_allow_html=True)
        render_browser_siren(active=(not stream_manager.alarm_muted))

    elif stream_manager.connection_status == "LIVE":
        tag = "[PRERECORDED TEST]" if stream_manager.is_prerecorded else "[LIVE FEED]"
        st.markdown(f"""
        <div class="status-live">
            🟢 {tag} ACTIVE — {stream_manager.source_label} | Normal Movement Observed (Alarm Silent)
        </div>
        """, unsafe_allow_html=True)
    elif stream_manager.connection_status == "CONNECTING":
        st.markdown("""
        <div class="status-standby" style="border-color: #f59e0b; color: #fbbf24;">
            🟡 CONNECTING TO CAMERA STREAM... Please wait while frames are acquired.
        </div>
        """, unsafe_allow_html=True)
    elif stream_manager.connection_status == "FROZEN":
        st.markdown("""
        <div class="status-standby" style="border-color: #ef4444; color: #f87171;">
            ⚠️ FEED FROZEN — No incoming camera frames for >3.5 seconds. Check camera connection.
        </div>
        """, unsafe_allow_html=True)
    elif stream_manager.connection_status == "CAMERA ERROR":
        st.markdown(f"""
        <div class="status-standby" style="border-color: #ef4444; color: #f87171;">
            ❌ CAMERA ERROR — {stream_manager.error_message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-standby">
            ⚪ SYSTEM STANDBY — Select your camera source and click 'Connect & Start Monitoring'.
        </div>
        """, unsafe_allow_html=True)

    # Action Controls: Start, Stop, Acknowledge, Mute, Reset
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        if st.button("▶️ Connect & Start Monitoring", type="primary", use_container_width=True, disabled=stream_manager.is_running):
            stream_manager.connect(camera_source, is_prerecorded=is_prerecorded, source_name=source_display_name)
            st.rerun()

    with ctrl_col2:
        if st.button("⏹️ Disconnect & Stop", type="secondary", use_container_width=True, disabled=not stream_manager.is_running):
            stream_manager.disconnect()
            st.rerun()

    with ctrl_col3:
        if st.button("✅ Acknowledge Incident", type="primary", use_container_width=True, disabled=not stream_manager.alarm_active):
            stream_manager.acknowledge_alarm()
            st.toast("✅ Incident acknowledged. Siren silenced.")
            st.rerun()

    with ctrl_col4:
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            if st.button("🔇 Mute", use_container_width=True, disabled=(not stream_manager.alarm_active or stream_manager.alarm_muted)):
                stream_manager.mute_alarm()
                st.rerun()
        with sub_c2:
            if st.button("🔄 Reset", use_container_width=True):
                stream_manager.reset_alarm()
                st.toast("🔄 Alarm and detection state reset.")
                st.rerun()

    st.divider()

    # Live Feed & Telemetry Grid
    live_feed_col, telemetry_grid_col = st.columns([1.6, 1.0], gap="large")

    with live_feed_col:
        st.markdown("#### 📺 **Continuous Camera Feed**")
        live_frame_placeholder = st.empty()

    with telemetry_grid_col:
        st.markdown("#### 📊 **Live Telemetry & Metrics**")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            ui_metric_status = st.empty()
        with m_col2:
            ui_metric_fps = st.empty()

        m_col3, m_col4 = st.columns(2)
        with m_col3:
            ui_metric_lstm = st.empty()
        with m_col4:
            ui_metric_falls = st.empty()

        ui_metric_time = st.empty()

        st.markdown("#### 📸 **Recent Fall Incident Evidence**")
        ui_incident_placeholder = st.empty()

    # Continuous Streaming Fragment (50ms refresh loop)
    @st.fragment(run_every="50ms")
    def run_live_stream_ui():
        frame, status, fps, alarm, elapsed_sec = stream_manager.get_latest_frame(return_raw=show_raw)

        if status == "LIVE" and frame is not None:
            # frame is a valid BGR image; channels="BGR" performs the exact one-time BGR->RGB conversion for display
            live_frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        elif status == "CONNECTING":
            live_frame_placeholder.info("🔄 Connecting to laptop webcam... Acquiring initial frames.")
        elif status == "FROZEN":
            live_frame_placeholder.warning("⚠️ Camera feed is frozen. Waiting for incoming stream data...")
        elif status == "CAMERA ERROR":
            live_frame_placeholder.error(f"❌ Camera Error: {stream_manager.error_message}")
        else:
            live_frame_placeholder.info("📹 Camera is currently disconnected. Click **'Connect & Start Monitoring'** above to initiate live feed.")

        # Metric Displays
        if status == "LIVE":
            ui_metric_status.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Status</div>
                <div class="metric-val" style="color: #52c41a;">ONLINE</div>
            </div>
            """, unsafe_allow_html=True)
        elif status == "CONNECTING":
            ui_metric_status.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Status</div>
                <div class="metric-val" style="color: #f59e0b;">CONNECTING</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            ui_metric_status.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Status</div>
                <div class="metric-val" style="color: #9ca3af;">OFFLINE</div>
            </div>
            """, unsafe_allow_html=True)

        ui_metric_fps.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Inference Speed</div>
            <div class="metric-val" style="color: #38bdf8;">{fps:.1f} <span style="font-size:0.9rem;">FPS</span></div>
        </div>
        """, unsafe_allow_html=True)

        prob_val = stream_manager.latest_fall_prob
        prob_color = "#ef4444" if prob_val >= stream_manager.lstm_threshold else ("#f59e0b" if prob_val >= 0.35 else "#38bdf8")
        ui_metric_lstm.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">LSTM Fall Prob</div>
            <div class="metric-val" style="color: {prob_color};">{prob_val * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        fall_count = len(stream_manager.incidents_history)
        fall_color = "#ef4444" if fall_count > 0 else "#e2e8f0"
        ui_metric_falls.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Falls Confirmed</div>
            <div class="metric-val" style="color: {fall_color};">{fall_count}</div>
        </div>
        """, unsafe_allow_html=True)

        ui_metric_time.markdown(f"""
        <div class="metric-card" style="margin-top:-4px;">
            <div class="metric-lbl">Monitoring Time</div>
            <div class="metric-val" style="color: #e2e8f0;">{elapsed_sec:.1f} <span style="font-size:0.9rem;">sec</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Incident Card
        if stream_manager.last_incident_info:
            inc = stream_manager.last_incident_info
            snap_file = inc.get("snapshot_file", "")
            snap_path = os.path.join(stream_manager.incident_dir, snap_file)

            card_html = f"""
            <div style="background: #161b22; border: 1px solid #ef4444; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
                <div style="color: #f87171; font-weight: 700; font-size: 1.0rem;">🚨 Fall Incident Detected</div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
                    • <b>Estimated Fall Time:</b> <span style="color:#fbbf24; font-weight:700;">{inc.get('fall_time_sec', 0.0)} seconds</span> after start<br>
                    • <b>Timestamp:</b> {inc.get('timestamp')}<br>
                    • <b>Camera ID:</b> {inc.get('camera_id')}<br>
                    • <b>Model:</b> <span style="color:#a78bfa; font-weight:600;">MobileNetV2-LSTM</span> (Confidence: {inc.get('confidence', 0.8) * 100:.1f}%)<br>
                    • <b>Status:</b> <span style="color:#38bdf8; font-weight:600;">{inc.get('status')}</span>
                </div>
            </div>
            """
            ui_incident_placeholder.markdown(card_html, unsafe_allow_html=True)

            if os.path.exists(snap_path):
                st.image(snap_path, caption=f"Evidence Snapshot: {snap_file}", use_container_width=True)
        else:
            ui_incident_placeholder.caption("No fall incidents recorded in this session.")

    run_live_stream_ui()

    # Incident Audit Log
    st.divider()
    log_hdr_col1, log_hdr_col2 = st.columns([1.5, 1.0])
    with log_hdr_col1:
        st.subheader("📋 Incident Audit Log & CSV Export")
    with log_hdr_col2:
        if st.button("🗑️ Clear Incident History", key="btn_clear_table", use_container_width=True, help="Safely wipes all incident snapshots and resets the CSV log."):
            st.session_state["confirm_clear_history"] = True
            st.rerun()

    if os.path.exists(stream_manager.csv_log_path) and os.path.getsize(stream_manager.csv_log_path) > 0:
        try:
            df_log = pd.read_csv(stream_manager.csv_log_path)
            if not df_log.empty:
                st.dataframe(df_log, use_container_width=True, hide_index=True)

                csv_bytes = df_log.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Incident Log (CSV)",
                    data=csv_bytes,
                    file_name="SafeCCTV_incident_log.csv",
                    mime="text/csv"
                )
            else:
                st.info("Log file is currently empty.")
        except Exception as e:
            st.warning(f"Could not read incident log: {e}")
    else:
        st.info("No incident records logged yet.")


# =============================================================================
# TAB 2: UPLOADED VIDEO ANALYSIS MODE (TESTING)
# =============================================================================
with tab_upload:
    st.subheader("📁 Uploaded Video Fall Analysis")
    st.caption("Upload prerecorded CCTV footage to run deep learning fall detection with our trained MobileNetV2-LSTM model or heuristic simulation.")

    upload_col1, upload_col2 = st.columns([1.1, 0.9], gap="medium")
    uploaded_video_path = None
    uploaded_video_name = None

    with upload_col1:
        st.markdown("##### ⚙️ **Detection Engine & Parameters**")
        detector_engine = st.radio(
            "Detection Engine",
            options=["MobileNetV2-LSTM (Trained Model)", "Mock Simulation (Heuristic Mode)"],
            index=0,
            horizontal=True,
            help="Select the trained MobileNetV2-LSTM neural network or the mock simulation."
        )

        c_p1, c_p2 = st.columns([1, 1])
        with c_p1:
            decision_threshold = st.slider(
                "Fall Decision Threshold",
                min_value=0.10,
                max_value=0.95,
                value=0.50,
                step=0.05,
                help="Prediction probability threshold for fall classification (default: 0.50)."
            )
        with c_p2:
            model_path_env = os.environ.get("FALL_MODEL_PATH", "research_model/best_fall_model.keras")
            custom_model_path = st.text_input(
                "Model Weights Path",
                value=model_path_env,
                help="Path to the trained .keras weights file."
            )

        up_file = st.file_uploader(
            "Upload Prerecorded MP4/AVI Video",
            type=["mp4", "avi", "mov", "mkv"],
            key="offline_uploader"
        )
        if up_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(up_file.read())
            uploaded_video_path = tfile.name
            uploaded_video_name = up_file.name

            st.success(f"✅ Loaded: **{uploaded_video_name}**")
            btn_run_offline = st.button("🔍 Analyze Uploaded Video", type="primary", use_container_width=True)
        else:
            btn_run_offline = False

    with upload_col2:
        if uploaded_video_path and os.path.exists(uploaded_video_path):
            st.markdown("**Original Video Preview:**")
            st.video(uploaded_video_path)

    if btn_run_offline and uploaded_video_path:
        st.divider()
        st.markdown("#### **Offline Video Analysis Progress**")
        prog_bar = st.progress(0.0)
        status_lbl = st.empty()

        def offline_progress_cb(pct: float, msg: str):
            prog_bar.progress(pct)
            status_lbl.markdown(f"**Progress:** {msg}")

        try:
            use_real = (detector_engine == "MobileNetV2-LSTM (Trained Model)")
            detector = get_detector(
                use_real_model=use_real,
                model_path=custom_model_path,
                threshold=decision_threshold
            )
            res: AnalysisResult = detector.analyze_video(uploaded_video_path, progress_callback=offline_progress_cb)
            st.session_state["offline_result"] = res
            st.success("✅ Video analysis complete!")
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")

    if "offline_result" in st.session_state:
        off_res: AnalysisResult = st.session_state["offline_result"]
        st.divider()
        st.markdown("#### **Analysis Results & Annotated Video**")

        # Summary Metric Cards
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            risk_color = "#ef4444" if off_res.fall_detected else "#22c55e"
            status_text = "FALL DETECTED" if off_res.fall_detected else "NORMAL"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Analysis Result</div>
                <div class="metric-val" style="color: {risk_color}; font-size:1.2rem;">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
        with s_col2:
            max_prob = max([ev.confidence for ev in off_res.fall_events]) if off_res.fall_events else (
                max([t.get("fall_probability", 0.0) for t in off_res.telemetry_data]) if off_res.telemetry_data else 0.0
            )
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Peak Fall Prob</div>
                <div class="metric-val" style="color: #38bdf8;">{max_prob * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with s_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Sequences Evaluated</div>
                <div class="metric-val" style="color: #e2e8f0;">{len(off_res.telemetry_data)}</div>
            </div>
            """, unsafe_allow_html=True)
        with s_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Processing Time</div>
                <div class="metric-val" style="color: #facc15;">{off_res.processing_time_sec:.1f}s</div>
            </div>
            """, unsafe_allow_html=True)

        tab_ann, tab_orig = st.tabs(["📺 Annotated Preview", "📹 Source Video"])
        with tab_ann:
            if off_res.annotated_video_path and os.path.exists(off_res.annotated_video_path):
                st.video(off_res.annotated_video_path)
            else:
                st.info("No annotated preview video generated.")
        with tab_orig:
            if off_res.video_path and os.path.exists(off_res.video_path):
                st.video(off_res.video_path)

        if off_res.fall_events:
            st.markdown("##### **Detected Fall Events**")
            events_data = [{
                "Timestamp": ev.timestamp,
                "Time (s)": f"{ev.time_sec:.2f}s",
                "Frame": ev.frame_idx,
                "Track ID": f"Person #{ev.track_id}",
                "Severity": ev.severity,
                "Torso Angle": f"{ev.torso_angle:.1f}°",
                "Descent Speed": f"{ev.vertical_velocity:+.1f} px",
                "Triggers": " | ".join(ev.condition_triggers)
            } for ev in off_res.fall_events]
            st.dataframe(pd.DataFrame(events_data), use_container_width=True, hide_index=True)
        else:
            st.info("🟢 No fall events detected. All evaluated sequences remained below the decision threshold.")

        if off_res.telemetry_data:
            with st.expander("📈 Temporal Sequence Prediction Breakdown (Raw Telemetry)"):
                st.dataframe(pd.DataFrame(off_res.telemetry_data), use_container_width=True, hide_index=True)

