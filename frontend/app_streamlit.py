from __future__ import annotations

import os
import time
from datetime import datetime
from math import sqrt

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.features import DivIcon
from folium.plugins import AntPath, Fullscreen, MiniMap
from streamlit_folium import st_folium

API_BASE_URL = os.getenv("SMART_AMBULANCE_API", "http://localhost:8000")

st.set_page_config(
    page_title="Smart Ambulance System",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .big-font { font-size: 30px !important; font-weight: 700; }
    .metric-card { background: #f6f7fb; padding: 18px; border-radius: 14px; border: 1px solid #e2e6ef; }
    .small-note { color: #5f6b7a; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, params: dict | None = None):
    return requests.get(f"{API_BASE_URL}{path}", params=params, timeout=20)


def api_post(path: str, payload: dict):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)


@st.cache_data(ttl=120, show_spinner=False)
def get_model_comparison_cached():
    response = api_get("/api/models/comparison")
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def get_nearby_hospitals_cached(lat: float, lon: float, radius: int):
    response = api_get("/api/hospitals/nearby", params={"lat": lat, "lon": lon, "radius_km": radius})
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=15, show_spinner=False)
def get_health_cached():
    response = api_get("/api/health")
    response.raise_for_status()
    return response.json()


def route_points(route_geometry: dict | None, origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float):
    if route_geometry and route_geometry.get("coordinates"):
        return [[coord[1], coord[0]] for coord in route_geometry["coordinates"]]
    return [[origin_lat, origin_lon], [dest_lat, dest_lon]]


def midpoint(points: list[list[float]]):
    if not points:
        return None
    return points[len(points) // 2]


def point_along_path(points: list[list[float]], progress: float):
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    clamped = max(0.0, min(1.0, progress))

    segment_lengths = []
    total_length = 0.0
    for idx in range(len(points) - 1):
        lat1, lon1 = points[idx]
        lat2, lon2 = points[idx + 1]
        seg_len = sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
        segment_lengths.append(seg_len)
        total_length += seg_len

    if total_length == 0:
        return points[0]

    target = total_length * clamped
    traversed = 0.0
    for idx, seg_len in enumerate(segment_lengths):
        if traversed + seg_len >= target:
            start_lat, start_lon = points[idx]
            end_lat, end_lon = points[idx + 1]
            ratio = 0.0 if seg_len == 0 else (target - traversed) / seg_len
            return [
                start_lat + (end_lat - start_lat) * ratio,
                start_lon + (end_lon - start_lon) * ratio,
            ]
        traversed += seg_len
    return points[-1]


def has_real_route(route_geometry: dict | None):
    return bool(route_geometry and route_geometry.get("coordinates") and len(route_geometry["coordinates"]) > 2)


def current_step_index(steps: list[dict], progress: float):
    if not steps:
        return None
    total_distance = sum(max(float(step.get("distance_m", 0.0)), 0.0) for step in steps)
    total_duration = sum(max(float(step.get("duration_s", 0.0)), 0.0) for step in steps)
    if total_distance <= 0 and total_duration <= 0:
        return 0

    target_ratio = max(0.0, min(1.0, progress))
    target_value = total_distance * target_ratio if total_distance > 0 else total_duration * target_ratio
    traversed = 0.0
    for idx, step in enumerate(steps):
        step_value = max(float(step.get("distance_m", 0.0)), 0.0) if total_distance > 0 else max(float(step.get("duration_s", 0.0)), 0.0)
        traversed += step_value
        if traversed >= target_value:
            return idx
    return len(steps) - 1


def format_step_instruction(step: dict):
    instruction = step.get("instruction", "Continue")
    distance_m = float(step.get("distance_m", 0.0))
    duration_min = float(step.get("duration_s", 0.0)) / 60.0
    if distance_m >= 1000:
        distance_label = f"{distance_m / 1000:.1f} km"
    else:
        distance_label = f"{distance_m:.0f} m"
    return f"{instruction}  \n{distance_label} | {duration_min:.1f} min"


def hospital_popup(hospital: dict, label: str):
    return folium.Popup(
        f"""
        <div style="min-width: 220px;">
            <strong>{label}</strong><br>
            <strong>{hospital['name']}</strong><br>
            ETA: {hospital['predicted_eta_minutes']:.1f} min<br>
            Beds: {hospital['predicted_beds_available']}<br>
            Score: {hospital.get('score', 0):.2f}
        </div>
        """,
        max_width=280,
    )


def render_choice_card(title: str, hospital: dict, accent: str, body: str):
    st.markdown(
        f"""
        <div style="
            border: 1px solid {accent};
            border-radius: 18px;
            padding: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
            box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
        ">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: {accent};">{title}</div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.45rem;">{hospital['name']}</div>
            <div style="display: flex; gap: 18px; margin-top: 0.8rem; flex-wrap: wrap;">
                <div><strong>ETA</strong><br>{hospital['predicted_eta_minutes']:.1f} min</div>
                <div><strong>Beds</strong><br>{hospital['predicted_beds_available']}</div>
                <div><strong>Score</strong><br>{hospital.get('score', 0):.2f}</div>
            </div>
            <div style="margin-top: 0.9rem; color: #c7ced9;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reason_badges(items: list[str]):
    badge_colors = ["#32d17c", "#4da3ff", "#ff8c42", "#c084fc", "#facc15"]
    html_parts = [
        '<div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 18px; background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));">',
        '<div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; color: #d7deea; margin-bottom: 12px;">Why This Hospital Won</div>',
        '<div style="display: flex; flex-wrap: wrap; gap: 10px;">',
    ]
    for idx, item in enumerate(items):
        color = badge_colors[idx % len(badge_colors)]
        html_parts.append(
            f'<div style="padding: 10px 14px; border-radius: 999px; border: 1px solid {color}; background: rgba(255,255,255,0.02); color: #eef3fb; font-size: 0.92rem; line-height: 1.35;">{item}</div>'
        )
    html_parts.extend(["</div>", "</div>"])
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def plot_route_map(origin_lat: float, origin_lon: float, result: dict, ambulance_progress: float = 0.5):
    optimal = result["optimal_hospital"]
    naive = result["naive_choice"]
    fmap = folium.Map(location=[origin_lat, origin_lon], zoom_start=11, tiles="CartoDB dark_matter")
    Fullscreen(position="topright").add_to(fmap)
    MiniMap(toggle_display=True, position="bottomright").add_to(fmap)

    folium.Marker(
        [origin_lat, origin_lon],
        popup="Emergency pickup point",
        icon=folium.Icon(color="red", icon="plus", prefix="fa"),
    ).add_to(fmap)
    folium.CircleMarker(
        [origin_lat, origin_lon],
        radius=16,
        color="#ff4b4b",
        fill=True,
        fill_opacity=0.18,
        weight=2,
    ).add_to(fmap)

    optimal_points = route_points(
        optimal.get("route_geometry"),
        origin_lat,
        origin_lon,
        optimal["location"]["latitude"],
        optimal["location"]["longitude"],
    )
    naive_points = route_points(
        naive.get("route_geometry"),
        origin_lat,
        origin_lon,
        naive["location"]["latitude"],
        naive["location"]["longitude"],
    )

    folium.Marker(
        [optimal["location"]["latitude"], optimal["location"]["longitude"]],
        popup=hospital_popup(optimal, "Smart Route Winner"),
        icon=folium.Icon(color="green", icon="hospital-o", prefix="fa"),
    ).add_to(fmap)
    for alt in result.get("alternative_hospitals", []):
        folium.Marker(
            [alt["location"]["latitude"], alt["location"]["longitude"]],
            popup=hospital_popup(alt, "Alternative"),
            icon=folium.Icon(color="lightgray", icon="hospital-o", prefix="fa"),
        ).add_to(fmap)
    if naive["hospital_id"] != optimal["hospital_id"]:
        folium.Marker(
            [naive["location"]["latitude"], naive["location"]["longitude"]],
            popup=hospital_popup(naive, "Nearest But Not Best"),
            icon=folium.Icon(color="orange", icon="hospital-o", prefix="fa"),
        ).add_to(fmap)

    folium.PolyLine(
        naive_points,
        color="#ff8c42",
        weight=3,
        opacity=0.45,
        dash_array="8, 12",
        tooltip="Nearest-hospital route",
    ).add_to(fmap)
    AntPath(
        optimal_points,
        color="#32d17c",
        pulse_color="#8ef7bc",
        weight=7,
        delay=900,
        tooltip="Smart ambulance route",
    ).add_to(fmap)

    ambulance_point = point_along_path(optimal_points, ambulance_progress)
    if ambulance_point:
        folium.Marker(
            ambulance_point,
            tooltip="Ambulance en route",
            icon=DivIcon(
                icon_size=(36, 36),
                icon_anchor=(18, 18),
                html="""
                <div style="
                    width: 36px;
                    height: 36px;
                    border-radius: 50%;
                    background: rgba(50, 209, 124, 0.18);
                    border: 2px solid #32d17c;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #ffffff;
                    font-size: 18px;
                    box-shadow: 0 0 18px rgba(50, 209, 124, 0.45);
                ">🚑</div>
                """,
            ),
        ).add_to(fmap)

    return fmap


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Emergency Input", "Model Comparison", "Hospital Dashboard", "SHAP Explanations", "Performance Metrics"],
)

if page == "Home":
    st.title("Smart Ambulance Routing System")
    st.markdown("### AI-powered emergency triage and hospital recommendation")
    st.write(
        "This app classifies patient severity, predicts ETA and bed availability, and recommends the best hospital based on both travel time and capacity rather than distance alone."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Severity Module", "Ready", "Rule + ML fallback")
    col2.metric("Routing Logic", "Smart", "Beds + ETA aware")
    col3.metric("Frontend", "Live", "FastAPI connected")

elif page == "Emergency Input":
    st.title("Emergency Request")
    st.write("Enter the incident location and patient vitals to get the recommended hospital.")

    # Initialize session state for results
    if "recommendation_result" not in st.session_state:
        st.session_state.recommendation_result = None
    if "ambulance_progress" not in st.session_state:
        st.session_state.ambulance_progress = 0.45
    if "live_guidance" not in st.session_state:
        st.session_state.live_guidance = False
    if "last_live_tick" not in st.session_state:
        st.session_state.last_live_tick = time.time()

    with st.form("emergency_form"):
        loc1, loc2 = st.columns(2)
        lat = loc1.number_input("Latitude", value=40.7589, format="%.4f")
        lon = loc2.number_input("Longitude", value=-73.9851, format="%.4f")

        v1, v2, v3, v4 = st.columns(4)
        heart_rate = v1.number_input("Heart Rate", 30, 220, 88)
        bp_sys = v1.number_input("BP Systolic", 40, 250, 118)
        bp_dia = v2.number_input("BP Diastolic", 20, 150, 78)
        spo2 = v2.number_input("SpO2", 60, 100, 97)
        resp_rate = v3.number_input("Respiratory Rate", 5, 50, 18)
        temperature = v3.number_input("Temperature (C)", 34.0, 42.0, 37.1, format="%.1f")
        gcs = v4.slider("GCS", 3, 15, 15)
        pain = v4.slider("Pain", 0, 10, 4)

        p1, p2, p3 = st.columns(3)
        age = p1.number_input("Age", 0, 120, 45)
        gender = p2.selectbox("Gender", ["Male", "Female"])
        chronic = p3.checkbox("Has chronic condition")
        complaint = st.selectbox(
            "Chief Complaint",
            ["chest_pain", "trauma", "respiratory_failure", "abdominal_pain", "fracture", "seizure", "stroke", "cardiac_arrest", "minor_laceration", "headache", "cold"],
        )
        use_ml_eta = st.checkbox("Use ML ETA", value=True)
        submitted = st.form_submit_button("Get Recommendation", type="primary")

    if submitted:
        payload = {
            "location": {"latitude": lat, "longitude": lon},
            "patient_vitals": {
                "heart_rate": heart_rate,
                "bp_systolic": bp_sys,
                "bp_diastolic": bp_dia,
                "spo2": spo2,
                "respiratory_rate": resp_rate,
                "temperature": temperature,
                "gcs_score": gcs,
                "pain_scale": pain,
                "age": age,
                "gender": gender,
                "has_chronic_condition": chronic,
                "chief_complaint": complaint,
            },
            "timestamp": datetime.utcnow().isoformat(),
            "use_ml_eta": use_ml_eta,
        }
        with st.spinner("Computing recommendation..."):
            try:
                response = api_post("/api/emergency/recommend", payload)
                response.raise_for_status()
                st.session_state.recommendation_result = response.json()
                st.session_state.ambulance_progress = 0.0
                st.session_state.last_live_tick = time.time()
            except Exception as exc:
                st.error(f"API request failed: {exc}")
                st.session_state.recommendation_result = None

    # Display results (persisted in session state)
    if st.session_state.recommendation_result:
        result = st.session_state.recommendation_result
        severity = result["severity_info"]
        optimal = result["optimal_hospital"]
        naive = result["naive_choice"]

        st.success("Recommendation generated")
        st.subheader(f"Severity: {severity['severity_label']}")
        s1, s2, s3 = st.columns(3)
        s1.metric("Severity Level", severity["severity"])
        s2.metric("Confidence", f"{severity['confidence'] * 100:.1f}%")
        s3.metric("Department", severity["required_department"])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Hospital", optimal["name"])
        r2.metric("ETA", f"{optimal['predicted_eta_minutes']:.1f} min")
        r3.metric("Available Beds", optimal["predicted_beds_available"])
        r4.metric("Score", f"{optimal['score']:.2f}")
        st.caption(f"Decision latency: {result.get('processing_time_ms', 0):.2f} ms")

        st.markdown(result["recommendation_reason"])
        breakdown_items = result.get("recommendation_breakdown", [])
        if breakdown_items:
            render_reason_badges(breakdown_items)
        if has_real_route(optimal.get("route_geometry")):
            st.caption("Displaying road-following route geometry with maneuver guidance using Leaflet + OSRM.")
        else:
            st.warning("Real road geometry is unavailable right now. The app now prefers OpenRouteService, then OSRM, with Mapbox only as an optional fallback.")
        progress_left, progress_right = st.columns([4, 1])
        with progress_left:
            if st.session_state.live_guidance:
                now = time.time()
                elapsed = max(0.0, now - st.session_state.last_live_tick)
                st.session_state.last_live_tick = now
                st.session_state.ambulance_progress = min(1.0, st.session_state.ambulance_progress + (elapsed / 45.0))
            st.session_state.ambulance_progress = st.slider(
                "Ambulance route progress",
                0.0,
                1.0,
                float(st.session_state.ambulance_progress),
                0.05,
                key="ambulance_progress_slider",
            )
        with progress_right:
            st.write("")
            st.write("")
            animate_route = st.button("Animate Route", use_container_width=True)

        live_left, live_right = st.columns([3, 1])
        with live_left:
            st.session_state.live_guidance = st.checkbox(
                "Live guidance mode",
                value=st.session_state.live_guidance,
                help="Continuously advances ambulance progress and updates the active maneuver.",
            )
        with live_right:
            if st.button("Reset Live", use_container_width=True):
                st.session_state.ambulance_progress = 0.0
                st.session_state.last_live_tick = time.time()
                st.rerun()

        active_idx = current_step_index(optimal.get("route_steps", []), st.session_state.ambulance_progress)
        if active_idx is not None:
            active_step = optimal["route_steps"][active_idx]
            st.info(f"Current direction: {format_step_instruction(active_step)}")
            if active_idx + 1 < len(optimal["route_steps"]):
                st.caption(f"Next: {optimal['route_steps'][active_idx + 1].get('instruction', 'Continue')}")

        map_placeholder = st.empty()
        map_placeholder_fallback = plot_route_map(lat, lon, result, st.session_state.ambulance_progress)
        map_placeholder_fallback_ref = map_placeholder_fallback
        if animate_route:
            for frame in range(13):
                animated_progress = frame / 12
                st.session_state.ambulance_progress = animated_progress
                map_placeholder.empty()
                with map_placeholder.container():
                    st_folium(plot_route_map(lat, lon, result, animated_progress), width=1000, height=520, key=f"animated_map_{frame}")
                time.sleep(0.18)

        with map_placeholder.container():
            st_folium(
                plot_route_map(lat, lon, result, st.session_state.ambulance_progress),
                width=1000,
                height=520,
                key=f"route_map_{int(st.session_state.ambulance_progress * 100)}",
            )
        if st.session_state.live_guidance and st.session_state.ambulance_progress < 1.0:
            time.sleep(1.0)
            st.rerun()

        compare_left, compare_right = st.columns(2)
        with compare_left:
            render_choice_card("Smart Choice", optimal, "#32d17c", result["recommendation_reason"])
        with compare_right:
            render_choice_card("Nearest Choice", naive, "#ff8c42", naive["why_not_optimal"])

        route_left, route_right = st.columns(2)
        with route_left:
            with st.expander("Smart Route Guidance", expanded=True):
                steps = optimal.get("route_steps", [])
                if steps:
                    for idx, step in enumerate(steps[:8], start=1):
                        st.markdown(f"**{idx}.** {format_step_instruction(step)}")
                else:
                    st.info("Turn-by-turn guidance is not available for this route.")
        with route_right:
            with st.expander("Nearest Route Guidance", expanded=False):
                steps = naive.get("route_steps", [])
                if steps:
                    for idx, step in enumerate(steps[:8], start=1):
                        st.markdown(f"**{idx}.** {format_step_instruction(step)}")
                else:
                    st.info("Turn-by-turn guidance is not available for the nearest route.")

        with st.expander("Real-Time Routing Requirements", expanded=False):
            st.markdown(
                """
                For realistic road-following routes and live re-routing, the app needs:

                1. Preferred: local OSRM via `LOCAL_OSRM_URL`
                2. Or `OPENROUTESERVICE_API_KEY` for a stable hosted provider
                3. Backend restarted after code changes
                4. Optional `MAPBOX_ACCESS_TOKEN` only as a final fallback

                Recommended PowerShell setup:
                ```powershell
                $env:LOCAL_OSRM_URL="http://127.0.0.1:5000/route/v1"
                .\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8016
                ```

                In a second terminal:
                ```powershell
                $env:SMART_AMBULANCE_API="http://127.0.0.1:8016"
                $env:LOCAL_OSRM_URL="http://127.0.0.1:5000/route/v1"
                .\\.venv\\Scripts\\python.exe -m streamlit run frontend/app_streamlit.py --server.port 8516
                ```
                """
            )

        sim_left, sim_right = st.columns([3, 1])
        with sim_left:
            simulated_traffic = st.slider("Traffic stress test", 1, 10, 5, key="traffic_slider")
        with sim_right:
            st.write("")
            st.write("")
            if st.button("Re-route", use_container_width=True):
                with st.spinner("Re-evaluating route under new traffic conditions..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/simulate/traffic",
                            params={"new_traffic_level": simulated_traffic},
                            json=result,
                            timeout=30,
                        )
                        response.raise_for_status()
                        updated = response.json()
                        updated["processing_time_ms"] = result.get("processing_time_ms", 0.0)
                        st.session_state.recommendation_result = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Traffic simulation failed: {exc}")

        left, right = st.columns(2)
        with left:
            st.markdown("### Naive Choice")
            st.json(naive)
        with right:
            st.markdown("### Alternatives")
            if result["alternative_hospitals"]:
                st.dataframe(pd.DataFrame(result["alternative_hospitals"]))
            else:
                st.info("No alternative hospitals returned.")

elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    try:
        payload = get_model_comparison_cached()
        for label, key in [("Severity Models", "severity_models"), ("ETA Models", "eta_models"), ("Bed Models", "bed_models")]:
            st.subheader(label)
            df = pd.DataFrame(payload.get(key, []))
            if df.empty:
                st.info("No metrics available")
                continue
            # Display as table first
            st.dataframe(df, use_container_width=True)
            # Filter numeric columns for chart
            numeric_cols = []
            for c in df.columns:
                if c != "model_name":
                    # Check if column has any valid numeric values
                    try:
                        pd.to_numeric(df[c], errors='coerce')
                        if not pd.to_numeric(df[c], errors='coerce').isna().all():
                            numeric_cols.append(c)
                    except:
                        pass
            if numeric_cols:
                # Convert to numeric for plotting
                plot_df = df.copy()
                for col in numeric_cols:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                fig = px.bar(plot_df, x="model_name", y=numeric_cols, barmode="group")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load model metrics: {exc}")

elif page == "Hospital Dashboard":
    st.title("Hospital Dashboard")
    lat = st.number_input("Reference Latitude", value=40.7589, format="%.4f", key="dash_lat")
    lon = st.number_input("Reference Longitude", value=-73.9851, format="%.4f", key="dash_lon")
    radius = st.slider("Radius (km)", 1, 50, 25)
    try:
        hospitals = get_nearby_hospitals_cached(lat, lon, radius)
        df = pd.DataFrame(hospitals)
        st.dataframe(df, use_container_width=True)
        if not df.empty:
            hover_fields = [field for field in ["distance_km", "total_icu_beds", "total_emergency_beds", "total_general_beds"] if field in df.columns]
            fig = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                hover_name="name",
                hover_data=hover_fields,
                zoom=10,
                height=500,
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to load hospitals: {exc}")

elif page == "SHAP Explanations":
    st.title("SHAP Explanations")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    shap_assets = [
        ("Severity Model", os.path.join(results_dir, "shap_severity.png")),
        ("ETA Model", os.path.join(results_dir, "shap_eta.png")),
        ("Bed Availability Model", os.path.join(results_dir, "shap_bed.png")),
        ("Legacy Summary", os.path.join(results_dir, "shap_summary.png")),
    ]
    available_assets = [(label, path) for label, path in shap_assets if os.path.exists(path) and os.path.getsize(path) > 0]
    missing_assets = [os.path.basename(path) for _, path in shap_assets if not (os.path.exists(path) and os.path.getsize(path) > 0)]

    if available_assets:
        st.caption("Saved SHAP plots from the training pipeline.")
        for label, path in available_assets:
            st.subheader(label)
            st.image(path, use_container_width=True)
    else:
        st.warning("No SHAP plot assets were found in `results/`.")

    if missing_assets:
        st.info("Missing SHAP assets: " + ", ".join(f"`{name}`" for name in missing_assets))
        st.markdown("Generate them from `notebooks/07_shap.ipynb` and save them into `results/`.")

elif page == "Performance Metrics":
    st.title("Performance Metrics")
    try:
        health = get_health_cached()
        st.json(health)
    except Exception as exc:
        st.warning(f"Health endpoint unavailable: {exc}")

    sample = pd.DataFrame(
        {
            "metric": ["API latency", "Severity confidence", "Average ETA error", "Average bed MAE"],
            "value": [210, 82, 3.5, 2.8],
        }
    )
    fig = px.bar(sample, x="metric", y="value", title="Current Demo Metrics")
    st.plotly_chart(fig, use_container_width=True)
