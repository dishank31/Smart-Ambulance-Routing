import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import datetime
import pandas as pd

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Smart Ambulance & Hospital Recommender",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #c0392b; color: white; }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("🚑 Smart Ambulance Dispatch System")
st.markdown("Real-time severity triage, ETA prediction, and bed availability recommendation.")

with st.sidebar:
    st.header("Patient Triage Form")
    
    with st.expander("Vitals", expanded=True):
        hr = st.slider("Heart Rate (bpm)", 30, 220, 85)
        bp_sys = st.slider("BP Systolic", 60, 200, 120)
        bp_dia = st.slider("BP Diastolic", 40, 130, 80)
        spo2 = st.slider("SpO2 (%)", 70, 100, 98)
        rr = st.slider("Respiratory Rate", 8, 40, 16)
        temp = st.slider("Temperature (°C)", 34.0, 42.0, 37.0, 0.1)
    
    with st.expander("Clinical Assessment", expanded=True):
        gcs = st.slider("GCS Score", 3, 15, 15)
        pain = st.slider("Pain Scale", 0, 10, 5)
        chief_complaint = st.selectbox(
            "Chief Complaint", 
            ['abdominal_pain', 'asthma_attack', 'cardiac_arrest', 'chest_pain', 
             'fracture', 'head_injury', 'seizure', 'severe_bleeding', 'stroke', 'headache', 'cold']
        )
    
    with st.expander("Demographics", expanded=False):
        age = st.number_input("Age", 0, 120, 45)
        gender = st.selectbox("Gender", ['M', 'F'])
        chronic = st.checkbox("Has Chronic Condition")
        
    st.header("Location Context")
    lat = st.number_input("Latitude", value=40.7128)
    lon = st.number_input("Longitude", value=-74.0060)
    
    dispatch_btn = st.button("🚀 REQUEST DISPATCH", use_container_width=True)

if 'dispatch_results' not in st.session_state:
    st.session_state.dispatch_results = None

if dispatch_btn:
    with st.spinner("Analyzing patient severity and assessing hospitals..."):
        payload = {
            "triage": {
                "heart_rate": hr,
                "bp_systolic": bp_sys,
                "bp_diastolic": bp_dia,
                "spo2": spo2,
                "respiratory_rate": rr,
                "temperature": temp,
                "gcs_score": gcs,
                "pain_scale": pain,
                "age": age,
                "has_chronic_condition": 1 if chronic else 0,
                "gender": gender,
                "chief_complaint": chief_complaint
            },
            "location": {
                "lat": lat,
                "lon": lon,
                "hour": datetime.datetime.now().hour,
                "day_of_week": datetime.datetime.now().weekday(),
                "month": datetime.datetime.now().month
            }
        }
        
        try:
            res = requests.post(f"{API_URL}/dispatch", json=payload)
            if res.status_code == 200:
                st.session_state.dispatch_results = res.json()
            elif res.status_code == 503:
                st.warning("Models are still training in the background. Please wait a minute and try again.")
                st.session_state.dispatch_results = None
            else:
                st.error(f"Error: {res.text}")
                st.session_state.dispatch_results = None
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Backend API. Make sure you run `uvicorn backend.main:app --reload`.")
            st.session_state.dispatch_results = None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.dispatch_results = None

if st.session_state.dispatch_results:
    data = st.session_state.dispatch_results
    triage = data['triage_result']
    recs = data['recommendations']
    
    bg_color = {1: "#c0392b", 2: "#e67e22", 3: "#f1c40f", 4: "#27ae60", 5: "#3498db"}.get(triage['severity_level'], "#95a5a6")
    text_color = "black" if triage['severity_level'] in [3, 4, 5] else "white"
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
        <h2 style="color: {text_color}; margin: 0;">{triage['emoji']} AI Triage: Level {triage['severity_level']} - {triage['severity_label']}</h2>
        <p style="font-size: 18px; margin: 5px 0;">Recommended Dept: <b>{triage['recommended_department']}</b> | AI Confidence: <b>{triage['confidence']:.1%}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if not recs:
        st.warning("No hospitals found in radius.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🏥 Recommendation Ranking")
            for i, r in enumerate(recs[:5]):
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 10px; border: 1px solid #ccc; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid {'#27ae60' if i==0 else '#bdc3c7'};">
                        <h4 style="margin: 0;">#{i+1} {r['name']}</h4>
                        <p style="margin: 5px 0; font-size: 14px;">
                        ⏱️ ETA: <b>{r['eta_min']:.1f} min</b> &nbsp;|&nbsp; 
                        🛏️ {r['department']} Beds: <b>{r['beds_available']}</b> &nbsp;|&nbsp;
                        ⭐ Score: <b>{r['score']:.2f}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        with col2:
            st.subheader("📍 Live Map")
            m = folium.Map(location=[lat, lon], zoom_start=11)
            folium.Marker([lat, lon], popup="Incident", icon=folium.Icon(color="red", icon="ambulance", prefix="fa")).add_to(m)
            
            best_h = recs[0]
            folium.Marker(
                [best_h['lat'], best_h['lon']], 
                popup=f"RECOMMENDED: {best_h['name']}", 
                icon=folium.Icon(color="green", icon="hospital", prefix="fa")
            ).add_to(m)
            
            for r in recs[1:5]:
                folium.Marker(
                    [r['lat'], r['lon']], 
                    popup=f"{r['name']}", 
                    icon=folium.Icon(color="blue", icon="h-square", prefix="fa")
                ).add_to(m)
                
            st_folium(m, width=500, height=400, key="dispatch_map")
