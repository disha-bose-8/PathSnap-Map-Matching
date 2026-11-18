import math
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap | Map Matching", page_icon="🧭", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0b1220;color:#e7edf7}.block-container{padding-top:1.5rem;max-width:1500px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:#f4f7fb}
[data-testid="stSidebar"]{background:#101a2b;border-right:1px solid #26344a}
[data-testid="stMetric"]{background:#121e30;border:1px solid #26364e;padding:15px;border-radius:14px}
[data-testid="stMetricLabel"]{color:#9eafc6!important}[data-testid="stMetricValue"]{color:#f5f8ff!important}
.panel{background:#111d2e;border:1px solid #26364e;border-radius:16px;padding:18px}
.muted{color:#9eafc6;font-size:.88rem}.eyebrow{color:#67d5c0;font-size:.76rem;letter-spacing:.13em;font-weight:700}
.pill{display:inline-block;padding:5px 10px;border-radius:99px;background:#153d3a;color:#7be0c9;font-size:.78rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

TRIPS = {
 "Trip 00000000":{"id":"00000000","points":42,"distance":"3.8 km","duration":"11 min","center":(12.9716,77.5946),"offset":.0008},
 "Trip 00000007":{"id":"00000007","points":35,"distance":"2.6 km","duration":"8 min","center":(12.9762,77.6012),"offset":.00065},
 "Trip 00000009":{"id":"00000009","points":58,"distance":"5.1 km","duration":"16 min","center":(12.9654,77.5858),"offset":.001},
 "Trip 00000040":{"id":"00000040","points":29,"distance":"1.9 km","duration":"6 min","center":(12.982,77.590),"offset":.00055},
}
def paths(center, offset, n):
    lat,lon=center; gps=[]; route=[]
    for i in range(n):
        t=i/(n-1); x=(t-.5)*.025; y=math.sin(t*math.pi*2.2)*.004
        a,b=lat+y,lon+x
        route.append((a,b))
        gps.append((a+math.sin(i*2.1)*offset*.23,b+math.cos(i*1.7)*offset*.23))
    return gps,route

with st.sidebar:
    st.markdown("## 🧭 PathSnap")
    st.caption("GPS map-matching dashboard")
    st.divider()
    st.markdown("### Trip workspace")
    selected=st.selectbox("Select a trip",list(TRIPS))
    trip=TRIPS[selected]
    st.caption(f"Trip ID: `{trip['id']}`")
    run=st.button("▶  Run map matching",type="primary",use_container_width=True)
    st.divider()
    st.markdown("### Map layers")
    show_gps=st.toggle("Original GPS points",True)
    show_route=st.toggle("Reconstructed route",True)
    show_markers=st.toggle("Start / end markers",True)
    st.divider()
    st.markdown('<span class="pill">UI DEMO MODE</span>',unsafe_allow_html=True)
    st.caption("Sample map data only. Backend connection comes later.")

c1,c2=st.columns([3,1])
with c1:
    st.markdown('<div class="eyebrow">MAP MATCHING WORKSPACE</div>',unsafe_allow_html=True)
    st.title("Trip analysis")
    st.markdown('<div class="muted">Explore raw GPS observations and the reconstructed road route.</div>',unsafe_allow_html=True)
with c2:
    st.markdown("<div style='text-align:right;padding-top:22px'><span class='pill'>● SYSTEM READY · DEMO</span></div>",unsafe_allow_html=True)

a,b,c,d=st.columns(4)
a.metric("GPS observations",trip["points"],"Original track points")
b.metric("Trip distance",trip["distance"],"Demo estimate")
c.metric("Travel duration",trip["duration"],"Demo estimate")
d.metric("Route continuity","94.2%","Illustrative demo value")

mapcol,side=st.columns([2.1,1],gap="large")
gps,route=paths(trip["center"],trip["offset"],trip["points"])
with mapcol:
    st.markdown("### Route visualisation")
    st.caption("Blue markers = original GPS · Green line = reconstructed route")
    fmap = folium.Map(location=trip["center"], zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Dark Mode",
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Labels",
        overlay=True,
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    if show_route:
        folium.PolyLine(route,color="#31d6a3",weight=5,opacity=.9,tooltip="Reconstructed route (demo)").add_to(fmap)
    if show_gps:
        for i,p in enumerate(gps):
            folium.CircleMarker(p,radius=3.5,color="#67a9ff",fill=True,fill_color="#67a9ff",fill_opacity=.9,tooltip=f"GPS point {i+1}").add_to(fmap)
    if show_markers:
        folium.Marker(route[0],tooltip="Start",icon=folium.Icon(color="green",icon="play")).add_to(fmap)
        folium.Marker(route[-1],tooltip="End",icon=folium.Icon(color="red",icon="stop")).add_to(fmap)
    st_folium(fmap,use_container_width=True,height=500,returned_objects=[])
with side:
    st.markdown("### Evaluation summary")
    st.markdown('<div class="panel"><b>Selected model</b><h3>HistGradientBoosting</h3><span class="muted">Candidate-road classifier</span></div>',unsafe_allow_html=True)
    st.metric("Macro F1","80.44%")
    st.metric("Accuracy","93.14%")
    st.metric("Precision","66.99%")
    st.metric("Recall","62.51%")
    st.markdown("### Trip details")
    st.markdown(f"""<div class="panel"><span class="muted">SELECTED TRIP</span><h3>{trip['id']}</h3>
    GPS observations <b>{trip['points']}</b><hr>Distance <b>{trip['distance']}</b><hr>Duration <b>{trip['duration']}</b></div>""",unsafe_allow_html=True)

st.write("")
st.markdown("### Model performance")
st.caption("Previously generated dataset-level comparison results — not trip-specific metrics.")
df=pd.DataFrame({
 "Model":["HistGradientBoosting","Random Forest","KNN","Decision Tree","Logistic Regression"],
 "Accuracy":[93.14,92.30,91.08,88.14,70.08],
 "Precision":[66.99,62.79,55.33,42.84,24.16],
 "Recall":[62.51,57.43,58.38,53.88,92.39],
 "Macro F1":[80.44,77.87,75.92,70.52,59.28]})
st.dataframe(df.style.format({x:"{:.2f}%" for x in ["Accuracy", "Precision", "Recall", "Macro F1"]}),use_container_width=True,hide_index=True)
st.markdown("""<div class="panel"><b>Prototype note</b><br><span class="muted">The map uses illustrative sample coordinates and trip details. The model metrics are the previously generated dataset-level comparison. No live prediction is performed yet. After integration, the API will supply real trips, predictions and evaluation results.</span></div>""",unsafe_allow_html=True)
st.caption("PathSnap · Frontend prototype · Streamlit + Folium")
