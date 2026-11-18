# PathSnap UI — Frontend Prototype

## Run on Windows / VS Code

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ui.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Included
- Trip selector and Run Map Matching button
- Interactive Folium map with original GPS marker layer
- Reconstructed route layer and start/end markers
- Toggle controls for map layers
- Model comparison metrics table

**Demo status:** The trip coordinates and trip-specific summary values are illustrative placeholders. The model comparison table shows previously generated dataset-level results. The Run button is currently a UI control only; live FastAPI integration with Laptop 1 will be added later.
