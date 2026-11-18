# PathSnap Laptop 1 Backend Setup

Laptop 1 runs the complete map-matching backend. Laptop 2 calls this API over the same Wi-Fi network.

## 1. Install dependencies

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Generate the local model

The trained model is a local artifact and is intentionally not committed to Git. The candidate dataset archive is already in `results/all_train_candidates.zip`.

Tune the selected HistGradientBoosting model before fitting when the demo
schedule allows it. This reserves the 10-trip holdout and searches only within
the other 70 trips:

```powershell
python -m src.bayesian_optimization
```

The training script automatically uses the saved Optuna parameters when they
are available. The default search uses 20 trials and a proportional sample of
up to 250,000 rows to keep runtime manageable.

```powershell
python -m src.train
```

This creates `models/candidate_classifier.joblib` and `models/metadata.json`. Keep the model on Laptop 1.

Optional: evaluate the held-out trips:

```powershell
python -m src.evaluation
```

## 3. Start the API

```powershell
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Keep this terminal running. On Laptop 1, open:

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Available trips: http://127.0.0.1:8000/trips

The health check should report `"status": "ok"` and `"model_loaded": true`.

## 4. API contract for Laptop 2

### List trips

`GET /trips`

Response:

```json
{"trip_ids": ["00000000", "00000007"]}
```

### Reconstruct a trip

`POST /predict`

Request:

```json
{"trip_id": "00000000"}
```

The response includes:

- `gps_points`: original GPS coordinates as [latitude, longitude]
- `predicted_arc_ids`: one predicted road-arc ID per GPS observation
- `classifier_only_arc_ids`: classifier-only baseline
- `predicted_route`: road-arc endpoint coordinates for map drawing
- `route_continuity`: fraction of consecutive predicted arcs connected in the directed graph
- `fallback_observation_indices`: observations where no candidate was available
- `evaluation`: optional nearest-reference-arc proxy when reference route data exists

The reference evaluation is a spatial proxy, not necessarily the official evaluator's scoring rule.

## 5. Connect Laptop 2

1. Keep both laptops on the same Wi-Fi network.
2. On Laptop 1, run `ipconfig` and find its IPv4 address (for example, `192.168.1.20`).
3. Allow inbound TCP port 8000 through Windows Firewall on Laptop 1.
4. Laptop 2 should call `http://192.168.1.20:8000` (replace with Laptop 1's actual IPv4 address).
5. Frontend requests:
   - `GET http://<LAPTOP1_IP>:8000/trips`
   - `POST http://<LAPTOP1_IP>:8000/predict`

Do not use `127.0.0.1` from Laptop 2; that would refer to Laptop 2 itself.

## Troubleshooting

- If `model_loaded` is false, activate the environment and run `python -m src.train` from the repository root.
- If Laptop 2 cannot connect, check both laptops' Wi-Fi, Laptop 1's firewall rule, and the IPv4 address.
- If the API returns 404 for a trip, check that the trip folder exists under `data/Student folder/student_test/`.
- The API defaults to the `student_test` partition. To change it, set `PATHSNAP_EVALUATION_PARTITION` before starting Uvicorn.
