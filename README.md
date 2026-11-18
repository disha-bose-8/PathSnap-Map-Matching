# PathSnap Map Matching

PathSnap maps a noisy GPS trip to directed road arcs. The inference path
generates nearby candidates, scores them with the fitted candidate classifier,
and uses road-network transitions with log-space Viterbi decoding.

## Install and run

From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The default model artifact is `models/candidate_classifier.joblib`. The default
trip partition is `data/Student folder/student_test`. Configure a different
location or partition before starting the server when the evaluator provides
separate trip files:

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\Student folder"
$env:PATHSNAP_EVALUATION_PARTITION = "student_test"
$env:PATHSNAP_MODEL_PATH = "C:\path\to\candidate_classifier.joblib"
```

The API provides `GET /health`, `GET /trips`, and `POST /predict` (also
available as `POST /match`). The prediction request contains only a trip ID:

```json
{"trip_id": "00000000"}
```

The prediction response includes the arc ID for each GPS observation, the
classifier-only baseline, route geometry, continuity, and server-side
reference comparison when the trip folder contains a `.route` file. Reference
arc IDs stay on the server and are not returned to the UI. The comparison
currently uses the nearest reference arc for each GPS point; it is a spatial
proxy and must be replaced if the evaluator specifies a different route metric
or request schema.

Open `http://127.0.0.1:8000/docs` to inspect the API. For the second laptop,
replace `127.0.0.1` with the server laptop's Wi-Fi IPv4 address and allow the
server port through the private-network firewall when prompted.

To compare classifier-only choices with HMM output on the permitted
development partition, run:

```powershell
python -m src.route_evaluation
```

This writes per-trip results to `results/hmm_comparison.csv`. Review the
evaluator's official route-scoring rule before presenting the spatial proxy
accuracy as a final score.

## Implementation notes

- `.arcs` row positions are retained as arc IDs; `.nodes` row positions are
  retained as node IDs.
- HMM states are candidate arcs for one GPS observation.
- Emissions use the trained model's positive-class probability.
- Transitions compare GPS displacement with directed shortest-path distance.
- Missing candidate layers recover with the nearest road arc and are listed in
  `fallback_observation_indices`.
- The route and model APIs do not expose reference arc sequences to the UI.
- Do not report experiment scores until the team has run the experiments and
  recorded their outputs.
