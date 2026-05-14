"""In-process smoke test for the FastAPI server.
Run:
    python api_smoke.py

Requires artifacts/model_v1.joblib and artifacts/deployment_config.json
to exist (produced by `python run_full.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from server import app

ROOT = Path(__file__).parent

# A real gamma-like row from the dataset (small fAlpha → likely gamma).
# Picked from telescope_data.csv row 1 — fAlpha = 6.36° is well into the
# source-pointing regime where gammas dominate.
GAMMA_LIKE = {
    "fLength": 31.6036,
    "fWidth": 11.7235,
    "fSize": 2.5185,
    "fConc": 0.5303,
    "fConc1": 0.3773,
    "fAsym": 26.2722,
    "fM3Long": 23.8238,
    "fM3Trans": -9.9574,
    "fAlpha": 6.3609,
    "fDist": 205.261,
}


def main() -> int:
    if not (ROOT / "artifacts" / "model_v1.joblib").exists():
        print(
            "ERROR: artifacts/model_v1.joblib missing. Run `python run_full.py` first.",
            file=sys.stderr,
        )
        return 1

    with TestClient(app) as client:
        # /health
        r = client.get("/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_loaded"] is True, body
        print("GET  /health         →", body)

        # /model_info
        r = client.get("/model_info")
        assert r.status_code == 200, r.text
        info = r.json()
        assert "deployment_threshold" in info
        print(
            f"GET  /model_info     → model_version={info['model_version']}, "
            f"threshold={info['deployment_threshold']:.4f}"
        )

        # /predict — single event
        r = client.post("/predict", json=GAMMA_LIKE)
        assert r.status_code == 200, r.text
        pred = r.json()
        assert 0.0 <= pred["gamma_probability"] <= 1.0
        assert pred["predicted_class"] in {"g", "h"}
        print(
            f"POST /predict        → P(gamma)={pred['gamma_probability']:.3f}, "
            f"class={pred['predicted_class']}"
        )

        # /predict_batch — three events
        batch_payload = {"events": [GAMMA_LIKE, GAMMA_LIKE, GAMMA_LIKE]}
        r = client.post("/predict_batch", json=batch_payload)
        assert r.status_code == 200, r.text
        batch = r.json()
        assert batch["n_events"] == 3
        print(f"POST /predict_batch  → {batch['n_events']} events scored")

        # Validation: out-of-range input should be 422
        bad = dict(GAMMA_LIKE)
        bad["fAlpha"] = 999  # out of [0, 90]
        r = client.post("/predict", json=bad)
        assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text}"
        print("POST /predict (bad)  → 422 as expected (validation rejected)")

        # Validation: missing field should be 422
        missing = {k: v for k, v in GAMMA_LIKE.items() if k != "fLength"}
        r = client.post("/predict", json=missing)
        assert r.status_code == 422
        print("POST /predict (missing field) → 422 as expected")

        # Validation: geometrically impossible ellipse (width > length) → 422
        bad_geom = dict(GAMMA_LIKE)
        bad_geom["fLength"] = 5.0
        bad_geom["fWidth"] = 50.0
        r = client.post("/predict", json=bad_geom)
        assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text}"
        print("POST /predict (width > length) → 422 as expected")

    print("\nALL API SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
