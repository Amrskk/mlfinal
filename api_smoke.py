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

# A real gamma-like row from the dataset (small fAlpha → likely gamma)
GAMMA_LIKE = {
    "fLength": 28.7967,
    "fWidth": 16.0021,
    "fSize": 2.6449,
    "fConc": 0.3918,
    "fConc1": 0.1982,
    "fAsym": 27.7004,
    "fM3Long": 22.011,
    "fM3Trans": -8.2027,
    "fAlpha": 40.092,
    "fDist": 81.8828,
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

    print("\nALL API SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
