"""Local claim desk. No API key. Open http://127.0.0.1:8741"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from kestrel import (
    DATA,
    MODEL_PATH,
    claims_from_frame,
    dedupe_train,
    history_before,
    load_artifact,
    load_tables,
    score_one,
)

HOST = "127.0.0.1"
PORT = 8741
WEB = Path(__file__).resolve().parent / "web" / "index.html"

EXAMPLES = [
    ("WC710876", "Pune, just under Rs 2,000"),
    ("WC710970", "Ordinary repair"),
    ("WC711262", "Quiet outlet, large repair"),
]


def load_runtime():
    artifact = None
    history = []
    examples = []
    note = ""
    if MODEL_PATH.exists():
        artifact = load_artifact(MODEL_PATH)
    else:
        note = "model/model.json is missing. Run python train.py first."
    train_path = DATA / "train.csv"
    test_path = DATA / "test_unlabelled.csv"
    if artifact and train_path.exists() and test_path.exists() and (DATA / "partners.csv").exists():
        train, test, _partners, _products = load_tables(DATA)
        kept, _extra = dedupe_train(train)
        history = claims_from_frame(kept, labeled=True) + claims_from_frame(test, labeled=False)
        raw = train.set_index("claim_id", drop=False)
        for claim_id, label in EXAMPLES:
            if claim_id not in raw.index:
                continue
            row = raw.loc[claim_id]
            if isinstance(row, type(train)):
                row = row.iloc[-1]
            examples.append(
                {
                    "label": label,
                    "record": {
                        "claim_id": str(row["claim_id"]),
                        "submitted_at": str(row["submitted_at"]),
                        "partner_id": str(row["partner_id"]),
                        "sku": str(row["sku"]),
                        "product_serial": str(row["product_serial"]),
                        "days_since_purchase": int(row["days_since_purchase"]),
                        "claim_amount_inr": float(row["claim_amount_inr"]),
                        "photo_attached": str(row["photo_attached"]),
                        "partner_inspected": str(row["partner_inspected"]),
                        "claim_description": str(row["claim_description"]).split(".")[0][:80],
                        "inspector_note": "" if pd_isna(row["inspector_note"]) else str(row["inspector_note"]),
                        "customer_prior_claims": int(row["customer_prior_claims"]),
                        "source": str(row["source"]),
                    },
                }
            )
    elif artifact:
        note = "The score file is here, but data/ is missing, so a claim cannot be placed in outlet history."
    return {"artifact": artifact, "history": history, "examples": examples, "note": note}


def pd_isna(value) -> bool:
    if value is None:
        return True
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        return False
    text = str(value).strip().lower()
    return text in {"", "nan", "none"}


RUNTIME = load_runtime()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            if not WEB.exists():
                self._json(500, {"error": "The screen file web/index.html is missing."})
                return
            self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self._json(
                200,
                {
                    "ok": RUNTIME["artifact"] is not None and bool(RUNTIME["history"]),
                    "note": RUNTIME["note"],
                    "examples": len(RUNTIME["examples"]),
                },
            )
            return
        if path == "/api/examples":
            self._json(200, {"examples": RUNTIME["examples"], "note": RUNTIME["note"]})
            return
        self._json(404, {"error": "That page is not on this desk."})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/score":
            self._json(404, {"error": "That page is not on this desk."})
            return
        if RUNTIME["artifact"] is None:
            self._json(503, {"error": RUNTIME["note"] or "The model is not on this machine."})
            return
        if not RUNTIME["history"]:
            self._json(
                503,
                {
                    "error": (
                        "The desk can start, but it has no claim history to score against. "
                        "Put train.csv, test_unlabelled.csv, partners.csv and products.csv in data/ "
                        "and run python train.py."
                    )
                },
            )
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            record = json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "The body is not JSON."})
            return
        try:
            result = score_one(record, RUNTIME["artifact"], history_before(record, RUNTIME["history"]))
        except KeyError as exc:
            self._json(400, {"error": f"Missing field {exc.args[0]}."})
            return
        except (TypeError, ValueError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, result)


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")
    server = None
    bound = None
    for port in range(PORT, PORT + 10):
        try:
            server = ThreadingHTTPServer((HOST, port), Handler)
            bound = port
            break
        except OSError:
            continue
    if server is None or bound is None:
        raise SystemExit(f"Could not bind {HOST}:{PORT} through {PORT + 9}.")
    print(f"Claim desk at http://{HOST}:{bound}")
    if RUNTIME["note"]:
        print(RUNTIME["note"])
    server.serve_forever()


if __name__ == "__main__":
    main()
