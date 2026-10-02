"""Warranty-claim queue for Kestrel Home.

Two scores, because the fraud changed shape on 1 May 2026.

Classic score: a repair that is a large share of the list price, from an
outlet that already has confirmed fraud, or from a customer with several
earlier claims. Fit on investigated claims from before 1 May 2026.

Cap score: from 1 May 2026, claims under Rs 2,000 are paid without an
inspection. Fit on investigated claims from that date onward. It watches
the auto-approval path, amounts just under Rs 2,000, and outlets that
have started filing a run of those claims.

The number handed back is the chance the claim is fraud. Higher means
more likely. A claim is worth a desk slot when that chance, times the
rupees, is larger than the goodwill of holding a genuine customer.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "model" / "model.json"

GOODWILL_INR = 380
CONTACT_INR = 260
CAP_INR = 2000
HUG_LOW = 1900
POLICY_AT = pd.Timestamp("2026-05-01")
WINDOW = pd.Timedelta(days=30)
SHRINK_K = 20
# Pre-May confirmed-fraud rate, rounded. New outlets start here, not at
# "new partner, therefore fraud".
SHRINK_RATE = 0.011
DESK_SLOTS = 40

CLASSIC_FEATURES = ["ratio", "partner_rate", "priors", "log_amt", "nophoto", "unins"]
CAP_FEATURES = ["cap_open", "hug", "small30", "known", "priors", "ratio", "burst", "nophoto"]


def load_tables(data_dir: Path = DATA):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_unlabelled.csv")
    partners = pd.read_csv(data_dir / "partners.csv")
    products = pd.read_csv(data_dir / "products.csv")
    return train, test, partners, products


def _yn(value) -> int:
    text = str(value).strip().upper()
    if text in {"Y", "YES", "TRUE", "1"}:
        return 1
    if text in {"N", "NO", "FALSE", "0"}:
        return 0
    raise ValueError(f"Expected Y or N, got {value!r}")


def _stamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def claims_from_frame(frame: pd.DataFrame, labeled: bool) -> list[dict]:
    rows = []
    for rec in frame.itertuples(index=False):
        fraud = None
        if labeled:
            raw = rec.is_fraud
            if pd.notna(raw):
                fraud = int(raw)
        rows.append(
            {
                "claim_id": str(rec.claim_id),
                "submitted_at": _stamp(rec.submitted_at),
                "partner_id": str(rec.partner_id),
                "sku": str(rec.sku),
                "amount": float(rec.claim_amount_inr),
                "photo": str(rec.photo_attached),
                "inspected": str(rec.partner_inspected),
                "prior_customer": int(rec.customer_prior_claims),
                "is_fraud": fraud,
            }
        )
    return rows


def dedupe_train(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Partners re-file a bounced claim under the same number. Keep the later one."""
    ordered = frame.sort_values(["claim_id", "submitted_at", "claim_id"])
    extra = int(ordered["claim_id"].duplicated(keep=False).sum())
    kept = ordered.drop_duplicates("claim_id", keep="last")
    return kept, extra


def partner_lookup(partners: pd.DataFrame) -> dict:
    out = {}
    for rec in partners.itertuples(index=False):
        out[str(rec.partner_id)] = {
            "city": str(rec.city),
            "onboarded_date": str(rec.onboarded_date)[:10],
            "partner_type": str(rec.partner_type),
        }
    return out


def product_lookup(products: pd.DataFrame) -> dict:
    out = {}
    for rec in products.itertuples(index=False):
        out[str(rec.sku)] = {
            "family": str(rec.family),
            "list_price_inr": float(rec.list_price_inr),
            "warranty_months": int(rec.warranty_months),
        }
    return out


def walk_features(claims: list[dict], partners: dict, products: dict) -> list[dict]:
    """Past-only features. A row never sees its own label or a later claim."""
    ordered = sorted(claims, key=lambda c: (c["submitted_at"], c["claim_id"]))
    fraud_n = defaultdict(int)
    decided_n = defaultdict(int)
    recent = defaultdict(deque)
    built = []
    for claim in ordered:
        pid = claim["partner_id"]
        when = claim["submitted_at"]
        dq = recent[pid]
        cutoff = when - WINDOW
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        hug30 = 0
        small30 = 0
        for ts, was_hug, was_small in dq:
            hug30 += was_hug
            small30 += was_small
        prior_f = fraud_n[pid]
        prior_d = decided_n[pid]
        partner = partners.get(pid)
        product = products.get(claim["sku"])
        price = product["list_price_inr"] if product else np.nan
        amount = float(claim["amount"])
        ratio = float(amount / price) if price and price > 0 else np.nan
        post = int(when >= POLICY_AT)
        unins = 0 if _yn(claim["inspected"]) else 1
        small = int(amount < CAP_INR)
        hug = int(HUG_LOW <= amount < CAP_INR)
        row = {
            "claim_id": claim["claim_id"],
            "submitted_at": when,
            "partner_id": pid,
            "sku": claim["sku"],
            "amount": amount,
            "is_fraud": claim["is_fraud"],
            "ratio": ratio,
            "partner_rate": (prior_f + SHRINK_K * SHRINK_RATE) / (prior_d + SHRINK_K),
            "prior_f": prior_f,
            "prior_d": prior_d,
            "priors": int(np.clip(claim["prior_customer"], 0, 4)),
            "log_amt": float(np.log(amount)) if amount > 0 else 0.0,
            "nophoto": 0 if _yn(claim["photo"]) else 1,
            "unins": unins,
            "hug": hug,
            "hug30": hug30,
            "small30": small30,
            "known": int(prior_f >= 1),
            "burst": int(small30 >= 4),
            "cap_open": int(post == 1 and small == 1 and unins == 1),
            "post": post,
            "city": partner["city"] if partner else "",
            "onboarded_date": partner["onboarded_date"] if partner else "",
            "partner_type": partner["partner_type"] if partner else "",
            "family": product["family"] if product else "",
            "list_price_inr": price if price == price else None,
            "prior_customer": int(claim["prior_customer"]),
        }
        built.append(row)
        dq.append((when, hug, small))
        if claim["is_fraud"] is not None:
            decided_n[pid] += 1
            fraud_n[pid] += int(claim["is_fraud"] == 1)
    return built


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["y"] = pd.to_numeric(frame["is_fraud"], errors="coerce")
    return frame


def _fit(frame: pd.DataFrame, features: list[str]) -> dict:
    model = LogisticRegression(C=0.5, max_iter=800)
    scaler = StandardScaler()
    x = scaler.fit_transform(frame[features].astype(float))
    model.fit(x, frame["y"].astype(int))
    return {
        "features": features,
        "coef": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v if v else 1.0) for v in scaler.scale_],
        "n": int(len(frame)),
        "frauds": int(frame["y"].sum()),
    }


def _proba(rows: pd.DataFrame, spec: dict) -> np.ndarray:
    x = rows[spec["features"]].astype(float).to_numpy()
    mean = np.asarray(spec["mean"])
    scale = np.asarray(spec["scale"])
    z = (x - mean) / scale
    logit = z @ np.asarray(spec["coef"]) + spec["intercept"]
    return 1.0 / (1.0 + np.exp(-logit))


def policy_cap_score(frame: pd.DataFrame) -> np.ndarray:
    """What we would have watched on 1 May, before any new labels existed.

    The operations note says claims under Rs 2,000 skip inspection from
    that date. The weights are a reading of that rule, not a fit.
    """
    z = (
        1.6 * frame["cap_open"].to_numpy()
        + 1.0 * frame["hug"].to_numpy()
        + 0.22 * frame["small30"].to_numpy()
        + 1.0 * frame["known"].to_numpy()
        + 0.4 * frame["priors"].to_numpy()
        + 1.3 * frame["ratio"].to_numpy()
        + 0.8 * frame["burst"].to_numpy()
    )
    return 1.0 / (1.0 + np.exp(-(z - 3.4)))


def combine(when: pd.Series, classic: np.ndarray, cap: np.ndarray) -> np.ndarray:
    post = when >= POLICY_AT
    both = 1.0 - (1.0 - classic) * (1.0 - cap)
    return np.where(post.to_numpy(), both, classic)


def score_parts(frame: pd.DataFrame, artifact: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classic = _proba(frame, artifact["classic"])
    cap = _proba(frame, artifact["cap"])
    return classic, cap, combine(frame["submitted_at"], classic, cap)


def worth_a_slot(score: float, amount: float) -> bool:
    """Hold the claim when the rupees at stake beat the goodwill."""
    return score * amount > (1.0 - score) * GOODWILL_INR


def reasons_for(row: dict, score: float) -> list[str]:
    amount = float(row["amount"])
    ratio = float(row["ratio"]) if row["ratio"] == row["ratio"] else None
    risks = []
    if int(row["prior_f"]) >= 1 and int(row["prior_d"]) > 0:
        rate = int(row["prior_f"]) / int(row["prior_d"])
        place = row["city"] or "an unknown city"
        kind = (row["partner_type"] or "outlet").replace("_", " ")
        joined = row["onboarded_date"] or "an unknown date"
        risks.append(
            f"{row['partner_id']} in {place} ({kind}, joined {joined}) "
            f"already has {int(row['prior_f'])} confirmed fraud"
            f"{'' if int(row['prior_f']) == 1 else 's'} "
            f"in {int(row['prior_d'])} investigated claims ({rate:.0%})."
        )
    if int(row["cap_open"]) == 1:
        risks.append(
            f"Rs {amount:,.0f} is under the Rs 2,000 line, and the outlet did not inspect it. "
            "Since 1 May 2026 that claim is paid unless someone pulls it."
        )
    elif int(row["hug"]) == 1:
        risks.append(
            f"The amount is Rs {amount:,.0f}, just under the Rs 2,000 auto-approval line."
        )
    if int(row["small30"]) >= 4:
        risks.append(
            f"This outlet filed {int(row['small30'])} other claims under Rs 2,000 in the previous 30 days."
        )
    elif int(row["hug30"]) >= 2:
        risks.append(
            f"This outlet filed {int(row['hug30'])} other claims between Rs 1,900 and Rs 2,000 "
            "in the previous 30 days."
        )
    if ratio is not None and ratio >= 0.55:
        risks.append(
            f"The claim is {ratio:.0%} of the {row['family'] or 'product'} list price "
            f"(Rs {row['list_price_inr']:,.0f}). Most genuine repairs are under 40% of list."
        )
    if int(row["priors"]) >= 2:
        risks.append(
            f"This customer already has {int(row['prior_customer'])} earlier warranty claims."
        )
    held = worth_a_slot(score, amount)
    at_stake = score * amount
    if held:
        if not risks:
            risks.append("The rupees at stake clear the goodwill of holding a genuine customer.")
        return risks[:4]
    if risks:
        return [
            (
                f"Pay it. The chance is {score:.1%}, so about Rs {at_stake:,.0f} is at stake. "
                f"That does not beat the Rs {GOODWILL_INR:,.0f} goodwill of delaying a genuine repair."
            ),
            risks[0],
        ]
    usual = f"{ratio:.0%} of list price" if ratio is not None else "in the usual range for a repair"
    lines = [f"Pay it. Nothing in this outlet's investigated history stands out, and the claim is {usual}."]
    if int(row["post"]) == 1 and amount < CAP_INR:
        lines.append("It is under Rs 2,000, and this outlet has not been filing a run of those claims.")
    return lines


def train_artifact(train: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame) -> dict:
    kept, _extra = dedupe_train(train)
    plook = partner_lookup(partners)
    klook = product_lookup(products)
    featured = _frame(walk_features(claims_from_frame(kept, labeled=True), plook, klook))
    classic_rows = featured[(featured["submitted_at"] < POLICY_AT) & featured["y"].notna()]
    cap_rows = featured[(featured["submitted_at"] >= POLICY_AT) & featured["y"].notna()]
    return {
        "goodwill_inr": GOODWILL_INR,
        "contact_inr": CONTACT_INR,
        "cap_inr": CAP_INR,
        "policy_date": str(POLICY_AT.date()),
        "desk_slots": DESK_SLOTS,
        "shrink_k": SHRINK_K,
        "shrink_rate": SHRINK_RATE,
        "classic": _fit(classic_rows, CLASSIC_FEATURES),
        "cap": _fit(cap_rows, CAP_FEATURES),
        "partners": plook,
        "products": klook,
    }


def attach_scores(featured: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    out = featured.copy()
    classic, cap, score = score_parts(out, artifact)
    out["p_classic"] = classic
    out["p_cap"] = cap
    out["score"] = score
    out["review"] = [
        worth_a_slot(float(s), float(a)) for s, a in zip(out["score"], out["amount"], strict=True)
    ]
    return out


def queue_stats(scored: pd.DataFrame, k: int = DESK_SLOTS) -> dict:
    """Take the k highest scores. Report rupees, goodwill, and accuracy."""
    known = scored[scored["y"].notna()].copy()
    known["y"] = known["y"].astype(int)
    if known.empty or known["y"].nunique() < 2:
        return {}
    order = known.sort_values(["score", "claim_id"], ascending=[False, True]).head(k)
    tp = int(order["y"].sum())
    fp = int(k - tp)
    caught = float(order.loc[order["y"] == 1, "amount"].sum())
    fraud_rs = float(known.loc[known["y"] == 1, "amount"].sum())
    fraud_n = int(known["y"].sum())
    n = int(len(known))
    # Accuracy if the queue is "fraud" and everything else is "not".
    correct = (n - fraud_n - fp) + tp
    acc_queue = correct / n
    acc_pay_all = (n - fraud_n) / n
    return {
        "n": n,
        "fraud_n": fraud_n,
        "fraud_rs": fraud_rs,
        "tp": tp,
        "fp": fp,
        "caught_rs": caught,
        "goodwill_rs": fp * GOODWILL_INR,
        "net_rs": caught - fp * GOODWILL_INR,
        "per_check_rs": caught / k,
        "net_per_check_rs": (caught - fp * GOODWILL_INR) / k,
        "after_contact_rs": caught - fp * GOODWILL_INR - k * CONTACT_INR,
        "acc_queue": acc_queue,
        "acc_pay_all": acc_pay_all,
        "auc": float(_auc(known["y"].to_numpy(), known["score"].to_numpy())),
        "ap": float(_ap(known["y"].to_numpy(), known["score"].to_numpy())),
    }


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _ap(y, p) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def monthly_backtest(featured: pd.DataFrame) -> list[dict]:
    """Each month is scored with a model that has not seen that month."""
    labeled = featured[featured["y"].notna()].copy()
    months = sorted(labeled["submitted_at"].dt.to_period("M").unique())
    rows = []
    for month in months:
        if str(month) < "2025-12":
            continue
        start = month.to_timestamp()
        end = start + pd.offsets.MonthBegin(1)
        block = labeled[(labeled["submitted_at"] >= start) & (labeled["submitted_at"] < end)]
        if block["y"].nunique() < 2:
            continue
        classic_train = labeled[(labeled["submitted_at"] < min(start, POLICY_AT))]
        cap_train = labeled[(labeled["submitted_at"] < start) & (labeled["submitted_at"] >= POLICY_AT)]
        classic_p = _proba(block, _fit(classic_train, CLASSIC_FEATURES))
        if int(cap_train["y"].sum()) >= 4:
            cap_p = _proba(block, _fit(cap_train, CAP_FEATURES))
            cap_how = "fit"
        else:
            cap_p = policy_cap_score(block)
            cap_how = "policy"
        # Before May the cap rule is not in force. Do not let it vote.
        if start < POLICY_AT:
            score = classic_p
            cap_how = "off"
        else:
            score = 1.0 - (1.0 - classic_p) * (1.0 - cap_p)
        trial = block.copy()
        trial["score"] = score
        stats = queue_stats(trial)
        stats["month"] = str(month)
        stats["cap_how"] = cap_how
        rows.append(stats)
    return rows


def build_scored_file(data_dir: Path = DATA) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    train, test, partners, products = load_tables(data_dir)
    kept, extra_rows = dedupe_train(train)
    artifact = train_artifact(kept, partners, products)
    plook = artifact["partners"]
    klook = artifact["products"]
    history = claims_from_frame(kept, labeled=True) + claims_from_frame(test, labeled=False)
    featured = _frame(walk_features(history, plook, klook))
    scored = attach_scores(featured, artifact)
    train_ids = set(kept["claim_id"].astype(str))
    scored_train = scored[scored["claim_id"].isin(train_ids)].copy()
    test_order = test["claim_id"].astype(str).tolist()
    scored_test = scored[scored["claim_id"].isin(test_order)].copy()
    scored_test["_order"] = scored_test["claim_id"].map({cid: i for i, cid in enumerate(test_order)})
    scored_test = scored_test.sort_values("_order")
    audit = {
        "train_rows_raw": int(len(train)),
        "train_rows_kept": int(len(kept)),
        "resubmit_rows": extra_rows,
        "open_investigations": int(kept["is_fraud"].isna().sum()),
        "confirmed_fraud": int((kept["is_fraud"] == 1).sum()),
        "test_rows": int(len(test)),
    }
    return artifact, scored_train, scored_test, audit


def prediction_frame(scored_test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": scored_test["claim_id"].to_numpy(),
            "score": np.round(scored_test["score"].to_numpy(), 6),
        }
    )


def save_artifact(artifact: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact))


def load_artifact(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text())


def score_one(record: dict, artifact: dict, earlier: list[dict]) -> dict:
    """Score a single claim. `earlier` must be claims strictly before it."""
    claim = {
        "claim_id": str(record.get("claim_id") or "NEW"),
        "submitted_at": _stamp(record["submitted_at"]),
        "partner_id": str(record["partner_id"]).strip(),
        "sku": str(record["sku"]).strip(),
        "amount": float(record["claim_amount_inr"]),
        "photo": record["photo_attached"],
        "inspected": record["partner_inspected"],
        "prior_customer": int(record["customer_prior_claims"]),
        "is_fraud": None,
    }
    if claim["sku"] not in artifact["products"]:
        raise ValueError(
            f"Unknown SKU {claim['sku']}. The price list has no such product, so the claim was not scored."
        )
    if claim["amount"] <= 0:
        raise ValueError("Claim amount has to be above zero.")
    if claim["partner_id"] not in artifact["partners"]:
        raise ValueError(
            f"Unknown outlet {claim['partner_id']}. It is not on the partner list, so the claim was not scored."
        )
    featured = _frame(walk_features(earlier + [claim], artifact["partners"], artifact["products"]))
    row = featured.iloc[-1]
    if row["claim_id"] != claim["claim_id"]:
        raise RuntimeError("Scoring aligned to the wrong claim.")
    frame = featured.iloc[[-1]].copy()
    classic, cap, score = score_parts(frame, artifact)
    score_v = float(score[0])
    amount = claim["amount"]
    review = worth_a_slot(score_v, amount)
    payload = row.to_dict()
    payload["prior_customer"] = claim["prior_customer"]
    text = reasons_for(payload, score_v)
    expected = score_v * amount
    goodwill = (1.0 - score_v) * GOODWILL_INR
    return {
        "claim_id": claim["claim_id"],
        "score": round(score_v, 6),
        "decision": "review" if review else "pay",
        "expected_fraud_inr": round(expected, 2),
        "goodwill_if_genuine_inr": round(goodwill, 2),
        "net_if_held_inr": round(expected - goodwill, 2),
        "p_classic": round(float(classic[0]), 6),
        "p_cap": round(float(cap[0]), 6),
        "reasons": text,
        "outlet": {
            "partner_id": claim["partner_id"],
            "city": payload["city"],
            "partner_type": payload["partner_type"],
            "onboarded_date": payload["onboarded_date"],
            "confirmed_frauds": int(payload["prior_f"]),
            "investigated_claims": int(payload["prior_d"]),
        },
        "claim": {
            "amount_inr": amount,
            "sku": claim["sku"],
            "family": payload["family"],
            "list_price_inr": payload["list_price_inr"],
            "share_of_list": None if payload["ratio"] != payload["ratio"] else round(float(payload["ratio"]), 3),
        },
    }


def history_before(record: dict, claims: list[dict]) -> list[dict]:
    when = _stamp(record["submitted_at"])
    cid = str(record.get("claim_id") or "NEW")
    return [c for c in claims if (c["submitted_at"], c["claim_id"]) < (when, cid)]
