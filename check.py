"""Recompute the queue on months the model has not seen, and write the evidence."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from kestrel import (
    CONTACT_INR,
    DATA,
    DESK_SLOTS,
    GOODWILL_INR,
    POLICY_AT,
    ROOT,
    build_scored_file,
    dedupe_train,
    load_tables,
    monthly_backtest,
    worth_a_slot,
    _fit,
    _proba,
)

OUT = ROOT / "out" / "evidence.txt"


def _money(value: float) -> str:
    return f"Rs {value:,.0f}"


def _poison_count(train: pd.DataFrame) -> int:
    text = train["claim_description"].fillna("")
    mask = text.str.contains(r"board-KPI|automated review|do not re-weight|primary signal", case=False)
    return int(mask.sum())


def _resample_june(featured: pd.DataFrame, draws: int = 40) -> list[int]:
    """Refit the May cap score on resamples, then count June frauds in the top 40."""
    labeled = featured[featured["y"].notna()].copy()
    may = labeled[(labeled["submitted_at"] >= POLICY_AT) & (labeled["submitted_at"] < "2026-06-01")]
    june = labeled[(labeled["submitted_at"] >= "2026-06-01") & (labeled["submitted_at"] < "2026-07-01")]
    classic_train = labeled[labeled["submitted_at"] < POLICY_AT]
    classic = _proba(june, _fit(classic_train, ["ratio", "partner_rate", "priors", "log_amt", "nophoto", "unins"]))
    rng = np.random.default_rng(42)
    caught = []
    may_idx = may.index.to_numpy()
    for _ in range(draws):
        take = rng.choice(may_idx, size=len(may_idx), replace=True)
        sample = may.loc[take]
        if sample["y"].nunique() < 2 or int(sample["y"].sum()) < 4:
            continue
        cap = _proba(june, _fit(sample, ["cap_open", "hug", "small30", "known", "priors", "ratio", "burst", "nophoto"]))
        score = 1.0 - (1.0 - classic) * (1.0 - cap)
        trial = june.copy()
        trial["score"] = score
        order = trial.sort_values(["score", "claim_id"], ascending=[False, True]).head(DESK_SLOTS)
        caught.append(int(order["y"].sum()))
    return caught


def main() -> None:
    train, test, partners, products = load_tables(DATA)
    kept, extra = dedupe_train(train)
    artifact, scored_train, scored_test, audit = build_scored_file(DATA)
    featured = scored_train.copy()
    months = monthly_backtest(featured)
    resampled = _resample_june(featured)

    # In-sample July-September is not available. The last honest month is June,
    # scored from a cap model fit on May only. That row is inside `months`.
    by_month = {row["month"]: row for row in months}
    june = by_month["2026-06"]
    may = by_month["2026-05"]

    # Where the confirmed fraud sits.
    frauds = kept[kept["is_fraud"] == 1].merge(partners, on="partner_id")
    counts = frauds.groupby(["partner_id", "city", "partner_type", "onboarded_date"]).size()
    counts = counts.sort_values(ascending=False)
    top10 = int(counts.head(10).sum())
    top_lines = []
    for (pid, city, kind, joined), n in counts.head(8).items():
        top_lines.append(f"  {pid}  {city}  {kind}  joined {joined}  {int(n)} confirmed")

    post = frauds[pd.to_datetime(frauds["submitted_at"]) >= POLICY_AT]
    post_counts = (
        post.groupby(["partner_id", "city"]).size().sort_values(ascending=False)
    )
    post_lines = [f"  {pid}  {city}  {int(n)}" for (pid, city), n in post_counts.items()]

    merged = kept.merge(partners, on="partner_id")
    merged["submitted_at"] = pd.to_datetime(merged["submitted_at"])
    merged["onboarded_date"] = pd.to_datetime(merged["onboarded_date"])
    decided = merged[merged["is_fraud"].notna()]
    new = decided["onboarded_date"] >= decided["submitted_at"] - pd.Timedelta(days=365)
    rate_new = float((decided.loc[new, "is_fraud"] == 1).mean())
    rate_old = float((decided.loc[~new, "is_fraud"] == 1).mean())

    # June misses and false holds, from the same backtest path.
    labeled = featured[featured["y"].notna()].copy()
    start = pd.Timestamp("2026-06-01")
    block = labeled[(labeled["submitted_at"] >= start) & (labeled["submitted_at"] < "2026-07-01")].copy()
    # Rebuild June scores the way monthly_backtest does, so the examples match the table.
    from kestrel import CAP_FEATURES, CLASSIC_FEATURES, combine

    classic_train = labeled[labeled["submitted_at"] < POLICY_AT]
    cap_train = labeled[(labeled["submitted_at"] < start) & (labeled["submitted_at"] >= POLICY_AT)]
    classic_p = _proba(block, _fit(classic_train, CLASSIC_FEATURES))
    cap_p = _proba(block, _fit(cap_train, CAP_FEATURES))
    block["score"] = combine(block["submitted_at"], classic_p, cap_p)
    ranked = block.sort_values(["score", "claim_id"], ascending=[False, True])
    top = ranked.head(DESK_SLOTS)
    missed = ranked[(ranked["y"] == 1) & ~ranked["claim_id"].isin(top["claim_id"])]
    held = top[top["y"] == 0].sort_values("amount", ascending=False)

    def brief(frame: pd.DataFrame, limit: int) -> list[str]:
        lines = []
        for rec in frame.head(limit).itertuples(index=False):
            lines.append(
                f"  {rec.claim_id}  {rec.partner_id}  {rec.city}  Rs {rec.amount:,.0f}  "
                f"score {rec.score:.3f}  share {rec.ratio:.0%}  "
                f"prior frauds {int(rec.prior_f)}  small-claims-30d {int(rec.small30)}"
            )
        return lines

    # Do-nothing accuracy on decided, deduped train.
    decided_n = int(kept["is_fraud"].notna().sum())
    fraud_n = int((kept["is_fraud"] == 1).sum())
    pay_all = (decided_n - fraud_n) / decided_n

    # Shipped scores on the unlabelled file: how many clear the money test.
    clear = [
        worth_a_slot(float(s), float(a))
        for s, a in zip(scored_test["score"], scored_test["amount"], strict=True)
    ]
    by_m = scored_test.copy()
    by_m["month"] = by_m["submitted_at"].dt.to_period("M").astype(str)
    by_m["clear"] = clear
    clear_by_month = by_m.groupby("month")["clear"].sum().to_dict()

    # Amounts sitting on the cap, post May, in the investigated file.
    post_claims = kept[pd.to_datetime(kept["submitted_at"]) >= POLICY_AT]
    near = post_claims["claim_amount_inr"].between(1990, 1999)
    near_fraud = int(((post_claims["is_fraud"] == 1) & near).sum())
    near_n = int(near.sum())

    # Test file, same amount.
    test_near = int(test["claim_amount_inr"].between(1990, 1999).sum())

    lines = []
    w = lines.append
    w("Kestrel warranty queue — evidence")
    w("Recomputed by python check.py from data/. The June row is the one that matters.")
    w("")
    w("Labels")
    w(f"  Raw train rows: {audit['train_rows_raw']}")
    w(f"  Kept one row per claim number: {audit['train_rows_kept']} (dropped {extra} resubmit copies)")
    w(f"  Confirmed fraud: {audit['confirmed_fraud']}")
    w(f"  Open investigations, blank in the CRM, left out of the fit: {audit['open_investigations']}")
    w("  Legacy Zoho rows have no blanks. Open cases there were stored as 0, so a few")
    w("  'not fraud' labels in 2025 are unfinished. The fit treats them as not fraud.")
    w(f"  Paying every decided claim scores {pay_all:.1%} accuracy and stops Rs 0.")
    w(
        f"  Rows whose description tries to instruct an automated reviewer: {_poison_count(train)} "
        "(4 claim numbers; one was resubmitted). They were not followed."
    )
    w("")
    w("New outlets are not the class")
    w(f"  Outlet joined in the last year: {rate_new:.1%} confirmed fraud")
    w(f"  Older outlet: {rate_old:.1%}")
    w(f"  Ten outlets account for {top10} of {fraud_n} confirmed frauds.")
    w("  Largest confirmed counts:")
    lines.extend(top_lines)
    w("")
    w("After 1 May 2026 the confirmed frauds are these outlets:")
    lines.extend(post_lines)
    w(f"  Claims filed at Rs 1,990–1,999 in May–June: {near_n}, of which {near_fraud} were confirmed fraud.")
    w(f"  The unlabelled July–September file has {test_near} claims in that same band.")
    w("")
    w(f"Desk of {DESK_SLOTS} a month. Goodwill for holding a genuine customer: {_money(GOODWILL_INR)}.")
    w(f"A service contact is {_money(CONTACT_INR)}; the desk is already staffed, so the headline net")
    w("does not subtract it. The last column does, as a sensitivity.")
    w("")
    w("Month     how      AUC    caught   of fraud-Rs    net after goodwill   accuracy if queued   accuracy if paid all")
    for row in months:
        w(
            f"{row['month']}  {row['cap_how']:<7}  {row['auc']:.3f}  "
            f"{_money(row['caught_rs']):>10}  {_money(row['fraud_rs']):>12}  "
            f"{_money(row['net_rs']):>12}   {row['acc_queue']:.1%}   {row['acc_pay_all']:.1%}"
            f"   tp {row['tp']} of {row['fraud_n']}"
        )
    w("")
    w("June is the honest test of the new rule: the cap score was fit on May only.")
    w(
        f"  Top 40 caught {june['tp']} of {june['fraud_n']} confirmed frauds, "
        f"{_money(june['caught_rs'])} of {_money(june['fraud_rs'])}."
    )
    w(
        f"  Goodwill on the {june['fp']} genuine claims held: {_money(june['goodwill_rs'])}. "
        f"Net {_money(june['net_rs'])}, about {_money(june['per_check_rs'])} of fraud stopped per claim checked, "
        f"{_money(june['net_per_check_rs'])} after goodwill."
    )
    w(
        f"  If the Rs {CONTACT_INR} contact is charged on every slot as well, June net is {_money(june['after_contact_rs'])}."
    )
    w(
        f"  Queue accuracy {june['acc_queue']:.1%}. Paying everyone that month scores {june['acc_pay_all']:.1%}."
    )
    w("  May had no post-change labels yet, so that month uses the written policy, not a fit.")
    w(
        f"  May top 40: {may['tp']} of {may['fraud_n']}, {_money(may['caught_rs'])} of {_money(may['fraud_rs'])}, "
        f"net {_money(may['net_rs'])}."
    )
    w("")
    if resampled:
        w(
            f"June catch count across {len(resampled)} resamples of May (seed 42): "
            f"min {min(resampled)}, median {int(np.median(resampled))}, max {max(resampled)}. "
            f"The point estimate of {june['tp']} is not a promise. "
            f"Spread: {dict(sorted(Counter(resampled).items()))}."
        )
        w("")
    w("June claims the queue missed:")
    lines.extend(brief(missed.sort_values("amount", ascending=False), 8))
    w("Largest genuine June claims the queue would have held:")
    lines.extend(brief(held, 5))
    w("")
    w("Unlabelled July–September file, scored with May and June both in the fit.")
    w(f"  Rows: {len(scored_test)}. Score min {scored_test['score'].min():.4f}, max {scored_test['score'].max():.4f}.")
    w("  Claims that clear the money test (chance × rupees > goodwill), by month:")
    for month, n in clear_by_month.items():
        w(f"    {month}: {int(n)}")
    w("  That count can exceed 40. The desk takes the 40 with the most rupees at stake.")
    w("")
    w("What this file does not prove: the July–September outcomes. Those are not in the pack.")
    w("Expectation written down before those outcomes are known: ROC AUC about 0.85,")
    w("in a range of roughly 0.78 to 0.90 if the same outlets keep filing under the cap.")
    w("A first month at a brand-new outlet, and a single large claim from a quiet outlet, will be missed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    print(text)
    summary = {
        "june": june,
        "may": may,
        "months": months,
        "resample_june_tp": resampled,
        "pay_all_accuracy": pay_all,
        "clear_by_month": {k: int(v) for k, v in clear_by_month.items()},
        "rate_new": rate_new,
        "rate_old": rate_old,
        "top10": top10,
        "fraud_n": fraud_n,
        "near_cap_may_jun": {"n": near_n, "fraud": near_fraud},
        "test_near_cap": test_near,
    }
    (ROOT / "out" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
