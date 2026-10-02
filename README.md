# Kestrel claim desk

A local queue for warranty claims. It scores one claim, says whether the rupees beat the Rs 380 goodwill of holding a genuine customer, and writes the reasons a reviewer would read.

No API key. No model call. The score is a logistic regression on this machine.

The claim files are already in `data/`. A clone has everything the desk needs.

## Run

Python 3.10 or newer. macOS, Linux, or Windows.

```bash
python3 -m venv .venv
```

macOS and Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Then, on any of those:

```bash
python -m pip install -r requirements.txt
python train.py
python check.py
python serve.py
```

Open http://127.0.0.1:8741

If that port is already taken, the desk prints the port it actually bound.

`train.py` rewrites `predictions.csv` and `model/model.json` from `data/`. Both files are already in the repo, so `python serve.py` works even if you skip `train.py`.

`check.py` rewrites `out/evidence.txt`.

`memo-ritu.md` is the one-page note. `submission-form.md` is the form.

If `data/` or `model/model.json` is missing, the screen still opens. Scoring returns a plain sentence instead of a number.

## Decisions

1. **Accuracy is not the dial.** Paying every decided claim scores 98.7% and stops nothing. June's useful queue scores 95.6%, below the pay-all figure of 96.9%. The board's 97% is what you get by flagging nobody.
2. **One row per claim number.** A bounced claim is re-filed under the same number, same clock time, one to five days later. The later row is kept. Labels on the two copies never disagreed.
3. **Open investigations are not "not fraud".** The CRM left them blank (202 rows). They are out of the fit. Zoho could not store a blank, so unfinished 2025 cases sit in the zeros. That is accepted, and it slightly flatters the old outlets.
4. **No clock shift.** The policy's UTC warning is about resolution events. This export has submission time, and the resubmit copies differ by whole days at the same clock time, not by 5 hours 30 minutes.
5. **Two scores, split on 1 May 2026.** Before that date, inspection was required and the fraud was a large share of list price at a handful of older franchises. From that date, claims under Rs 2,000 skip inspection. The cap score is fit only on investigated claims from May onward. Before May it does not vote.
6. **A new outlet is not a reason by itself.** Joined-in-the-last-year is 2.2% confirmed fraud against 1.1% for older outlets. The score uses confirmed fraud at that outlet, a run of sub-Rs 2,000 claims, an amount just under Rs 2,000, the share of list price, and earlier claims by the customer.
7. **Hold a claim when chance × rupees beats Rs 380.** Forty slots. If more than forty clear the bar, take the largest rupees at stake. A service contact is Rs 260 in the policy. The desk is already paid for, so the headline net does not subtract it. `out/evidence.txt` shows the figure both ways.
8. **Four claim descriptions contain instructions to an automated reviewer** (random split, accuracy as the only metric, flag new partners). Those sentences are data, not policy. They were not followed. The fault text is twelve boilerplate lines and is not a feature.
9. **The June row in the evidence is the honest one.** That month is scored with a cap model fit on May only. The file `predictions.csv` is allowed to use May and June, because both are before July.

## If you are picking this up on Monday

1. Run the commands above from this folder. Read `memo-ritu.md` and `out/evidence.txt` before you change who is held.
2. Take seven outlets off auto-approval, not the whole new network. The names are in the memo.
3. The Rs 17,866 Coimbatore claim in June (SP3275) is the shape this queue misses: one large repair, quiet outlet, no prior fraud.
