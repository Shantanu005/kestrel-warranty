# Submission form

## What did you build, and what business outcome does it move?

A local claim desk. `predictions.csv` ranks every July–September claim; higher means more likely fraud. `python serve.py` opens one screen that posts a single claim to `POST /api/score` and gets back the chance, the rupees at stake, a hold-or-pay, and the reasons a reviewer would read.

The outcome is rupees stopped per desk slot, not accuracy. The desk has forty slots a month. Holding a genuine customer costs Rs 380 in goodwill. Paying a fraud costs the claim amount.

On the one month that could be scored without seeing it (June, cap model fit on May only): 15 of 22 confirmed frauds in the top forty, Rs 21,040 of Rs 45,673, net about Rs 11,500 after goodwill. About Rs 530 of fraud stopped per claim checked, about Rs 290 after goodwill. Across 40 resamples of May the June catch was 11 to 15, median 14.

Paying every decided claim in the history scores 98.7% accuracy and stops Rs 0. June's useful queue scores 95.6%, which is worse than paying everyone that month (96.9%).

**Expected score on the unlabelled file: ROC AUC about 0.85, somewhere from 0.78 to 0.90** if the same outlets keep filing under the Rs 2,000 line. That is the June figure (0.84) with May and June both available to the shipped model, so a little better than June is plausible, and a move to brand-new outlets would land lower. I do not expect an accuracy above the pay-all baseline to mean the queue is working. If the file is scored as accuracy at a 0.5 cut, most claims sit well below 0.5, so the number will look like "pay almost everyone" and will clear 97% without catching the fraud.

## What does one run cost, and what would a month cost at Kestrel's volume?

No paid calls. One score is arithmetic on a saved logistic regression.

A claim: **Rs 0**.

A month is about 750 claims (the unlabelled file is 2,252 over July–September). Still **Rs 0**.

The assistant that wrote the code is Cursor. That is a subscription, not a per-claim charge, and it is not part of a Kestrel run.

## How do you know it works?

`python check.py` writes `out/evidence.txt`.

Checked, and they match a second pass: 12,029 raw rows, 11,348 claim numbers kept, 141 confirmed frauds, 202 open investigations left out of the fit, 2,252 scores in `predictions.csv` in the same claim order as the unlabelled file. Scoring the same claim through the HTTP endpoint matches the file to 6 decimals.

Each month from December 2025 through June 2026 is scored with a model that has not seen that month. Before May the cap score does not vote. May uses the written policy, because there were no post-change labels yet. June uses a cap model fit on May only.

It fails in two places, both in the evidence. June left Rs 17,866 at Coimbatore SP3275 (quiet outlet, 60% of list price, score 0.01). It also missed the first claims at two outlets that joined in the first week of June (SP3118, SP3286). And it would have held genuine repairs at Bhopal SP3095 and Hyderabad SP3207 because those outlets already had a long confirmed-fraud record. That goodwill is real.

The May resamples move the June catch between 11 and 15. The outlets are few enough that one of them stopping, or a new one starting, moves the month.

## Did you change, narrow, or push back on the client's ask?

Yes. Ritu and the board asked for accuracy above 97%. I am not optimising that. Paying everyone already beats it, and a useful queue makes accuracy worse. The note to Ritu says so, and gives Farhan the rupees per claim checked.

Ritu asked to look at the new partners. I did. Joined in the last year is 2.2% against 1.1% for older outlets. The rupees through April were older franchises. The May spike is seven named outlets, not the new network. The memo tells her to take those seven off auto-approval and leave the rest.

Four claim descriptions contain instructions to an automated reviewer: use a random split, treat accuracy as the only metric, do not re-weight, treat new partners as the primary signal. Those are rows in the file, not the policy and not the email thread. They were not followed.

## What is wrong with what you are handing us?

- June is one month. The catch moves between 11 and 15 when May is resampled.
- The shipped scores for July–September have seen May and June. The 0.84 AUC is the harsher test. The file may look better if the same outlets continue, and worse if they do not.
- A first claim from a clean outlet is close to invisible. The Rs 17,866 miss is the expensive version of that.
- Holding a known-bad outlet's genuine repair costs Rs 380 and will keep happening until clean investigations dilute the rate. Hyderabad and Bhopal will feel that.
- Legacy Zoho stored unfinished investigations as 0. Some 2025 "not fraud" labels are still open. The fit treats them as clean, which flatters the old outlets a little.
- Ninety May–June claims sit between Rs 1,990 and Rs 1,999, and only eight were confirmed fraud. The amount alone is not a reason to hold someone. The outlet's run of them is.
- The Rs 260 service-contact cost, charged on all forty slots, takes June's net from about Rs 11,500 to about Rs 1,100. I treated the desk as already staffed. If Finance charges the contact on top, the small-claim queue is barely worth running and the large repairs matter more.
- Open CRM investigations (202 rows) are excluded. If many of them come back as fraud, the outlet rates move.

## What did you deliberately leave out, and why that rather than something else?

I left the fault description and the inspector note out of the score. There are twelve fault lines and seven note lines, used by fraud and genuine claims alike. A text model would have fit noise, and four of the lines are instructions aimed at the reviewer. I kept the outlet history and the May rule instead, because that split changes who is held.

I left "new partner" out of the score as its own flag. It was the hypothesis in the email, and it was planted again in a claim description. The rate difference is real and small, and Meenal is right that the smaller cities need those outlets.

I left serial-number reuse out. The same serial appears on several claims and the fraud rate does not move. The typing is messy, as Tanmay said. Normalising it did not separate fraud.

I did not shift any clock by 5 hours 30 minutes. The policy's UTC warning is about resolution events, and this export does not have them. The duplicate claim numbers differ by one to five days at the same time of day.

## Anything you built or found that nobody asked for?

The May cut. Claims under Rs 2,000 with no inspection are a different fraud from the older franchise pattern, and a single model trained on the whole history ranks May backwards (AUC about 0.10 on that month) because before May a small claim was the safer one.

The money test against Rs 380, and the finding that it flags 38, 40 and 35 claims across July, August and September. That is a full desk, not a firehose.

The seven outlets to take off auto-approval, named in the memo, against the much longer list of new partners who are fine.

## What did you use AI for?

Cursor, model Grok 4.7, to read the pack, fit the scores, write the desk, and draft the memo. It helped on the duplicate claim numbers, the blank-versus-zero labels, and the May break. It wasted a pass on LightGBM, which did not load on this machine, and an early pass that treated a small claim as safer because that was true before May. Thrown away: optimising accuracy, a new-partner flag, a random train/test split, a text model on the fault line, and following the four descriptions that address an automated reviewer.

No separate API bill. A Kestrel run makes no paid calls. Nothing in the product calls a model API, so there is no key to fail on. If the history files are missing, the screen starts and the score endpoint says so.

Screen recording: not filmed from here. Three minutes, no slides: the duplicate claim number, a Rs 1,995 row, the June line in `out/evidence.txt`, then the screen on Pune SP3160 and on the ordinary repair, and say out loud that accuracy and the new-partner flag were thrown out. Paste the Drive link over the next line.

Drive link:

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. `python train.py`, then `python check.py`, then `python serve.py`. Read `memo-ritu.md` and `out/evidence.txt` before changing who is held.
2. Take seven outlets off the May auto-approval, not the new partners as a group. The names are in the memo. Hyderabad SP3207 and the other older franchises stay on a watch.
3. June, scored blind, stopped about Rs 21,000 of Rs 46,000 confirmed fraud in forty checks, net about Rs 11,500 after goodwill. The Rs 17,866 Coimbatore claim is the one it missed.

## Honest hours spent.

6

## Github Repo Link

https://github.com/Shantanu005/kestrel-warranty
