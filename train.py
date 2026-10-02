"""Fit the two scores and write predictions.csv for the unlabelled claims."""

from pathlib import Path

from kestrel import MODEL_PATH, ROOT, build_scored_file, prediction_frame, save_artifact

OUT = ROOT / "predictions.csv"


def main() -> None:
    artifact, _train, scored_test, audit = build_scored_file()
    save_artifact(artifact, MODEL_PATH)
    frame = prediction_frame(scored_test)
    frame.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(frame)} rows)")
    print(f"Wrote {MODEL_PATH}")
    print(
        "Train rows kept {train_rows_kept} of {train_rows_raw}. "
        "Confirmed fraud {confirmed_fraud}. Open investigations left out of the fit: {open_investigations}.".format(
            **audit
        )
    )


if __name__ == "__main__":
    main()
