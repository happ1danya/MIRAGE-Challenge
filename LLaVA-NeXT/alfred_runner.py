# alfred_runner.py
import json, csv, argparse, time
from pathlib import Path
from predict import Predictor          # your class; keep filename identical

# -------------------------------------------------------------------------
# 1. Build a concise, single-action prompt for each frame
def build_prompt(context: str) -> str:
    """
    Turn ALFRED context into one explicit low-level command.
    """
    return (
        "You are a household-robot assistant. "
        "Given the main goal and current frame, reply with exactly **one** "
        "low-level action in imperative form, e.g. "
        "'MoveAhead', 'TurnRight', 'PickupObject[spoon]', 'ToggleOn'.\n\n"
        "Context: " + context.strip()
    )

# -------------------------------------------------------------------------
# 2. Run all samples
def run(model: Predictor, ann_path: Path, img_root: Path, out_csv: Path):
    data = json.load(ann_path.open())["annotations" if "annotations" in
                                      json.load(ann_path.open()) else "data"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "prediction"])

        for samp in data:
            sid   = samp["sample_id"]
            ctx   = samp["task_instance"]["context"]
            img_p = img_root / samp["task_instance"]["images_path"][0]

            t0 = time.time()
            stream = model.predict(
                image  = img_p,
                prompt = build_prompt(ctx),
                temperature = 0.0,
                top_p = 1.0,
                max_tokens = 16   # one short action
            )
            reply = "".join(list(stream)).strip()
            print(f"{sid:>4}: {reply:<35}  ({time.time()-t0:4.1f}s)")
            writer.writerow([sid, reply])

# -------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",    type=Path, default=Path("test/test.json"))
    ap.add_argument("--img-dir", type=Path, default=Path("test"))
    ap.add_argument("--out",     type=Path, default=Path("alfred_predictions.csv"))
    args = ap.parse_args()

    predictor = Predictor()
    predictor.setup()            # loads LLaVA weights once (first run ~2-3 min)

    run(predictor, args.json, args.img_dir, args.out)
    print("\nDone! Predictions written to", args.out.resolve())
