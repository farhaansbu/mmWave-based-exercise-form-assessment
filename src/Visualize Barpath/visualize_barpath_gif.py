import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# ============================================================
#   PHYSICALLY-INFORMED BAR TRACKER
# ============================================================

class PhysicallyInformedBarTracker:

    def __init__(self, w_dist=0.6, w_vel=0.2, w_smooth=0.15, w_y=0.05, w_snr=0.05):
        self.prev_x = None
        self.prev_v = None
        self.weights = (w_dist, w_vel, w_smooth, w_y, w_snr)

    def score(self, obj, prev_x, prev_v):
        x, y, v, snr = obj["x"], obj["y"], obj["v"], obj["snr"]

        temporal = 0 if prev_x is None else abs(x - prev_x)

        if prev_x is None:
            vel_mismatch = 0
        else:
            dx = x - prev_x
            vel_mismatch = 0 if np.sign(dx) == np.sign(v) else 1

        acc = 0 if prev_v is None else abs(v - prev_v)
        y_penalty = abs(y)
        snr_reward = snr

        w1, w2, w3, w4, w5 = self.weights

        return (
            w1 * temporal +
            w2 * vel_mismatch +
            w3 * acc +
            w4 * y_penalty -
            w5 * snr_reward
        )

    def pick(self, frame_df):
        if len(frame_df) == 0:
            return None, None, None

        best = None
        best_score = float("inf")

        for _, row in frame_df.iterrows():
            s = self.score(row, self.prev_x, self.prev_v)
            if s < best_score:
                best_score = s
                best = row

        self.prev_x = best["x"]
        self.prev_v = best["v"]

        return best["x"], best["y"], best["v"]


# ============================================================
#   SYNTHETIC HEIGHT GENERATOR
# ============================================================

def synthetic_height(n_frames, baseline=1.0, amplitude=0.4):
    t = np.linspace(0, np.pi, n_frames)
    return baseline - amplitude * np.sin(t)


# ============================================================
#   GIF CREATOR (TRUE vs PRED LABEL INCLUDED)
# ============================================================

def create_barpath_gif(rep_df, outpath, true_label, pred_label):
    tracker = PhysicallyInformedBarTracker()

    frames = sorted(rep_df.frame.unique())
    xs = []

    for f in frames:
        frame_df = rep_df[rep_df.frame == f]
        x, y, v = tracker.pick(frame_df)
        xs.append(x)

    ys = synthetic_height(len(xs))

    # ========= CONSISTENT X-AXIS FOR FULL REP =========
    margin = 0.1
    xmin = rep_df["x"].min() - margin
    xmax = rep_df["x"].max() + margin

    ymin = min(ys) - 0.1
    ymax = max(ys) + 0.1

    images = []
    os.makedirs("tmp_frames", exist_ok=True)

    correct = (true_label.lower() == pred_label.lower())
    label_color = "green" if correct else "red"

    for i in range(len(xs)):
        plt.figure(figsize=(5, 5))
        plt.plot(xs[:i+1], ys[:i+1], "-o", color="blue")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("Horizontal Motion (x)")
        plt.ylabel("Synthetic Height")
        plt.title(f"Bar Path (frame {i})")

        # ---- Overlay TRUE vs PREDICTED ----
        text = f"True: {true_label.upper()}   Pred: {pred_label.upper()}"
        plt.text(
            xmin + 0.02, ymax - 0.05,
            text,
            fontsize=13,
            color=label_color,
            fontweight="bold"
        )

        fname = f"tmp_frames/frame_{i}.png"
        plt.savefig(fname)
        plt.close()

        images.append(imageio.imread(fname))

    imageio.mimsave(outpath, images, fps=15)
    print(f"✔ Saved GIF: {outpath} (True={true_label}, Pred={pred_label})")


# ============================================================
#   MAIN FUNCTION: USE test_frames.csv + test_rep_predictions.csv
# ============================================================

def generate_all_test_gifs(test_frames_csv="test_frames.csv",
                           predictions_csv="test_rep_predictions.csv"):

    df = pd.read_csv(test_frames_csv)
    preds = pd.read_csv(predictions_csv)

    os.makedirs("gif_outputs", exist_ok=True)

    # Merge prediction labels into frame-level data
    df = df.merge(
        preds,
        on=["subject_id", "set_id", "rep_id"],
        how="left",
        validate="many_to_one"
    )

    # Must contain: subject_id, set_id, rep_id, label, pred_label
    unique_reps = df[["subject_id", "set_id", "rep_id", "label", "pred_label_str"]].drop_duplicates()

    print(f"Found {len(unique_reps)} test reps to visualize.\n")

    for _, row in unique_reps.iterrows():
        subject = row.subject_id
        set_id = row.set_id
        rep_id = row.rep_id
        true_label = row.label
        pred_label = row.pred_label_str

        rep_df = df[
            (df.subject_id == subject) &
            (df.set_id == set_id) &
            (df.rep_id == rep_id)
        ]

        if len(rep_df) == 0:
            print(f"⚠ Skipping empty rep: {subject} set {set_id} rep {rep_id}")
            continue

        outname = f"gif_outputs/{subject}_set{set_id}_rep{rep_id}.gif"
        create_barpath_gif(rep_df, outname, true_label, pred_label)

    print("\n✨ DONE — GIFs saved in ./gif_outputs/ ✨")


# ============================================================
#   RUN IT
# ============================================================

if __name__ == "__main__":
    generate_all_test_gifs()
