import pandas as pd
from sklearn.model_selection import train_test_split


VAL_FORCED = ["Person_G"]
TEST_FORCED = ["Person_E", "Person_F"]
TRAINABLE = ["Person_A", "Person_B", "Person_C", "Person_D"]

VAL_FRAC = 0.10
TEST_FRAC = 0.10


def balance(df):
    good = df[df.label == "good"]
    bad = df[df.label == "bad"]
    if len(good) == 0 or len(bad) == 0:
        return df
    m = min(len(good), len(bad))
    return pd.concat([
        good.sample(m, random_state=42),
        bad.sample(m, random_state=42)
    ])


def split_dataset(master="master_dataset.csv"):
    df = pd.read_csv(master)

    # group reps
    reps = df.groupby(["subject_id", "set_id", "rep_id"]).first().reset_index()

    total = len(reps)
    target_val = int(total * VAL_FRAC)
    target_test = int(total * TEST_FRAC)

    # initialize split
    reps["split"] = "train"

    # assign forced subjects
    reps.loc[reps.subject_id.isin(VAL_FORCED), "split"] = "val"
    reps.loc[reps.subject_id.isin(TEST_FORCED), "split"] = "test"

    # remaining A–D data
    remaining = reps[(reps.subject_id.isin(TRAINABLE)) &
                     (reps.split == "train")]

    # ---------------------------------------
    # Fill VAL set up to target size
    # ---------------------------------------
    current_val = len(reps[reps.split == "val"])
    need_val = max(0, target_val - current_val)

    if need_val > 0:
        # ensure some BAD in validation
        bad_pool = remaining[remaining.label == "bad"]
        if len(bad_pool) > 0:
            take_bad = min(need_val // 3, len(bad_pool))  # ~33% bad
            extra_val = bad_pool.sample(take_bad, random_state=42)
            reps.loc[extra_val.index, "split"] = "val"
            need_val -= take_bad
            remaining = reps[(reps.subject_id.isin(TRAINABLE)) &
                             (reps.split == "train")]

        # fill remainder with mix of good/bad
        if need_val > 0:
            extra_val2 = remaining.sample(need_val, random_state=42)
            reps.loc[extra_val2.index, "split"] = "val"

    # ---------------------------------------
    # Fill TEST set up to target size
    # ---------------------------------------
    remaining = reps[(reps.subject_id.isin(TRAINABLE)) &
                     (reps.split == "train")]

    current_test = len(reps[reps.split == "test"])
    need_test = max(0, target_test - current_test)

    if need_test > 0:
        bad_pool = remaining[remaining.label == "bad"]
        if len(bad_pool) > 0:
            take_bad = min(need_test // 3, len(bad_pool))
            extra_test = bad_pool.sample(take_bad, random_state=42)
            reps.loc[extra_test.index, "split"] = "test"
            need_test -= take_bad
            remaining = reps[(reps.subject_id.isin(TRAINABLE)) &
                             (reps.split == "train")]

        if need_test > 0:
            extra_test2 = remaining.sample(need_test, random_state=42)
            reps.loc[extra_test2.index, "split"] = "test"


    # BALANCE TRAIN (initial)
    # ------------------------------
    train_balanced = balance(reps[reps.split == "train"])

    # ------------------------------
    # ADD LEFTOVER GOOD REPS TO TRAIN
    # ------------------------------
    # Good reps that were NOT used in train_balanced
    balanced_keys = set(zip(train_balanced.subject_id,
                            train_balanced.set_id,
                            train_balanced.rep_id))

    all_train_candidates = reps[reps.split == "train"]

    leftover_goods = all_train_candidates[
        (all_train_candidates.label == "good") &
        (~all_train_candidates.apply(
            lambda r: (r.subject_id, r.set_id, r.rep_id) in balanced_keys, axis=1))
    ]

    # Combine balanced + leftover goods
    train_final = pd.concat([train_balanced, leftover_goods], ignore_index=True)

    # ------------------------------
    # FINAL SPLITS
    # ------------------------------
    val = reps[reps.split == "val"]
    test = reps[reps.split == "test"]

    train_final.to_csv("train_reps.csv", index=False)
    val.to_csv("val_reps.csv", index=False)
    test.to_csv("test_reps.csv", index=False)

    print("\nFinal Split Results")
    print(" Train:", len(train_final))
    print(" Val:", len(val))
    print(" Test:", len(test))

    print("\nLabel counts:")
    print(" Train:\n", train_final.label.value_counts())
    print(" Val:\n", val.label.value_counts())
    print(" Test:\n", test.label.value_counts())


if __name__ == "__main__":
    split_dataset()
