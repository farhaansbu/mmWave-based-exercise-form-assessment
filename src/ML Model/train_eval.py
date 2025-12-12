# train_eval.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from radar_dataset import RadarRepsDataset
from model_pointnet_lstm import PointNetLSTMClassifier

# TODO: make sure these match your actual column names in *_frames.csv
FEATURES = ["x", "y", "z", "v", "range_m", "azimuth_deg", "snr", "noise"]

MAX_FRAMES = 30
MAX_OBJECTS = 5
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X).squeeze(1)          # (batch,)
        loss = criterion(logits, y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X).squeeze(1)
            loss = criterion(logits, y.float())

            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def collect_predictions(model, loader, device):
    """Return lists of y_true, y_pred, y_prob for a given loader."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(1)
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).long()

            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_prob.extend(probs.cpu().numpy().tolist())

    return all_true, all_pred, all_prob


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Datasets / loaders
    train_ds = RadarRepsDataset("train_frames.csv", FEATURES,
                                max_frames=MAX_FRAMES, max_objects=MAX_OBJECTS)
    val_ds   = RadarRepsDataset("val_frames.csv", FEATURES,
                                max_frames=MAX_FRAMES, max_objects=MAX_OBJECTS)
    test_ds  = RadarRepsDataset("test_frames.csv", FEATURES,
                                max_frames=MAX_FRAMES, max_objects=MAX_OBJECTS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model / loss / optimizer
    model = PointNetLSTMClassifier(in_features=len(FEATURES)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # For plotting
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss {train_loss:.4f} | Acc {train_acc:.4f}")
        print(f"  Val   Loss {val_loss:.4f} | Acc {val_acc:.4f}")

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("  Saved best model.")

        # Log history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    # Save training history to CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv("training_history.csv", index=False)
    print("\nSaved training history to training_history.csv")

    # Plot accuracy and loss curves
    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    print("Saved accuracy plot to accuracy_curve.png")

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("Saved loss plot to loss_curve.png")

    # -------------------------
    #  TEST EVALUATION
    # -------------------------
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print("\nTEST PERFORMANCE:")
    print(f"  Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")

    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    # Confusion matrix, precision, recall, F1
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1], zero_division=0
    )
    print("\nClass-wise precision/recall/F1 (0=bad, 1=good):")
    print("  Class 0 (bad):  P={:.3f} R={:.3f} F1={:.3f} (n={})".format(
        precision[0], recall[0], f1[0], support[0]
    ))
    print("  Class 1 (good): P={:.3f} R={:.3f} F1={:.3f} (n={})".format(
        precision[1], recall[1], f1[1], support[1]
    ))

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["bad", "good"], zero_division=0))


if __name__ == "__main__":
    main()
