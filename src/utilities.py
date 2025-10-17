from monai.utils import first
import monai
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss, HausdorffDTLoss
from monai.networks.utils import one_hot
import torch.nn.functional as F


from monai.transforms import AsDiscrete, Compose, KeepLargestConnectedComponent
from monai.metrics import SurfaceDistanceMetric, DiceMetric

from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from torch.amp import GradScaler, autocast

import time
import torch
import wandb

def accumulate_confusion(pred_logits, label_bin):
    """
    pred_logits: [B, 2, ...] logits
    label_bin:   [B, 1, ...] binary (0/1) long/bool
    Returns tuple: (tp, fp, fn) as torch tensors on the same device.
    """
    pred_bin = pred_logits.argmax(dim=1, keepdim=True) == 1      # [B,1,...] bool
    label_bin = label_bin.bool()                                  # [B,1,...] bool

    tp = (pred_bin & label_bin).sum()
    fp = (pred_bin & ~label_bin).sum()
    fn = (~pred_bin & label_bin).sum()
    return tp, fp, fn

def dice_from_confusion(tp, fp, fn, eps=1e-8):
    return (2.0 * tp.float()) / (2.0 * tp.float() + fp.float() + fn.float() + eps)


def compute_recall(pred, label):
    pred_bin = pred.argmax(dim=1).bool()
    label_bin = label.bool()
    tp = (pred_bin & label_bin).sum().float()
    fn = (~pred_bin & label_bin).sum().float()
    return tp / (tp + fn + 1e-8)

def compute_precision(pred, label):
    pred_bin = pred.argmax(dim=1).bool()
    label_bin = label.bool()
    tp = (pred_bin & label_bin).sum().float()
    fp = (pred_bin & ~label_bin).sum().float()
    return tp / (tp + fp + 1e-8)


def train(model, train_loader, val_loader, test_loader, loss, optim, max_epochs, model_dir,
          patience=10, spatial_size = (128, 128, 48), overlap=0.5, test_interval=1, device=torch.device("cuda:0"), scheduler=None, scaler=None):
    best_metric = -1.0
    best_metric_epoch = -1
    epochs_no_improve = 0

    save_loss_train, save_loss_val = [], []
    save_metric_train, save_metric_val = [], []

    train_dice = DiceMetric(include_background=False, reduction="mean")
    val_dice   = DiceMetric(include_background=False, reduction="mean")
    surface_metric = SurfaceDistanceMetric(include_background=False, symmetric=True)
    keep_largest_cc = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=False)

    # Post transforms for DiceMetric (multi-class with background excluded)

    post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        start_time = time.time()

        # ------------------ TRAIN ------------------
        model.train()
        running_train_loss = 0.0
        train_steps = 0
        train_dice.reset()

        for step, batch_data in enumerate(train_loader, 1):
            train_steps += 1

            volume = batch_data["conc_image"].to(device)      # [B, C, H, W, D]
            label = (batch_data["seg_tumor"] != 0).long().to(device)

            optim.zero_grad()


            if scaler:
                with autocast("cuda"):
                    outputs = model(volume)
                    train_loss = loss(outputs, label)
                scaler.scale(train_loss).backward()
                scaler.step(optim)
                scaler.update()
            else:        
                outputs = model(volume)
                train_loss = loss(outputs, label)
                train_loss.backward()
                optim.step()
            


            running_train_loss += train_loss.item()

            # --- DiceMetric on training batch ---
            y_pred = [post_pred(o) for o in decollate_batch(outputs)]
            y_true = [post_label(l.squeeze(1)) for l in decollate_batch(label)]
            train_dice(y_pred=y_pred, y=y_true)

        train_epoch_loss = running_train_loss / max(train_steps, 1)
        train_mean_dice = train_dice.aggregate().item()
        train_dice.reset()

        print(f"Train Loss: {train_epoch_loss:.4f} | Train Dice: {train_mean_dice:.4f}")
        wandb.log({"train_loss": train_epoch_loss, "train_dice": train_mean_dice, "epoch": epoch + 1})

        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(train_mean_dice)
        np.save(os.path.join(model_dir, "loss_train.npy"), save_loss_train)
        np.save(os.path.join(model_dir, "metric_train.npy"), save_metric_train)

        # ---------------- VALIDATION ----------------
        if (epoch + 1) % test_interval == 0:
            model.eval()
            val_dice.reset()
            surface_metric.reset()

            running_val_loss = 0.0

            running_val_loss = 0.0
            val_steps = 0
            sum_recall = 0.0
            sum_precision = 0.0

            # NEW: global confusion accumulators for epoch-level Dice
            glob_tp = torch.tensor(0, device=device)
            glob_fp = torch.tensor(0, device=device)
            glob_fn = torch.tensor(0, device=device)

            val_steps = 0
            sum_recall = 0.0
            sum_precision = 0.0

            with torch.no_grad():
                for step, val_data in enumerate(val_loader, 1):
                    val_steps += 1
                    val_volume = val_data["conc_image"].to(device)
                    val_label = (val_data["seg_tumor"] != 0).long().to(device)


                    with autocast("cuda"):
                        val_outputs = sliding_window_inference(
                            inputs=val_volume,
                            roi_size= spatial_size,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=overlap,
                        )
                        if isinstance(val_outputs, list):
                            val_outputs = val_outputs[-1]

                        val_loss = loss(val_outputs, val_label)
                    running_val_loss += val_loss.item()

                    # --- DiceMetric  ---
                    y_pred = [post_pred(o) for o in decollate_batch(val_outputs)]
                    y_true = [post_label(l.squeeze(1)) for l in decollate_batch(val_label)]
                    val_dice(y_pred=y_pred, y=y_true)

                    tp, fp, fn = accumulate_confusion(val_outputs, val_label)
                    glob_tp += tp
                    glob_fp += fp
                    glob_fn += fn

                    # --- ASSD ---
                    val_pred_label = torch.argmax(val_outputs, dim=1)              # [B, H, W, D]
                    val_pred_tumor = (val_pred_label == 1).float().unsqueeze(1)   # [B,1,H,W,D]
                    val_label_tumor = (val_label == 1).float()                     # [B,1,H,W,D]

                    for i in range(val_pred_tumor.shape[0]):
                        val_pred_tumor[i] = keep_largest_cc(val_pred_tumor[i])

                    valid_preds, valid_labels = [], []
                    for p, l in zip(val_pred_tumor, val_label_tumor):
                        if p.sum() > 0 and l.sum() > 0:
                            valid_preds.append(p)
                            valid_labels.append(l)
                    if len(valid_preds) > 0:
                        batch_pred = torch.stack(valid_preds)
                        batch_label = torch.stack(valid_labels)
                        surface_metric(y_pred=batch_pred, y=batch_label)

                    sum_recall   += compute_recall(val_outputs, val_label).item()
                    sum_precision+= compute_precision(val_outputs, val_label).item()

            val_epoch_loss = running_val_loss / max(val_steps, 1)
            val_mean_dice  = val_dice.aggregate().item()
            val_dice.reset()

            global_dice_val = dice_from_confusion(glob_tp, glob_fp, glob_fn).item()


            try:
                epoch_surface_val = surface_metric.aggregate().item()
            except (ValueError, AttributeError):
                epoch_surface_val = 0.0
            surface_metric.reset()

            epoch_recall_val    = sum_recall / max(val_steps, 1)
            epoch_precision_val = sum_precision / max(val_steps, 1)

            

            print(
                f"Validation Loss: {val_epoch_loss:.4f} | "
                f"Validation Dice: {val_mean_dice:.4f} | "
                f"Global Dice: {global_dice_val:.4f} | "
                f"Recall: {epoch_recall_val:.4f} | "
                f"Precision: {epoch_precision_val:.4f} | "
                f"ASSD: {epoch_surface_val:.4f}"
            )

            wandb.log({
                "val_loss": val_epoch_loss,
                "val_dice": val_mean_dice,
                "val_dice_global": global_dice_val,
                "val_recall": epoch_recall_val,
                "val_precision": epoch_precision_val,
                "val_surface": epoch_surface_val
            })

            save_loss_val.append(val_epoch_loss)
            save_metric_val.append(val_mean_dice)
            np.save(os.path.join(model_dir, "loss_val.npy"), save_loss_val)
            np.save(os.path.join(model_dir, "metric_val.npy"), save_metric_val)

            # ------- SCHEDULER -------------

            if scheduler is not None:
                scheduler.step(1.0 - val_mean_dice)


            # --- early stopping / checkpoint on val Dice ---
            if val_mean_dice > best_metric:
                best_metric = val_mean_dice
                best_metric_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")

            print(
                f"Current Epoch: {epoch + 1} | Validation Mean Dice: {val_mean_dice:.4f}"
                f"\nBest Validation Dice: {best_metric:.4f} at Epoch: {best_metric_epoch}"
            )

            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1} completed in {elapsed:.2f} seconds.")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                print(f"Training stopped. Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
                break

    print(f"Training completed. Best validation Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    wandb.log({"BEST_val_dice": best_metric})

    # ---------------- TEST (best checkpoint) ----------------
    print("\nFinal Test Evaluation (On Unseen Data)")
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
    wandb.save(os.path.join(model_dir, "best_metric_model.pth"))
    model.eval()

    test_dice = DiceMetric(include_background=False, reduction="mean")
    postprocessed_dice_scores = []
    test_recalls, test_precisions = [], []

    test_tp = torch.tensor(0, device=device)
    test_fp = torch.tensor(0, device=device)
    test_fn = torch.tensor(0, device=device)

    with torch.no_grad():
        test_dice.reset()
        for test_data in test_loader:
            test_volume = test_data["conc_image"].to(device)
            test_label  = (test_data["seg_tumor"] != 0).long().to(device)

            with autocast("cuda"):
                test_outputs = sliding_window_inference(
                    inputs=test_volume,
                    roi_size=spatial_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )
                if isinstance(test_outputs, list):
                    test_outputs = test_outputs[-1]
                       

            # --- DiceMetric on test logits ---
            y_pred = [post_pred(o) for o in decollate_batch(test_outputs)]
            y_true = [post_label(l.squeeze(1)) for l in decollate_batch(test_label)]
            test_dice(y_pred=y_pred, y=y_true)

            # update global confusion
            tp, fp, fn = accumulate_confusion(test_outputs, test_label)
            test_tp += tp
            test_fp += fp
            test_fn += fn


            # recall / precision
            test_recalls.append(compute_recall(test_outputs, test_label).item())
            test_precisions.append(compute_precision(test_outputs, test_label).item())


    final_dice = test_dice.aggregate().item()
    test_dice.reset()
    final_global_dice = dice_from_confusion(test_tp, test_fp, test_fn).item()
    final_recall = float(np.mean(test_recalls)) if len(test_recalls) else 0.0
    final_precision = float(np.mean(test_precisions)) if len(test_precisions) else 0.0

    print(f"Final Test Dice (case-mean): {final_dice:.4f}")
    print(f"Final Test Global Dice: {final_global_dice:.4f}")
    print(f"Final Test Recall: {final_recall:.4f}")
    print(f"Final Test Precision: {final_precision:.4f}")

    wandb.log({
        "test_dice": final_dice,
        "test_global_dice": final_global_dice,
        "test_recall": final_recall,
        "test_precision": final_precision
    })
    wandb.finish()