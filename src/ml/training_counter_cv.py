import sys
import os
import time
import random
import platform

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
import pandas as pd
import numpy as np
import argparse

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import (
    get_counter_train_transforms,
    get_counter_test_transforms,
)
from src.ml.models.model_dictionary import MODEL_DICTIONARY

# Device & dataloader settings
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"[INFO] Using device: {DEVICE}")

# Global defaults & training constants
# "Best" hyperparameters from previous cv
BEST_LR = 1e-4
BEST_WEIGHT_DECAY = 0.0
BEST_BETA = 1.0

CRITERION = nn.SmoothL1Loss(beta=BEST_BETA)
IMG_SIZE = 224
BATCH_SIZE = 16

# safer default (no multiprocessing issues with lambdas on macOS)
IS_MACOS = (platform.system() == "Darwin")
NUM_WORKERS = 0 if IS_MACOS else 2

# pin_memory only meaningful/supported on CUDA
PIN_MEMORY = (DEVICE == "cuda")

# Hyperparameter search configuration

# We only grid over lr and weight_decay; other choices are fixed in CV.
# The lists are centred around the "best" defaults above.
HYPERPARAM_COMMON = {
    "lr": [BEST_LR, 5e-4],
    "weight_decay": [BEST_WEIGHT_DECAY, 1e-4],
}

# SmoothL1 betas to test in CV (includes the best beta)
SMOOTHL1_BETAS = [BEST_BETA, 0.5]


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Random seed set to {seed}")


def train_model(
    csv_path: str,
    model_class,             # Example: EfficientNetB0Regressor
    model_kwargs=None,       # Extra args for model
    optimizer_class=optim.AdamW,
    optimizer_kwargs=None,
    criterion=CRITERION,
    batch_size: int = BATCH_SIZE,
    epochs: int = 30,
    lr: float = BEST_LR,
    img_size: int = IMG_SIZE,  # currently unused but kept for API symmetry
    save_path: str = "model.pth",
    scheduler_class=None,
    scheduler_kwargs=None,
):
    """
    General training loop for ANY PyTorch regression model.

    Args that control optimization are:
      - optimizer_class
      - optimizer_kwargs (if None → {'lr': lr, 'weight_decay': 0.0})
      - criterion
      - batch_size
      - epochs
    """

    model_kwargs = model_kwargs or {}
    if optimizer_kwargs is None:
        # Use the CLI / default lr plus no weight decay by default
        optimizer_kwargs = {"lr": lr, "weight_decay": BEST_WEIGHT_DECAY}

    #Load and clean data 
    df = pd.read_csv(csv_path).dropna(subset=["value"])
    df = df[df["value"] >= 0]  # Keep only valid labels
    print(f"[TRAIN] Loaded '{csv_path}' with {len(df)} valid samples.")

    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    print(f"[TRAIN] Split into train={len(train_df)} and val={len(val_df)}")

    #Dataset & loaders
    train_ds = ColonyDataset(df=train_df, transform=get_counter_train_transforms())
    val_ds = ColonyDataset(df=val_df, transform=get_counter_test_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Model & optimizer
    model = model_class(**model_kwargs).to(DEVICE)
    print(f"[TRAIN] Model: {model_class.__name__}, kwargs={model_kwargs}")
    print(f"[TRAIN] Optimizer: {optimizer_class.__name__}, kwargs={optimizer_kwargs}")
    print(f"[TRAIN] Criterion: {criterion.__class__.__name__}")
    print(f"[TRAIN] Batch size: {batch_size}, Epochs: {epochs}")

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    #Scheduler (must be created AFTER optimizer)
    scheduler = None
    if scheduler_class is not None:
        scheduler = scheduler_class(optimizer, **(scheduler_kwargs or {}))
        print(f"[TRAIN] Scheduler: {scheduler_class.__name__}, kwargs={scheduler_kwargs}")

    train_losses = []
    val_losses = []

    #Best model tracking 
    best_val_loss = float("inf")
    best_model_state = None

    start_time = time.perf_counter()
    print(f"{'Epoch':^12} | {'Train Loss':^12} | {'Val Loss':^12} | {'Epoch time':^12}")
    print("-" * 60)

    # Training Loop 
    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        # Train
        model.train()
        total_train_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            # Last batch can have different size
            total_train_loss += loss.item() * imgs.size(0)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                preds = model(imgs)
                total_val_loss += criterion(preds, labels).item() * imgs.size(0)

        avg_train_loss = total_train_loss / len(train_ds)
        avg_val_loss = total_val_loss / len(val_ds)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Checkpoint: keep best model by val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_time = time.perf_counter() - epoch_start

        epoch_str = f"{epoch + 1}/{epochs}"
        time_str = f"{epoch_time:.2f}s"
        print(
            f"{epoch_str:^12} | "
            f"{avg_train_loss:^12.4f} | "
            f"{avg_val_loss:^12.4f} | "
            f"{time_str:^12}  "
        )
        print("-" * 60)

        #Scheduler step (if used)
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != current_lr:
                print(f"[INFO] Epoch {epoch + 1}: LR adjusted from {current_lr:.2e} to {new_lr:.2e}")

    total_time = time.perf_counter() - start_time
    print("-" * 60)
    print(f"[TRAIN] Training finished in {total_time:0.1f} seconds.")
    print(f"[TRAIN] Best val loss = {best_val_loss:.4f}")

    #Restore best model before final save/return
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), save_path)
    print(f"[TRAIN] Best model (val loss={best_val_loss:.4f}) saved to {save_path}")

    return model, (train_losses, val_losses)


# -------------------------------------------------------------------------
# K-fold training for a single hyperparameter configuration
# -------------------------------------------------------------------------


def train_single_fold(
    df: pd.DataFrame,
    train_idx,
    val_idx,
    model_class,
    model_kwargs,
    optimizer_class,
    optimizer_kwargs,
    criterion,
    batch_size: int,
    epochs: int,
    img_size: int = IMG_SIZE,  # currently unused but kept for API symmetry
    scheduler_class=None,
    scheduler_kwargs=None,
    save_path: str = None,
):
    """
    Train the model on a single train/val split specified by indices.
    Returns the best validation loss achieved during this fold.
    """

    # Create per-fold train/val dataframes
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(
        f"[FOLD SETUP] train={len(train_df)}, val={len(val_df)}, "
        f"batch_size={batch_size}, epochs={epochs}"
    )

    # Datasets & loaders
    train_ds = ColonyDataset(df=train_df, transform=get_counter_train_transforms())
    val_ds = ColonyDataset(df=val_df, transform=get_counter_test_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Model & optimizer
    model = model_class(**(model_kwargs or {})).to(DEVICE)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # Optional scheduler
    scheduler = None
    if scheduler_class is not None:
        scheduler = scheduler_class(optimizer, **(scheduler_kwargs or {}))

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        total_train_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * imgs.size(0)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                preds = model(imgs)
                total_val_loss += criterion(preds, labels).item() * imgs.size(0)

        avg_val_loss = total_val_loss / len(val_ds)

        # Track best model for this fold
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Step scheduler at end of epoch (if any)
        if scheduler is not None:
            scheduler.step()

    # Optionally save the best model for this fold
    if save_path is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), save_path)

    # Free some memory on GPU between folds
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[FOLD RESULT] Best val loss (this fold) = {best_val_loss:.4f}")
    return best_val_loss


# -------------------------------------------------------------------------
# K-fold cross-validation over a hyperparameter grid
# -------------------------------------------------------------------------


def run_kfold_cv(
    csv_path: str,
    model_class,
    model_kwargs=None,
    k_folds: int = 5,
    hyperparam_grid: dict = None,
    epochs: int = 30,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    results_csv: str = "cv_results.csv",
):
    """
    Run K-fold cross-validation for all combinations in hyperparam_grid.
    Saves a CSV with one row per hyperparameter configuration, containing:
    - hyperparameter values
    - mean and std of the best validation loss across folds

    Returns:
        results_df (pd.DataFrame): full CV results.
        best_config (dict): hyperparameters of the best configuration.
    """

    model_kwargs = model_kwargs or {}
    if hyperparam_grid is None:
        hyperparam_grid = HYPERPARAM_COMMON  # default

    # Load and clean data ONCE, then split via KFold
    df = pd.read_csv(csv_path).dropna(subset=["value"])
    df = df[df["value"] >= 0].reset_index(drop=True)
    print(f"[CV] Loaded '{csv_path}' with {len(df)} valid samples.")
    print(f"[CV] Using {k_folds} folds.")
    print(f"[CV] Batch size for all folds: {batch_size}")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Base grid over lr and weight_decay
    base_grid = list(ParameterGrid(hyperparam_grid))
    print(f"[CV] Base hyperparameter combinations: {len(base_grid)}")

    # Expand grid over SmoothL1 betas
    grid = []
    for base in base_grid:
        for beta in SMOOTHL1_BETAS:
            cfg = dict(base)
            cfg["smoothl1_beta"] = beta
            grid.append(cfg)

    print(f"[CV] Total expanded configs (lr, weight_decay, beta): {len(grid)}")
    results = []

    print(f"[CV] Starting K-fold CV with {k_folds} folds and {len(grid)} configs.")

    for config_id, params in enumerate(grid):
        print(f"\n=== [CONFIG {config_id + 1}/{len(grid)}] ===")
        print(f"[CONFIG PARAMS] {params}")

        lr = params["lr"]
        weight_decay = params["weight_decay"]
        beta = params["smoothl1_beta"]

        # Fixed choices for CV
        optimizer_name = "adamw"
        optimizer_class = optim.AdamW
        scheduler_name = "step"
        scheduler_class = optim.lr_scheduler.StepLR
        scheduler_kwargs = {"step_size": 10, "gamma": 0.1}

        criterion = nn.SmoothL1Loss(beta=beta)

        optimizer_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
        }

        print(
            f"[CONFIG SUMMARY] opt={optimizer_name}, lr={lr}, batch={batch_size}, "
            f"wd={weight_decay}, sched={scheduler_name}, "
            f"criterion=SmoothL1, beta={beta}"
        )

        fold_val_losses = []

        # Loop over folds
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
            print(f"[CONFIG {config_id + 1}] Fold {fold_idx}/{k_folds} ...")
            best_val_loss = train_single_fold(
                df=df,
                train_idx=train_idx,
                val_idx=val_idx,
                model_class=model_class,
                model_kwargs=model_kwargs,
                optimizer_class=optimizer_class,
                optimizer_kwargs=optimizer_kwargs,
                criterion=criterion,
                batch_size=batch_size,
                epochs=epochs,
                img_size=img_size,
                scheduler_class=scheduler_class,
                scheduler_kwargs=scheduler_kwargs,
                save_path=None,  # or provide a format string if you want per-fold weights
            )
            print(
                f"[CONFIG {config_id + 1}] Fold {fold_idx} best val loss: {best_val_loss:.4f}"
            )
            fold_val_losses.append(best_val_loss)

        mean_loss = float(np.mean(fold_val_losses))
        std_loss = float(np.std(fold_val_losses))

        print(
            f"[CONFIG RESULT] {config_id + 1}/{len(grid)}: "
            f"mean val loss = {mean_loss:.4f} ± {std_loss:.4f}"
        )

        # Store one row per config
        results.append(
            {
                "config_id": config_id,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "criterion": "smoothl1",
                "smoothl1_beta": beta,
                "k_folds": k_folds,
                "mean_best_val_loss": mean_loss,
                "std_best_val_loss": std_loss,
                "fold_best_val_losses": fold_val_losses,
            }
        )
        # Save immediately after every config finishes
        pd.DataFrame(results).to_csv(results_csv, index=False)
        print(f"[CV] Updated results in '{results_csv}'")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)
    print(f"\n[CV] Saved cross-validation results to '{results_csv}'")

    #pick best hyperparameters
    best_idx = results_df["mean_best_val_loss"].idxmin()
    best_row = results_df.loc[best_idx]

    best_config = {
        "lr": float(best_row["lr"]),
        "weight_decay": float(best_row["weight_decay"]),
        "smoothl1_beta": float(best_row["smoothl1_beta"]),
        "batch_size": int(best_row["batch_size"]),
        "optimizer": best_row["optimizer"],
        "scheduler": best_row["scheduler"],
    }

    print("\n[CV] Best configuration by mean val loss:")
    for k, v in best_config.items():
        print(f"    {k}: {v}")
    print(f"[CV] Best mean val loss = {best_row['mean_best_val_loss']:.4f} "
          f"± {best_row['std_best_val_loss']:.4f}")

    return results_df, best_config


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name from MODEL_DICTIONARY",
    )
    parser.add_argument("--csv", type=str, default="data/training.csv")
    parser.add_argument("--epochs", type=int, default=10)

    # Hyperparameters (used directly in standard train mode, and for final
    # retraining after CV). Defaults are the "best" ones.
    parser.add_argument(
        "--lr",
        type=float,
        default=BEST_LR,
        help=f"Learning rate (default: {BEST_LR})",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=BEST_WEIGHT_DECAY,
        help=f"Weight decay (default: {BEST_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=BEST_BETA,
        help=f"SmoothL1Loss beta (default: {BEST_BETA})",
    )

    # --- CLI options for cross-validation ---
    parser.add_argument(
        "--cv",
        action="store_true",
        help="If set, run K-fold cross-validation over a hyperparameter grid.",
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="Number of folds for K-fold cross-validation.",
    )
    parser.add_argument(
        "--cv_csv",
        type=str,
        default="cv_results.csv",
        help="Path to CSV file where CV results are saved.",
    )

    args = parser.parse_args()

    # Lookup model
    if args.model not in MODEL_DICTIONARY:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {list(MODEL_DICTIONARY.keys())}"
        )

    entry = MODEL_DICTIONARY[args.model]

    model_class = entry["class"]
    model_kwargs = entry.get("kwargs", {})
    save_path = entry["weights"]  # automatic save file name

    print(f"\n[MAIN] Using model key: {args.model}")
    print(f"[MAIN] Model class: {model_class.__name__}")
    print(f"[MAIN] Weights path: {save_path}")
    print(f"[MAIN] CSV: {args.csv}, epochs: {args.epochs}, cv={args.cv}")
    print(
        f"[MAIN] Hyperparams → lr={args.lr}, weight_decay={args.weight_decay}, "
        f"batch_size={args.batch_size}, beta={args.beta}"
    )
    print(f"[MAIN] Number of workers: {NUM_WORKERS}, Pin memory: {PIN_MEMORY}")

    if args.cv:
        print("[MAIN] Running K-fold cross-validation mode.")
        results_df, best_config = run_kfold_cv(
            csv_path=args.csv,
            model_class=model_class,
            model_kwargs=model_kwargs,
            k_folds=args.kfolds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            results_csv=args.cv_csv,
        )

        # final training with best hyperparameters
        print("\n[MAIN] Retraining final model with best CV hyperparameters.")

        best_lr = best_config["lr"]
        best_weight_decay = best_config["weight_decay"]
        best_beta = best_config["smoothl1_beta"]
        final_batch_size = best_config["batch_size"]

        final_criterion = nn.SmoothL1Loss(beta=best_beta)
        final_optimizer_kwargs = {
            "lr": best_lr,
            "weight_decay": best_weight_decay,
        }

        scheduler_cls = optim.lr_scheduler.StepLR
        scheduler_kw = {"step_size": 10, "gamma": 0.1}

        train_model(
            csv_path=args.csv,
            model_class=model_class,
            model_kwargs=model_kwargs,
            optimizer_class=optim.AdamW,
            optimizer_kwargs=final_optimizer_kwargs,
            criterion=final_criterion,
            batch_size=final_batch_size,
            epochs=args.epochs,
            save_path=save_path,
            scheduler_class=scheduler_cls,
            scheduler_kwargs=scheduler_kw,
        )

    else:
        print("[MAIN] Running standard train/val mode.")
        # Best default hyperparameters:
        #   lr=0.0001, weight_decay=0.0, batch_size=16, beta=1.0
        scheduler_cls = optim.lr_scheduler.StepLR
        scheduler_kw = {"step_size": 10, "gamma": 0.1}

        standard_criterion = nn.SmoothL1Loss(beta=args.beta)
        standard_optimizer_kwargs = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

        train_model(
            csv_path=args.csv,
            model_class=model_class,
            model_kwargs=model_kwargs,
            optimizer_class=optim.AdamW,
            optimizer_kwargs=standard_optimizer_kwargs,
            criterion=standard_criterion,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_path=save_path,
            scheduler_class=scheduler_cls,
            scheduler_kwargs=scheduler_kw,
        )
