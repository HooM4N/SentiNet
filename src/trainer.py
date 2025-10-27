from time import time
from copy import deepcopy
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

def _batch_to_device(X, device):
    if isinstance(X, dict):
        return {k: v.to(device) for k, v in X.items()}
    return X.to(device)
    
def evaluate(model, metric, loss_fn, data_loader, device, enable_mixed_precision=False):
    """Evaluate model on a validation/test set."""
    metric.reset()
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for X_batch, y_batch in data_loader:
            X_batch = _batch_to_device(X_batch, device)
            y_batch = y_batch.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=enable_mixed_precision):
                yhat = model(X_batch)
                total_loss += loss_fn(yhat, y_batch).item()
            metric.update(yhat, y_batch)
        return (total_loss / len(data_loader) , metric.compute().item())
        
def trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    metric: torchmetrics.Metric,
    n_epochs: int,
    device:"str",
    train_loader: DataLoader,
    val_loader: DataLoader,
    enable_mixed_precision:bool = False,
    restore_best_model:bool = False,
    use_early_stopping:bool = False,
    early_stopping_patience:int = 5,
    scheduler: Optional[Any] = None,
    scheduler_monitor: str = "val_loss",
    epoch_callback: Optional[Callable] = None
) -> Dict[str, list]:
    """Train a model with logging, validation, and optional scheduler & early stopping.
    """    
    train_logs = {"train_loss":[] , "train_metric":[], "val_loss":[] , "val_metric":[], "lr":[]}

    # GRAD SCLAER FOR MIXED PRECISION
    scaler = torch.amp.GradScaler(enabled = enable_mixed_precision)
    
    best_val_loss_restore_model = float('inf')
    best_val_loss_early_stopping = float('inf')
    best_model_params = None
    early_stopping_counter = 0
    log_intervals = max(1, len(train_loader)//5)
            
    for epoch in range(n_epochs):
        
        # ON EPOCH START
        start_time = time()
        model.train()
        total_loss = 0.0
        
        if epoch_callback is not None:
            epoch_callback(model, epoch)

        # INNER LOOP
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = _batch_to_device(X_batch, device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=enable_mixed_precision):
                yhat = model(X_batch)
                loss = loss_fn(yhat, y_batch)
            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if idx % log_intervals == 0 or idx == 0:
                print(f"\r Epoch {epoch + 1}/{n_epochs}", end="")
                print(f", Step {idx+1}/{len(train_loader)}", end="")
                print(f", train_loss: {total_loss / (idx+1):.4f}", end="")

        # LOGGING
        train_logs["train_loss"].append(total_loss / len(train_loader))
        eval_loss, eval_metric = evaluate(model, metric, loss_fn, val_loader, device, enable_mixed_precision)
        train_logs["val_loss"].append(eval_loss)
        train_logs["val_metric"].append(eval_metric)
        train_logs["lr"].append(optimizer.param_groups[0]['lr'])
            
        # LR SCHEDULE
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_loss if scheduler_monitor=="val_loss" else eval_metric)
            else:
                scheduler.step()

        print(f"\r Epoch {epoch + 1}/{n_epochs}", end="")
        print(f", train_loss: {train_logs["train_loss"][-1]:.4f}", end="")
        print(f', val_loss: {train_logs["val_loss"][-1]:.4f}', end="")
        print(f', val_metric: {train_logs["val_metric"][-1]:.4f}', end="")
        print(f", lr: {train_logs["lr"][-1]}", end="")
        print(f', epoch_time: {time() - start_time:.2f}s')

        # SAVE BEST MODEL
        if restore_best_model:
            if train_logs["val_loss"][-1] < best_val_loss_restore_model:
                best_val_loss_restore_model = train_logs["val_loss"][-1]
                best_model_params = deepcopy(model.state_dict())
                best_model_epoch = epoch
            
        # EARLRY STOPPING
        if use_early_stopping:
            if train_logs["val_loss"][-1] < best_val_loss_early_stopping:
                best_val_loss_early_stopping = train_logs["val_loss"][-1]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early Stopping at Epoch: {epoch+1}")
                break
                
    if restore_best_model and best_model_params is not None:
        model.load_state_dict(best_model_params)
        print(f"Restoring best model from epoch {best_model_epoch+1}")
        
    return train_logs