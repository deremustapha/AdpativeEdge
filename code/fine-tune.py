from utils.data_preparation import *
from utils.preprocessing import *
from utils.features import *
from utils.tools import *
from utils.optimizers import *
# NOTE: we only reuse your gradient variance util; masking/hooks are implemented here structurally
from utils.adaptive import compute_gradient_variance

from models.model import *
from models.mcunet.mcunet.model_zoo import build_model

import argparse, os, time, random, tracemalloc
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# --------------------------
# Reproducibility
# --------------------------
def set_random_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------
# Metrics helpers
# --------------------------
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_effectively_updatable_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if not p.requires_grad: 
            continue
        mask = getattr(p, "ae_mask", None)
        if mask is None:
            total += p.numel()
        else:
            total += int(mask.sum().item()) if mask.dtype == torch.bool else int((mask != 0).sum().item())
    return total

def count_effective_grad_elems(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += int(torch.count_nonzero(p.grad).item())
    return total


class GPUEnergyMeter:
    def __init__(self, device_index: int = 0):
        self.enabled = False; self.total_kJ = 0.0
        if _NVML_OK:
            try:
                pynvml.nvmlInit(); self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.enabled = True
            except Exception:
                self.enabled = False
    def _power_W(self) -> Optional[float]:
        if not self.enabled: return None
        try: return pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000.0
        except Exception: return None
    def integrate(self, dt_s: float):
        if not self.enabled: return
        p1 = self._power_W(); time.sleep(0); p2 = self._power_W()
        if p1 is None and p2 is None: return
        if p1 is None: p1 = p2
        if p2 is None: p2 = p1
        self.total_kJ += 0.5*(p1+p2)*dt_s/1000.0
    def reset(self): self.total_kJ = 0.0
    def shutdown(self):
        if self.enabled:
            try: pynvml.nvmlShutdown()
            except Exception: pass


class BackwardMemoryTracker:
    def __init__(self, device: torch.device):
        self.device = device
    def start(self):
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        else: tracemalloc.start()
    def stop_and_bytes(self) -> int:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            return int(torch.cuda.max_memory_allocated(self.device))
        else:
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); return int(peak)


@dataclass
class BackwardCost:
    time_ms: float
    grad_elems: int

def measure_backward(loss: torch.Tensor, model: nn.Module, device: torch.device) -> BackwardCost:
    t0 = time.perf_counter(); loss.backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    dt_ms = (time.perf_counter()-t0)*1000.0
    elems = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
    return BackwardCost(time_ms=dt_ms, grad_elems=elems)


@dataclass
class EpochMetrics:
    regime: str; epoch: int; num_batches: int
    avg_backward_time_ms: float; peak_backward_mem_MB: float; avg_backward_grad_elems: float
    train_time_s: float; energy_kJ: float
    train_loss: float; train_acc: float; test_loss: float; test_acc: float
    trainable_params: int; effective_updatable_params: int


# --------------------------
# Data & model
# --------------------------
def prepare_data(path, session, subject, num_gesture, num_repetitions,
                 training_type, window_time, overlap):
    record_time=5; fs=250; no_channel=8
    low_cut=10.0; high_cut=120.0; notch_freq=60.0; order=5
    train_percent=80; activate_session=True
    selected_gesture=[1,2,3,4,5,6,7]
    emg_prep = EMGDataPreparation(base_path=path, fs=fs, rec_time=record_time)

    if training_type=="tsts":
        train_rep = np.arange(1, num_repetitions+1).tolist(); test_rep=[1,2]
        subject_path, train_g, test_g = emg_prep.get_per_subject_file(
            subject_number=subject, num_gesture=num_gesture, session=session, activate_session=activate_session,
            train_repetition=train_rep, test_repetition=test_rep
        )
        train_data, _ = emg_prep.load_data_per_subject(
            subject_path, selected_gesture=selected_gesture, train_gesture=train_g, test_gesture=test_g
        )
        train_data, train_labels = emg_prep.get_data_labels(train_data)
        preprocess = EMGPreprocessing(fs=fs, notch_freq=notch_freq, low_cut=low_cut, high_cut=high_cut, order=order)
        train_data = preprocess.highpass_filter(preprocess.remove_mains(train_data))
        w_tr, l_tr = emg_prep.window_with_overlap(train_data, train_labels, window_time=window_time, overlap=overlap, no_channel=no_channel)
        w_tr, l_tr = shuffle_data(w_tr, l_tr)
        w_tr, l_tr, w_te, l_te = data_split(w_tr, l_tr, train_percent=train_percent)
        X_train, y_train = shuffle_data(w_tr, l_tr); X_test, y_test = shuffle_data(w_te, l_te)
        # print(f"Train samples: {X_train.shape} | Test samples: {X_test.shape}")
        # print(f"Train labels: {y_train.shape} | Test labels: {y_test.shape}")
        X_train, y_train, X_test, y_test = X_train[:100,:], y_train[:100], X_test[:100, :], y_test[:100]
        return X_train, y_train, X_test, y_test

    elif training_type=="lro":
        num_rep = np.arange(1, num_repetitions+1).tolist()
        test_numbers = random.sample(num_rep, k=int(len(num_rep)*0.3))
        train_numbers = [n for n in num_rep if n not in test_numbers]
        subject_path, train_g, test_g = emg_prep.get_per_subject_file(
            subject_number=subject, num_gesture=num_gesture, session=session, activate_session=activate_session,
            train_repetition=train_numbers, test_repetition=test_numbers
        )
        tr_data, te_data = emg_prep.load_data_per_subject(
            subject_path, selected_gesture=selected_gesture, train_gesture=train_g, test_gesture=test_g
        )
        tr_data, tr_labels = emg_prep.get_data_labels(tr_data)
        te_data, te_labels = emg_prep.get_data_labels(te_data)
        preprocess = EMGPreprocessing(fs=fs, notch_freq=notch_freq, low_cut=low_cut, high_cut=high_cut, order=order)
        tr_data = preprocess.highpass_filter(preprocess.remove_mains(tr_data))
        te_data = preprocess.highpass_filter(preprocess.remove_mains(te_data))
        w_tr, l_tr = emg_prep.window_with_overlap(tr_data, tr_labels, window_time=window_time, overlap=overlap, no_channel=no_channel)
        w_te, l_te = emg_prep.window_with_overlap(te_data, te_labels, window_time=window_time, overlap=overlap, no_channel=no_channel)
        X_train, y_train = shuffle_data(w_tr, l_tr); X_test, y_test = shuffle_data(w_te, l_te)
        return X_train, y_train, X_test, y_test

    else:
        raise ValueError("Invalid training type. Choose 'tsts' or 'lro'.")


def process_input_data(input_type, X_train, X_test):
    if input_type=="raw":
        return get_raw_data(X_train), get_raw_data(X_test)
    if input_type=="stft":
        return get_stft_features(X_train), get_stft_features(X_test)
    if input_type=="cwt":
        return get_cwt_features(X_train), get_cwt_features(X_test)
    raise ValueError("Invalid input_type")


def initialize_model(model_type, input_type, training_type,
                     in_channel, num_gesture, device, load_weights, weights_root):
    load_file = f"KD_{model_type}_Input_{input_type}_Train_Type_{training_type}.pth"
    weights_path = os.path.join(weights_root, load_file)
    if model_type=="EMGNet": model = EMGNet(in_channel, num_gesture).to(device)
    elif model_type=="EMGNetFAN": model = EMGNetFAN(in_channel, num_gesture).to(device)
    elif model_type=="EMGNas": model = EMGNas(in_channel, num_gesture).to(device)
    elif model_type=="EMGNasFAN": model = EMGNasFAN(in_channel, num_gesture).to(device)
    elif model_type=="MCUNet":
        def ctor(ch_in, n_cls):
            mcunet, _, _ = build_model(net_id="mcunet-in3", pretrained=True)
            mcunet.first_conv.conv = nn.Conv2d(ch_in, 16, 3, 1, 1, bias=False)
            mcunet.classifier = nn.Linear(160, n_cls); return mcunet
        model = ctor(in_channel, num_gesture).to(device)
    elif model_type=="MobileNet": model = MobileNet(in_channel, num_gesture).to(device)
    elif model_type=="ProxyLessNas": model = ProxyLessNas(in_channel, num_gesture).to(device)
    else: raise ValueError("Invalid model_type")
    if load_weights and os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Loaded default/random weights.")
    return model


# --------------------------
# Eval helper
# --------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion)->Tuple[float,float]:
    model.eval(); total=0; loss_sum=0.0; correct=0
    for X,y in loader:
        X=X.float().to(device, non_blocking=True); y=y.long().to(device, non_blocking=True)
        out=model(X); loss=criterion(out,y)
        loss_sum += loss.item()*y.size(0); total += y.size(0)
        correct += (out.argmax(1)==y).sum().item()
    return loss_sum/max(1,total), correct/max(1,total)


# --------------------------
# Structured Adaptive-Edge
# --------------------------
def build_struct_masks(
    model: nn.Module,
    grad_var: Dict[str, torch.Tensor],
    tensor_keep_ratio: float = 0.6,
    intra_keep_ratio: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Create boolean masks per parameter:
      1) Score each param tensor by mean(|variance|), keep top tensor_keep_ratio fraction entirely.
      2) (Optional) For kept tensors, keep only top intra_keep_ratio elements by variance (unstructured within tensor).
    Returns dict(name -> bool mask with same shape as param).
    """
    # 1) gather scores
    scores = []
    name_to_param = {n: p for n, p in model.named_parameters()}
    for n, p in name_to_param.items():
        v = grad_var.get(n, None)
        if v is None:
            continue
        scores.append((n, float(v.abs().mean().item())))
    if not scores:
        # fallback: keep everything
        return {n: torch.ones_like(p, dtype=torch.bool, device=p.device) for n,p in name_to_param.items()}

    scores.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(round(len(scores)*tensor_keep_ratio)))
    keep_names = set(n for n,_ in scores[:k])

    masks = {}
    for n, p in name_to_param.items():
        device = p.device
        v = grad_var.get(n, None)
        if n not in keep_names or v is None:
            masks[n] = torch.zeros_like(p, dtype=torch.bool, device=device)  # fully frozen
            continue
        if intra_keep_ratio is None or intra_keep_ratio >= 1.0:
            masks[n] = torch.ones_like(p, dtype=torch.bool, device=device)   # keep whole tensor
        else:
            # elementwise top-k within this tensor
            flat_v = v.reshape(-1).abs()
            k_elem = max(1, int(round(flat_v.numel()*intra_keep_ratio)))
            thresh = torch.topk(flat_v, k_elem, largest=True).values.min()
            masks[n] = (v.abs() >= thresh).to(torch.bool).to(device)
            # ensure shape match
            if masks[n].shape != p.shape:
                masks[n] = masks[n].reshape(p.shape).to(torch.bool).to(device)
    return masks


def attach_masks_and_freeze(model: nn.Module, masks: Dict[str, torch.Tensor]):
    """
    Attach .ae_mask and set requires_grad=False for fully-masked tensors.
    """
    frozen_tensors = 0
    frozen_elems = 0
    for n, p in model.named_parameters():
        m = masks.get(n, None)
        if m is None:
            # No mask known -> keep all
            m = torch.ones_like(p, dtype=torch.bool, device=p.device)
        # persist on param for metrics
        p.ae_mask = m
        # freeze if fully zero
        if not bool(m.any().item()):
            if p.requires_grad:
                p.requires_grad = False
                p.grad = None
            frozen_tensors += 1
            frozen_elems += p.numel()
        else:
            p.requires_grad = True
    return {"frozen_tensors": frozen_tensors, "frozen_elems": frozen_elems}


def register_param_mask_hooks(model: nn.Module):
    """
    Register autograd hooks that zero out masked gradient elements for params that remain trainable.
    """
    for p in model.parameters():
        m = getattr(p, "ae_mask", None)
        if m is not None and p.requires_grad:
            def _hook(grad, mask=m):
                return grad * mask.to(grad.dtype)
            p.register_hook(_hook)


# --------------------------
# Train one regime with metrics
# --------------------------
def train_regime(model: nn.Module,
                 regime_name: str,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epochs: int,
                 out_dir: str,
                 device_index: int = 0,
                 report_effective_params: bool = False) -> List[EpochMetrics]:

    energy = GPUEnergyMeter(device_index=device_index)
    rows: List[EpochMetrics] = []

    logical_params = count_trainable_params(model)
    effective_params = count_effectively_updatable_params(model) if report_effective_params else logical_params
    print(f"[{regime_name}] Trainable parameters: {logical_params:,}"
          + (f" (effective updatable: {effective_params:,})" if report_effective_params else ""))

    for epoch in range(1, epochs+1):
        model.train(); epoch_start = time.perf_counter(); energy.reset()
        peak_bwd_bytes = 0; bwd_times=[]; bwd_grad_counts=[]
        run_loss=0.0; run_correct=0; run_total=0

        for X,y in train_loader:
            batch_t0 = time.perf_counter()
            X = X.float().to(device, non_blocking=True); y = y.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(X); loss = criterion(out, y)

            mem = BackwardMemoryTracker(device); mem.start()
            cost = measure_backward(loss, model, device)
            peak_bwd_bytes = max(peak_bwd_bytes, mem.stop_and_bytes())

            # effective non-zero gradients after masking
            bwd_grad_counts.append(count_effective_grad_elems(model))
            bwd_times.append(cost.time_ms)

            optimizer.step()

            run_loss += loss.item()*y.size(0); run_total += y.size(0)
            run_correct += (out.argmax(1)==y).sum().item()
            energy.integrate(dt_s=(time.perf_counter()-batch_t0))

        train_time_s = time.perf_counter()-epoch_start
        train_loss = run_loss/max(1,run_total); train_acc = run_correct/max(1,run_total)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        m = EpochMetrics(
            regime=regime_name, epoch=epoch, num_batches=len(train_loader),
            avg_backward_time_ms=float(sum(bwd_times)/max(1,len(bwd_times))),
            peak_backward_mem_MB=peak_bwd_bytes/(1024.0**2),
            avg_backward_grad_elems=float(sum(bwd_grad_counts)/max(1,len(bwd_grad_counts))),
            train_time_s=train_time_s, energy_kJ=energy.total_kJ if energy.enabled else float("nan"),
            train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc,
            trainable_params=logical_params, effective_updatable_params=effective_params
        )
        rows.append(m)
        print(f"[{regime_name} | Epoch {epoch}] time {m.train_time_s:.2f}s | bwd {m.avg_backward_time_ms:.2f}ms | "
              f"peak_bwd_mem {m.peak_backward_mem_MB:.2f}MB | grad_elems {m.avg_backward_grad_elems:.0f} | "
              f"energy {m.energy_kJ:.4f}kJ | train {m.train_acc*100:.2f}% | test {m.test_acc*100:.2f}%")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"metrics_{regime_name.replace(' ', '_').lower()}.csv")
    pd.DataFrame([asdict(r) for r in rows]).to_csv(csv_path, index=False)
    print(f"[{regime_name}] Saved metrics to {csv_path}")
    energy.shutdown()
    return rows


# --------------------------
# Main orchestration
# --------------------------
def run(path, session, subject, input_type, num_gesture,
        num_repetitions, window_time, overlap, training_type,
        model_type, epochs, save_path, load_path, seed,
        tensor_keep_ratio=0.6, intra_keep_ratio=0.6):

    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    batch_size = 4; lr = 1e-3

    print(f"Participant {subject} | Device: {device}")

    # Data
    X_train, y_train, X_test, y_test = prepare_data(
        path, session, subject, num_gesture, num_repetitions, training_type, window_time, overlap
    )
    X_train, X_test = process_input_data(input_type, X_train, X_test)
    print(f"Train: {X_train.shape}/{y_train.shape} | Test: {X_test.shape}/{y_test.shape}")

    train_ds = EMGDataset(X_train, y_train); test_ds = EMGDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=(device.type=="cuda"))

    in_channel = X_train.shape[1]
    os.makedirs(save_path, exist_ok=True)
    csv_dir = os.path.join(save_path, "metrics_csv"); os.makedirs(csv_dir, exist_ok=True)

    # Baseline
    base = initialize_model(model_type, input_type, training_type, in_channel,
                            num_gesture, device, load_weights=True, weights_root=load_path)
    base_loss, base_acc = evaluate(base, test_loader, device, criterion)
    print("##########################################################")
    print(f"Baseline (no FT) | test acc: {base_acc*100:.2f}% | loss: {base_loss:.4f}")
    print("##########################################################")

    # Regime 1: Full-Train
    model = initialize_model(model_type, input_type, training_type, in_channel,
                             num_gesture, device, load_weights=True, weights_root=load_path)
    opt = optim.Adam(model.parameters(), lr=lr)
    train_regime(model, "Full-Train", train_loader, test_loader, device, criterion, opt, epochs, csv_dir)

    # Regime 2: Last-Layer
    model = initialize_model(model_type, input_type, training_type, in_channel,
                             num_gesture, device, load_weights=True, weights_root=load_path)
    last_layer_name = list(model.named_children())[-1][0]
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(last_layer_name)
        if 'bias' in name: p.requires_grad = True
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_regime(model, "Last-Layer", train_loader, test_loader, device, criterion, opt, epochs, csv_dir)

    # Regime 3: TinyTL (bias-only)
    model = initialize_model(model_type, input_type, training_type, in_channel,
                             num_gesture, device, load_weights=True, weights_root=load_path)
    for name, p in model.named_parameters():
        p.requires_grad = ('bias' in name)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_regime(model, "TinyTL", train_loader, test_loader, device, criterion, opt, epochs, csv_dir)

    # Regime 4: Structured Adaptive-Edge
    model = initialize_model(model_type, input_type, training_type, in_channel,
                             num_gesture, device, load_weights=True, weights_root=load_path)

    # 1) Gradient variance over subset
    grad_var = compute_gradient_variance(model, train_loader, criterion,
                                         num_batches=batch_size, device=device)  # expects dict[name]->tensor

    # 2) Build STRUCTURED masks (freeze whole tensors by score; optional intra-tensor masking)
    masks = build_struct_masks(
        model, grad_var,
        tensor_keep_ratio=tensor_keep_ratio,   # e.g., keep top 60% tensors
        intra_keep_ratio=intra_keep_ratio      # e.g., within-kept tensors, keep top 60% elements; or None to keep all
    )

    # 3) Attach masks & freeze fully masked tensors (real optimizer/memory reduction)
    freeze_summary = attach_masks_and_freeze(model, masks)
    print(f"[Adaptive-Edge] Fully frozen tensors: {freeze_summary['frozen_tensors']} "
          f"({freeze_summary['frozen_elems']:,} elems)")

    # 4) Register hooks so masked elements inside kept tensors get zeroed during backward
    register_param_mask_hooks(model)

    # 5) Free temp + empty cache
    del grad_var
    if device.type=="cuda": torch.cuda.empty_cache()

    # 6) Optimizer excludes frozen tensors
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 7) Train with accurate effective counts
    train_regime(model, "Adaptive-Edge", train_loader, test_loader, device, criterion, opt,
                 epochs, csv_dir, report_effective_params=True)

    # Save final model
    final_path = os.path.join(save_path, f"FineTuned_{model_type}_Input_{input_type}_Train_Type_{training_type}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final weights to: {final_path}")

# ProxyLessNas
# MobileNet
def main():
    ap = argparse.ArgumentParser(description="EMG fine-tuning with STRUCTURED Adaptive-Edge")
    ap.add_argument("--path", type=str, default="")
    ap.add_argument("--session", type=int, default=4)
    ap.add_argument("--subject", type=int, default=2)
    ap.add_argument("--input_type", type=str, default="raw", choices=["raw","stft","cwt"])
    ap.add_argument("--num_gesture", type=int, default=7)
    ap.add_argument("--num_repetitions", type=int, default=9)
    ap.add_argument("--window_time", type=int, default=160)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--training_type", type=str, default="tsts", choices=["tsts","lro"])
    ap.add_argument("--model_type", type=str, default="ProxyLessNas",
                    choices=["EMGNet","EMGNetFAN","EMGNas","EMGNasFAN","MCUNet","MobileNet","ProxyLessNas"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--save_path", type=str, default="")
    ap.add_argument("--load_path", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tensor_keep_ratio", type=float, default=0.6, help="Fraction of parameter tensors to KEEP entirely.")
    ap.add_argument("--intra_keep_ratio", type=float, default=0.6, help="Within kept tensors, fraction of elements to keep; set to 1.0 or None to keep all.")
    args = ap.parse_args()

    run(args.path, args.session, args.subject, args.input_type, args.num_gesture,
        args.num_repetitions, args.window_time, args.overlap, args.training_type,
        args.model_type, args.epochs, args.save_path, args.load_path, args.seed,
        tensor_keep_ratio=args.tensor_keep_ratio, intra_keep_ratio=args.intra_keep_ratio)


if __name__ == "__main__":
    main()


