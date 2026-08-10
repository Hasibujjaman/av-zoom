import os
import glob

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MVDRDataset
from crn_light import CRN_Light
from losses import log_mag_loss, si_sdr_loss, mrstft_loss
from stft_utils import stft_mag, istft

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- CONFIG (edit values here) --------------------
MVDR_DIR = "mvdr_outputs"
CLEAN_DIR = "true_labels"

SAMPLE_RATE = 16000

BATCH_SIZE = 8
EPOCHS = 50
LR = 2e-4

# Loss weights. Note: `si_sdr_loss()` returns NEGATIVE SI-SDR (a loss), so better SI-SDR => more negative.
SISDR_WARMUP_EPOCHS = 0  # set e.g. 2-5 if you want to warm up on spectral losses first
SI_SDR_WEIGHT_MAX = 1.0
MRSTFT_WEIGHT = 0.2
LOGMAG_WEIGHT = 0.1

# Penalize degrading SI-SDR vs MVDR (helps prevent the model from making things worse early on).
# This does NOT change the optimum vs pure SI-SDR, but helps stability.
IMPROVEMENT_HINGE_WEIGHT = 0.5

TRAIN_RATIO = 0.85
SEED = 1337

NUM_WORKERS = 4 if DEVICE == "cuda" else 0

# Train on fixed-size segments to avoid padding artifacts in waveform losses.
SEGMENT_SECONDS = 2.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
SEGMENT_CANDIDATES = 8  # higher -> less silence, more CPU

# Validation SI-SDRi should be measured on full utterances (recommended).
VAL_FULL_UTTERANCE = True

# Debug / smoke-test helpers (set to an int to stop early)
MAX_TRAIN_BATCHES_PER_EPOCH = None
MAX_VAL_BATCHES_PER_EPOCH = None

SAVE_EVERY_EPOCHS = 1
CHECKPOINT_PREFIX = "crn"

# -------------------------------------------------------------------

def _pad_collate(batch):
    # Default collate: center-crop/pad to a fixed segment.
    mvdr_list, clean_list = zip(*batch)
    mvdr_out = []
    clean_out = []

    for mvdr_wav, clean_wav in zip(mvdr_list, clean_list):
        if SEGMENT_SAMPLES is None:
            mvdr_out.append(mvdr_wav)
            clean_out.append(clean_wav)
            continue

        length = mvdr_wav.shape[-1]
        if length >= SEGMENT_SAMPLES:
            # Pick the most energetic segment among a few evenly-spaced candidates (more stable metrics).
            max_start = length - SEGMENT_SAMPLES
            if max_start == 0:
                start = 0
            else:
                candidates = torch.linspace(0, max_start, steps=min(SEGMENT_CANDIDATES, max_start + 1)).long()
                best_start = 0
                best_energy = None
                for s in candidates.tolist():
                    seg = clean_wav[s : s + SEGMENT_SAMPLES]
                    energy = torch.mean(seg * seg)
                    if best_energy is None or energy > best_energy:
                        best_energy = energy
                        best_start = s
                start = best_start

            mvdr_wav = mvdr_wav[start : start + SEGMENT_SAMPLES]
            clean_wav = clean_wav[start : start + SEGMENT_SAMPLES]
        else:
            pad = SEGMENT_SAMPLES - length
            mvdr_wav = torch.nn.functional.pad(mvdr_wav, (0, pad))
            clean_wav = torch.nn.functional.pad(clean_wav, (0, pad))

        mvdr_out.append(mvdr_wav)
        clean_out.append(clean_wav)

    return torch.stack(mvdr_out, dim=0), torch.stack(clean_out, dim=0)


def _train_collate(batch):
    # Random-crop/pad to fixed segment for training.
    mvdr_list, clean_list = zip(*batch)
    mvdr_out = []
    clean_out = []

    for mvdr_wav, clean_wav in zip(mvdr_list, clean_list):
        if SEGMENT_SAMPLES is None:
            mvdr_out.append(mvdr_wav)
            clean_out.append(clean_wav)
            continue

        length = mvdr_wav.shape[-1]
        if length >= SEGMENT_SAMPLES:
            max_start = length - SEGMENT_SAMPLES
            if max_start == 0:
                start = 0
            else:
                # Sample a few random crops and pick the one with highest clean energy (avoids silence).
                best_start = 0
                best_energy = None
                for _ in range(max(1, SEGMENT_CANDIDATES)):
                    s = int(torch.randint(low=0, high=max_start + 1, size=(1,)).item())
                    seg = clean_wav[s : s + SEGMENT_SAMPLES]
                    energy = torch.mean(seg * seg)
                    if best_energy is None or energy > best_energy:
                        best_energy = energy
                        best_start = s
                start = best_start

            mvdr_wav = mvdr_wav[start : start + SEGMENT_SAMPLES]
            clean_wav = clean_wav[start : start + SEGMENT_SAMPLES]
        else:
            pad = SEGMENT_SAMPLES - length
            mvdr_wav = torch.nn.functional.pad(mvdr_wav, (0, pad))
            clean_wav = torch.nn.functional.pad(clean_wav, (0, pad))

        mvdr_out.append(mvdr_wav)
        clean_out.append(clean_wav)

    return torch.stack(mvdr_out, dim=0), torch.stack(clean_out, dim=0)


def _validate_paths():
    if not os.path.isdir(MVDR_DIR):
        raise FileNotFoundError(f"MVDR_DIR not found: {MVDR_DIR}")
    if not os.path.isdir(CLEAN_DIR):
        raise FileNotFoundError(f"CLEAN_DIR not found: {CLEAN_DIR}")

    mvdr_wavs = glob.glob(os.path.join(MVDR_DIR, "*.wav"))
    clean_wavs = glob.glob(os.path.join(CLEAN_DIR, "*.wav")) + glob.glob(os.path.join(CLEAN_DIR, "*.flac"))
    if len(mvdr_wavs) == 0:
        raise RuntimeError(f"No .wav files found in {MVDR_DIR}")
    if len(clean_wavs) == 0:
        raise RuntimeError(f"No .wav/.flac files found in {CLEAN_DIR}")


def _make_loaders():
    dataset = MVDRDataset(MVDR_DIR, CLEAN_DIR, sample_rate=SAMPLE_RATE)
    if len(dataset) < 2:
        raise RuntimeError(f"Need at least 2 samples to split train/val, got {len(dataset)}")

    train_len = int(round(len(dataset) * TRAIN_RATIO))
    train_len = max(1, min(train_len, len(dataset) - 1))
    val_len = len(dataset) - train_len

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=_train_collate,
    )
    if VAL_FULL_UTTERANCE:
        # batch_size=1 avoids padding and keeps SI-SDRi honest.
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
        )
    else:
        val_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
            collate_fn=_pad_collate,
        )

    return train_loader, val_loader


def _sisdr_weight_for_epoch(epoch_idx: int) -> float:
    if SISDR_WARMUP_EPOCHS <= 0:
        return SI_SDR_WEIGHT_MAX
    if epoch_idx <= SISDR_WARMUP_EPOCHS:
        return 0.0
    # Linear ramp over 5 epochs after warmup
    ramp = min(1.0, (epoch_idx - SISDR_WARMUP_EPOCHS) / 5.0)
    return SI_SDR_WEIGHT_MAX * ramp


def _forward_losses(model, mvdr_wav, clean_wav, epoch_idx: int):
    mvdr_mag, mvdr_complex = stft_mag(mvdr_wav, DEVICE)
    clean_mag, _ = stft_mag(clean_wav, DEVICE)

    # Build model input as [real, imag]
    x = torch.stack([mvdr_complex.real, mvdr_complex.imag], dim=1)  # [B,2,F,T]

    # Predict complex mask [B,2,F,T]
    m = model(x)
    m_r = m[:, 0, :, :]
    m_i = m[:, 1, :, :]

    x_r = mvdr_complex.real
    x_i = mvdr_complex.imag

    # Complex multiply: Y = M * X
    y_r = m_r * x_r - m_i * x_i
    y_i = m_r * x_i + m_i * x_r
    est_complex = torch.complex(y_r, y_i)

    # Waveform estimate
    est_wav = istft(est_complex, DEVICE, length=clean_wav.shape[-1])

    # Losses
    est_mag = torch.abs(est_complex).unsqueeze(1)
    clean_mag = clean_mag.unsqueeze(1)
    loss_logmag = log_mag_loss(est_mag, clean_mag)
    loss_sisdr = si_sdr_loss(est_wav, clean_wav)
    loss_mrstft = mrstft_loss(est_wav, clean_wav)

    # Baseline SI-SDR (MVDR vs clean) for debugging / improvement tracking.
    baseline_sisdr_loss = si_sdr_loss(mvdr_wav, clean_wav)

    # If the enhanced SI-SDR is worse than baseline, loss_sisdr > baseline_sisdr_loss.
    # Penalize that region to stabilize training.
    improvement_hinge = torch.relu(loss_sisdr - baseline_sisdr_loss)

    si_sdr_weight = _sisdr_weight_for_epoch(epoch_idx)

    total_loss = (
        si_sdr_weight * loss_sisdr
        + MRSTFT_WEIGHT * loss_mrstft
        + LOGMAG_WEIGHT * loss_logmag
        + IMPROVEMENT_HINGE_WEIGHT * improvement_hinge
    )

    # Return additional values for logging.
    return total_loss, loss_logmag, loss_sisdr, baseline_sisdr_loss


def train_one_epoch(model, optimizer, train_loader, epoch_idx):
    model.train()

    total = 0.0
    total_mag = 0.0
    total_sisdr = 0.0
    total_base_sisdr = 0.0
    n = 0

    pbar = tqdm(train_loader, desc=f"Train {epoch_idx:03d}", leave=False)
    for batch_idx, (mvdr_wav, clean_wav) in enumerate(pbar, start=1):
        mvdr_wav = mvdr_wav.to(DEVICE)
        clean_wav = clean_wav.to(DEVICE)

        loss, loss_mag, loss_sisdr, base_sisdr = _forward_losses(model, mvdr_wav, clean_wav, epoch_idx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total += loss.item()
        total_mag += loss_mag.item()
        total_sisdr += loss_sisdr.item()
        total_base_sisdr += base_sisdr.item()
        n += 1

        # si_sdr_loss is negative SI-SDR. Convert to dB for display.
        est_db = -(total_sisdr / n)
        base_db = -(total_base_sisdr / n)
        imp_db = est_db - base_db

        pbar.set_postfix(
            loss=f"{total/n:.3f}",
            mag=f"{total_mag/n:.3f}",
            sisdr_db=f"{est_db:.2f}",
            imp_db=f"{imp_db:.2f}",
        )

        if MAX_TRAIN_BATCHES_PER_EPOCH is not None and batch_idx >= MAX_TRAIN_BATCHES_PER_EPOCH:
            break

    denom = max(1, n)
    return total / denom, total_mag / denom, total_sisdr / denom, total_base_sisdr / denom


@torch.no_grad()
def validate(model, val_loader, epoch_idx):
    model.eval()

    total = 0.0
    total_mag = 0.0
    total_sisdr = 0.0
    total_base_sisdr = 0.0
    n = 0

    pbar = tqdm(val_loader, desc=f"Val   {epoch_idx:03d}", leave=False)
    for batch_idx, (mvdr_wav, clean_wav) in enumerate(pbar, start=1):
        mvdr_wav = mvdr_wav.to(DEVICE)
        clean_wav = clean_wav.to(DEVICE)

        loss, loss_mag, loss_sisdr, base_sisdr = _forward_losses(model, mvdr_wav, clean_wav, epoch_idx)

        total += loss.item()
        total_mag += loss_mag.item()
        total_sisdr += loss_sisdr.item()
        total_base_sisdr += base_sisdr.item()
        n += 1

        est_db = -(total_sisdr / n)
        base_db = -(total_base_sisdr / n)
        imp_db = est_db - base_db

        pbar.set_postfix(
            loss=f"{total/n:.3f}",
            mag=f"{total_mag/n:.3f}",
            sisdr_db=f"{est_db:.2f}",
            imp_db=f"{imp_db:.2f}",
        )

        if MAX_VAL_BATCHES_PER_EPOCH is not None and batch_idx >= MAX_VAL_BATCHES_PER_EPOCH:
            break

    denom = max(1, n)
    return total / denom, total_mag / denom, total_sisdr / denom, total_base_sisdr / denom


def main():
    _validate_paths()
    train_loader, val_loader = _make_loaders()

    model = CRN_Light().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Device: {DEVICE}")
    print(f"Train/Val sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Val full utterance: {VAL_FULL_UTTERANCE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_mag, tr_sisdr, tr_base = train_one_epoch(model, optimizer, train_loader, epoch)
        va_loss, va_mag, va_sisdr, va_base = validate(model, val_loader, epoch)

        tr_sisdr_db = -tr_sisdr
        va_sisdr_db = -va_sisdr
        tr_base_db = -tr_base
        va_base_db = -va_base

        # Note: "accuracy" isn't meaningful for speech enhancement; SI-SDR is a standard quality metric.
        print(
            f"Epoch {epoch:03d} | "
            f"train: loss={tr_loss:.4f}, base={tr_base_db:.2f}dB, enh={tr_sisdr_db:.2f}dB, imp={tr_sisdr_db - tr_base_db:.2f}dB | "
            f"val: loss={va_loss:.4f}, base={va_base_db:.2f}dB, enh={va_sisdr_db:.2f}dB, imp={va_sisdr_db - va_base_db:.2f}dB"
        )

        if SAVE_EVERY_EPOCHS and (epoch % SAVE_EVERY_EPOCHS == 0):
            torch.save(model.state_dict(), f"{CHECKPOINT_PREFIX}_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()