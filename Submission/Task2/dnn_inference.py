#!/usr/bin/env python3
"""
dnn_inference.py — RealNet V2 speech enhancement inference.
SP Cup 2026 Phase 2

Enhances MVDR+PF beamformer output using a trained DNN model.

Usage:
    python3 dnn_inference.py --input in.wav --output out.wav --checkpoint model.pt
"""

import argparse
import importlib
import inspect
import os
import site
import sys
import subprocess


def _ensure_packages():
    """Install any missing dependencies."""
    required = {
        'numpy': 'numpy',
        'torch': 'torch',
        'soundfile': 'soundfile',
        'torchaudio': 'torchaudio',
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        importlib.invalidate_caches()
        site.main()
        user_sp = site.getusersitepackages()
        if isinstance(user_sp, str) and user_sp not in sys.path:
            sys.path.insert(0, user_sp)
        elif isinstance(user_sp, list):
            for p in user_sp:
                if p not in sys.path:
                    sys.path.insert(0, p)


_ensure_packages()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    sf = None
    _HAS_SF = False

try:
    import torchaudio
    _HAS_TA = True
except ImportError:
    torchaudio = None
    _HAS_TA = False

# STFT config
N_FFT = 512
HOP = 128
WIN_LEN = 512
SR = 16000


def _window(n, device):
    return torch.sqrt(torch.hann_window(n, periodic=True, device=device))


def do_stft(wav, device):
    w = _window(WIN_LEN, device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN_LEN,
                      window=w, center=True, return_complex=True)


def do_istft(spec, device, length=None):
    w = _window(WIN_LEN, device)
    if spec.dim() == 2:
        spec = spec.unsqueeze(0)
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP, win_length=WIN_LEN,
                       window=w, center=True, length=length)


# ---- Model components ----

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=True):
        super().__init__()
        assert in_features % groups == 0 and out_features % groups == 0
        self.groups = groups
        self.in_g = in_features // groups
        self.out_g = out_features // groups
        self.weight = nn.Parameter(torch.randn(groups, self.out_g, self.in_g) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        bshape = x.shape[:-1]
        x = x.view(*bshape, self.groups, self.in_g)
        x = torch.einsum('...gi,goi->...go', x, self.weight)
        x = x.view(*bshape, -1)
        if self.bias is not None:
            x = x + self.bias
        return x


class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch * 2, ks, padding=ks // 2, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class ERBEncoder(nn.Module):
    def __init__(self, n_freqs, n_erb, hdim, n_layers=2):
        super().__init__()
        self.erb_proj = nn.Linear(n_freqs, n_erb)
        layers = []
        ind = n_erb
        for i in range(n_layers):
            od = hdim if i == n_layers - 1 else n_erb * 2
            layers.append(ConvGLU(ind, od, ks=3))
            layers.append(nn.BatchNorm1d(od))
            ind = od
        self.layers = nn.Sequential(*layers)

    def forward(self, mag):
        mag = torch.log1p(mag * 10)
        erb = self.erb_proj(mag.transpose(1, 2)).transpose(1, 2)
        return self.layers(erb)


class TemporalGRU(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(in_dim, h_dim, num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout if n_layers > 1 else 0,
                          bidirectional=False)
        self.proj = nn.Linear(h_dim, in_dim) if h_dim != in_dim else nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = self.proj(x)
        return x.transpose(1, 2)


class SpectralFilterModule(nn.Module):
    """Multi-tap complex FIR filter in frequency domain."""
    def __init__(self, hdim, n_freqs, filt_order=5, filt_bins=128):
        super().__init__()
        self.filt_order = filt_order
        self.filt_bins = filt_bins
        coef_dim = 2 * filt_order * filt_bins

        self.gain_proj = nn.Sequential(
            nn.Linear(hdim, hdim // 2), nn.ReLU(),
            nn.Linear(hdim // 2, n_freqs), nn.Sigmoid())
        self.coef_proj = nn.Sequential(
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, coef_dim))

        nn.init.zeros_(self.coef_proj[-1].weight)
        nn.init.zeros_(self.coef_proj[-1].bias)

    def forward(self, features, spec):
        B, _, T = spec.shape
        feat = features.transpose(1, 2)
        gains = self.gain_proj(feat).transpose(1, 2)
        coefs = self.coef_proj(feat).view(B, T, 2, self.filt_order, self.filt_bins)
        cr = coefs[:, :, 0]
        ci = coefs[:, :, 1]

        out = spec * gains

        padded = F.pad(spec[:, :self.filt_bins, :], (self.filt_order - 1, 0))
        filt = torch.zeros(B, self.filt_bins, T, dtype=spec.dtype, device=spec.device)

        for k in range(self.filt_order):
            frame = padded[:, :, self.filt_order - 1 - k:self.filt_order - 1 - k + T]
            r = cr[:, :, k, :].transpose(1, 2)
            i = ci[:, :, k, :].transpose(1, 2)
            filt = filt + torch.complex(r*frame.real - i*frame.imag,
                                        r*frame.imag + i*frame.real)

        out[:, :self.filt_bins, :] = 0.5 * filt + 0.5 * out[:, :self.filt_bins, :]
        return out


class RealNetV2(nn.Module):
    """
    Lightweight causal speech enhancement network.
    ERB encoder -> GRU -> spectral filter + gain mask.
    """
    def __init__(self, n_fft=512, n_erb_bands=40, hidden_dim=80, gru_dim=128,
                 gru_layers=2, df_order=5, df_bins=128, dropout=0.1):
        super().__init__()
        self.n_freqs = n_fft // 2 + 1
        self.erb_encoder = ERBEncoder(self.n_freqs, n_erb_bands, hidden_dim, n_layers=2)
        self.temporal_gru = TemporalGRU(hidden_dim, gru_dim, n_layers=gru_layers, dropout=dropout)
        self.spectral_filter = SpectralFilterModule(
            hidden_dim, self.n_freqs, df_order, min(df_bins, self.n_freqs))

    def forward(self, spec):
        mag = torch.abs(spec)
        feat = self.erb_encoder(mag)
        feat = self.temporal_gru(feat)
        return self.spectral_filter(feat, spec)


# ---- I/O ----

def load_wav(path):
    if _HAS_SF:
        data, sr = sf.read(path, dtype='float32')
        if sr != SR:
            raise ValueError(f"Expected {SR}Hz, got {sr}Hz")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data
    elif _HAS_TA:
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.mean(dim=0).numpy()
    raise ImportError("Need soundfile or torchaudio")


def save_wav(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if _HAS_SF:
        sf.write(path, data, SR)
    elif _HAS_TA:
        torchaudio.save(path, torch.from_numpy(data).unsqueeze(0), SR)
    else:
        raise ImportError("Need soundfile or torchaudio")


def load_model(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    defaults = dict(n_fft=N_FFT, n_erb_bands=40, hidden_dim=80, gru_dim=128,
                    gru_layers=2, df_order=5, df_bins=128, dropout=0.1)
    cfg = ckpt.get('model_config', defaults)

    # only keep valid constructor args
    valid = set(inspect.signature(RealNetV2.__init__).parameters.keys()) - {'self'}
    cfg = {k: v for k, v in cfg.items() if k in valid}

    model = RealNetV2(**cfg).to(device)
    state = ckpt.get('model_state', ckpt)

    # remap old checkpoint keys to new attribute names
    remapped = {}
    for k, v in state.items():
        k2 = k.replace('deep_filter.', 'spectral_filter.')
        k2 = k2.replace('.df_proj.', '.coef_proj.')
        remapped[k2] = v

    model.load_state_dict(remapped)
    model.eval()
    return model


@torch.no_grad()
def enhance(wav_np, model, device='cpu'):
    wav = torch.from_numpy(wav_np).float().to(device)
    n = wav.shape[0]
    scale = wav.std() + 1e-8
    spec = do_stft(wav / scale, device)
    out = model(spec)
    y = do_istft(out, device, length=n).squeeze(0) * scale
    return y.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description='RealNet V2 inference')
    ap.add_argument('--input', required=True, help='Input wav')
    ap.add_argument('--output', required=True, help='Output wav')
    ap.add_argument('--checkpoint', required=True, help='Checkpoint .pt')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: {args.checkpoint} not found", file=sys.stderr)
        sys.exit(1)

    model = load_model(args.checkpoint, args.device)
    wav = load_wav(args.input)
    out = enhance(wav, model, args.device)

    pk = np.max(np.abs(out))
    if pk > 0:
        out = 0.99 * out / pk

    save_wav(args.output, out)
    print(f"Saved: {args.output} ({len(out)} samples)")


if __name__ == '__main__':
    main()
