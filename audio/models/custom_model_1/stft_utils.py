import torch


def stft_mag(wav, device=None):
    if device is None:
        device = wav.device
    window = torch.sqrt(
        torch.hann_window(256, periodic=True)
    ).to(device)

    spec = torch.stft(
        wav,
        n_fft=512,
        hop_length=128,
        win_length=256,
        window=window,
        center=True,
        return_complex=True
    )
    return torch.abs(spec), spec

def istft(spec, device=None, length=None):
    if device is None:
        device = spec.device
    window = torch.sqrt(
        torch.hann_window(256, periodic=True)
    ).to(device)

    return torch.istft(
        spec,
        n_fft=512,
        hop_length=128,
        win_length=256,
        window=window,
        center=True,
        length=length,
    )