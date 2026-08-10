import torch
import torch.nn.functional as F

def log_mag_loss(est, ref):
    return F.l1_loss(
        torch.log(est + 1e-8),
        torch.log(ref + 1e-8)
    )

def si_sdr_loss(est, ref, eps=1e-8):
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    proj = (torch.sum(est * ref, dim=-1, keepdim=True) * ref) / \
           (torch.sum(ref ** 2, dim=-1, keepdim=True) + eps)

    noise = est - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)

    return -10 * torch.log10(ratio + eps).mean()


def _stft(x, n_fft, hop_length, win_length, window):
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )


def mrstft_loss(
    est_wav: torch.Tensor,
    ref_wav: torch.Tensor,
    fft_sizes=(512, 1024, 2048),
    hop_ratios=(0.25, 0.25, 0.25),
    win_lengths=None,
    eps: float = 1e-8,
):
    """Multi-resolution STFT loss: spectral convergence + log-mag L1.

    This is a common objective for speech enhancement that tends to correlate better with
    perceptual metrics (PESQ/STOI) than single-resolution magnitude loss alone.

    Shapes: est_wav/ref_wav: [B, T]
    """
    if win_lengths is None:
        win_lengths = fft_sizes
    device = est_wav.device

    sc_total = 0.0
    mag_total = 0.0
    n = 0

    for n_fft, hop_ratio, win_length in zip(fft_sizes, hop_ratios, win_lengths):
        hop = int(round(n_fft * hop_ratio))
        window = torch.hann_window(win_length, periodic=True, device=device)

        est = _stft(est_wav, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window)
        ref = _stft(ref_wav, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window)

        est_mag = torch.abs(est)
        ref_mag = torch.abs(ref)

        sc = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + eps)
        mag = F.l1_loss(torch.log(est_mag + eps), torch.log(ref_mag + eps))

        sc_total = sc_total + sc
        mag_total = mag_total + mag
        n += 1

    sc_total = sc_total / max(1, n)
    mag_total = mag_total / max(1, n)
    return sc_total + mag_total