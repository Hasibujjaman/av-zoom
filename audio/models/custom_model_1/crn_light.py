import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual2d(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        kernel_size=(3, 3),
        causal: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        hidden = channels * expansion
        self.causal = causal
        self.kernel_size = kernel_size

        self.expand = nn.Conv2d(channels, hidden, kernel_size=1, bias=bias)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=0 if causal else (kernel_size[0] // 2, kernel_size[1] // 2),
            groups=hidden,
            bias=bias,
        )
        self.project = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x
        x = F.relu(self.expand(x))

        if self.causal:
            # Causal along time (last dim). Keep freq padding symmetric.
            pad_f = self.kernel_size[0] // 2
            pad_t = self.kernel_size[1] - 1
            x = F.pad(x, (pad_t, 0, pad_f, pad_f))

        x = F.relu(self.dw(x))
        x = self.project(x)
        return x + residual


class CRN_Light(nn.Module):
    """
    Tiny CRN-like U-Net for MVDR post-enhancement
    ~0.15M parameters (target for on-device)
    Input/Output: [B, 1, 257, T]
    """

    def __init__(
        self,
        freq_bins: int = 257,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 16,
        bottleneck_channels: int = 48,
        bottleneck_blocks: int = 6,
        bottleneck_expansion: int = 4,
        causal: bool = False,
    ):
        super().__init__()

        # -------- Encoder (depthwise-separable) --------
        c1 = base_channels
        c2 = int(round(base_channels * 1.5))
        c3 = int(round(base_channels * 2.0))
        c4 = bottleneck_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.enc1 = DepthwiseSeparableConv2d(in_channels, c1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.enc2 = DepthwiseSeparableConv2d(c1, c2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc3 = DepthwiseSeparableConv2d(c2, c3, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc4 = DepthwiseSeparableConv2d(c3, c4, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))

        # Three strided layers reduce F as: F_out = floor((F_in + 1)/2) applied 3 times (== ceil(F/8)).
        self.freq_reduced = (freq_bins + 7) // 8

        # -------- Bottleneck (inverted residual blocks) --------
        self.bottleneck = nn.Sequential(
            *[
                InvertedResidual2d(
                    channels=c4,
                    expansion=bottleneck_expansion,
                    kernel_size=(3, 3),
                    causal=causal,
                )
                for _ in range(bottleneck_blocks)
            ]
        )

        # -------- Decoder (upsample + depthwise-separable conv) --------
        self.up4 = nn.Upsample(scale_factor=(2, 1), mode="nearest")
        self.dec4 = DepthwiseSeparableConv2d(c4 + c3, c3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.up3 = nn.Upsample(scale_factor=(2, 1), mode="nearest")
        self.dec3 = DepthwiseSeparableConv2d(c3 + c2, c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.up2 = nn.Upsample(scale_factor=(2, 1), mode="nearest")
        self.dec2 = DepthwiseSeparableConv2d(c2 + c1, c1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

        # Start close to identity for complex masking: m_r≈1, m_i≈0.
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # x: [B,1,F,T]

        def _align_to(ref, y):
            """Crop/pad y on (freq,time) dims to match ref."""
            ref_f, ref_t = ref.shape[2], ref.shape[3]
            y_f, y_t = y.shape[2], y.shape[3]

            if y_f > ref_f:
                y = y[:, :, :ref_f, :]
            elif y_f < ref_f:
                y = F.pad(y, (0, 0, 0, ref_f - y_f))

            if y_t > ref_t:
                y = y[:, :, :, :ref_t]
            elif y_t < ref_t:
                y = F.pad(y, (0, ref_t - y_t, 0, 0))

            return y

        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        e4 = F.relu(self.enc4(e3))

        b = self.bottleneck(e4)

        d4 = self.up4(b)
        d4 = _align_to(e3, d4)
        d4 = F.relu(self.dec4(torch.cat([d4, e3], dim=1)))

        d3 = self.up3(d4)
        d3 = _align_to(e2, d3)
        d3 = F.relu(self.dec3(torch.cat([d3, e2], dim=1)))

        d2 = self.up2(d3)
        d2 = _align_to(e1, d2)
        d2 = F.relu(self.dec2(torch.cat([d2, e1], dim=1)))

        mask = torch.tanh(self.out(d2))

        # If using complex mask (2 channels), keep real part near 1 and imag near 0.
        if mask.shape[1] == 2:
            m_r = 1.0 + 0.5 * mask[:, 0:1]
            m_i = 0.5 * mask[:, 1:2]
            return torch.cat([m_r, m_i], dim=1)

        # Otherwise (e.g., 1-channel magnitude mask), keep it bounded.
        return mask
    


if __name__ == "__main__":
    model = CRN_Light()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {n_params:.2f}M")

    x = torch.randn(1,2,257,200)
    y = model(x)
    print("Output shape:", y.shape)