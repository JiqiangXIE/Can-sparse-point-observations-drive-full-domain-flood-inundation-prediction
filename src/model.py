import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w,
                               dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w,
                               dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_h, :self.modes_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes_h, :self.modes_w], self.weights1)
        out_ft[:, :, -self.modes_h:, :self.modes_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes_h:, :self.modes_w], self.weights2)
        out = torch.fft.irfft2(out_ft, s=(H, W))
        return out.to(x.dtype)


class FNOBlock2d(nn.Module):

    def __init__(self, width, modes_h, modes_w):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes_h, modes_w)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral_conv(x) + self.bypass(x)))


class FNO_LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, modes_h, modes_w):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_process = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1),
            FNOBlock2d(hidden_dim, modes_h, modes_w),
        )
        self.gates = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size=1)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        processed = self.input_process(torch.cat([x, h_prev], dim=1))
        gates = self.gates(processed)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class FNO_LSTM(nn.Module):

    def __init__(self, in_channels, out_channels, modes_h, modes_w, width,
                 lstm_hidden_dim, lstm_num_layers=1, num_fno_layers=4):
        super().__init__()
        self.width = width
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lifting = nn.Conv2d(in_channels, width, kernel_size=1)

        n_encoder = num_fno_layers // 2
        self.encoder = nn.ModuleList([
            FNOBlock2d(width, modes_h, modes_w) for _ in range(n_encoder)])

        self.fno_lstm = FNO_LSTMCell(width, lstm_hidden_dim, modes_h, modes_w)

        n_decoder = num_fno_layers // 2
        self.decoder = nn.ModuleList([
            FNOBlock2d(lstm_hidden_dim, modes_h, modes_w)
            for _ in range(n_decoder)])

        self.projection = nn.Conv2d(lstm_hidden_dim, out_channels, kernel_size=1)
        nn.init.zeros_(self.projection.bias)
        self._pad_cache = {}

    @staticmethod
    def _next_fft_friendly(n):
        while True:
            m = n
            for p in [2, 3, 5]:
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
            n += 1

    def _get_pad_size(self, H, W):
        key = (H, W)
        if key not in self._pad_cache:
            self._pad_cache[key] = (self._next_fft_friendly(H) - H,
                                    self._next_fft_friendly(W) - W)
        return self._pad_cache[key]

    def init_hidden(self, batch_size, height, width, device):
        return (torch.zeros(batch_size, self.lstm_hidden_dim, height, width,
                            device=device),
                torch.zeros(batch_size, self.lstm_hidden_dim, height, width,
                            device=device))

    def forward(self, x, hidden_state=None):
        B, T, C, H, W = x.shape
        pad_h, pad_w = self._get_pad_size(H, W)
        if pad_h > 0 or pad_w > 0:
            x = x.reshape(B * T, C, H, W)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            x = x.reshape(B, T, C, H + pad_h, W + pad_w)
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W

        if hidden_state is None:
            hidden_state = self.init_hidden(B, H_pad, W_pad, x.device)

        outputs = []
        for t in range(T):
            h = self.lifting(x[:, t])
            for block in self.encoder:
                h = block(h)
            h_lstm, c_lstm = self.fno_lstm(h, hidden_state)
            hidden_state = (h_lstm, c_lstm)
            h = h_lstm
            for block in self.decoder:
                h = block(h)
            outputs.append(self.projection(h))

        out = torch.stack(outputs, dim=1)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :, :H, :W]
        return out, hidden_state


def create_model(in_channels=6, out_channels=2, hp=None, device="cuda"):
    hp = hp or {}
    return FNO_LSTM(
        in_channels=in_channels,
        out_channels=out_channels,
        modes_h=hp.get("fno_modes_h", 12),
        modes_w=hp.get("fno_modes_w", 16),
        width=hp.get("fno_width", 32),
        lstm_hidden_dim=hp.get("lstm_hidden_dim", 64),
        lstm_num_layers=hp.get("lstm_num_layers", 1),
        num_fno_layers=hp.get("num_fno_layers", 4),
    ).to(device)


def set_seed(seed=42):
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
