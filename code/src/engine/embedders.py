import torch
import math
import torch.nn as nn


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder(nn.Module):
    def __init__(self, input_dims, num_freq, include_input=True, log_sampling=True):
        super().__init__()
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = num_freq - 1
        self.num_freq = num_freq
        self.log_sampling = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]
        self.construct()

    def construct(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freq

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def step(self):
        pass

    def eval(self):
        pass

    def embed(self, inputs):
        encoded_x = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return encoded_x


class BarfEmbedder(Embedder):
    def __init__(
        self,
        input_dims,
        num_freq,
        start,
        end,
        dev,
        include_input=True,
        log_sampling=True,
        # no_ft_pose=False,
        no_barf=False,
    ):
        super().__init__(
            input_dims, num_freq, include_input=include_input, log_sampling=log_sampling
        )

        self.no_barf = no_barf
        self.start = start
        self.end = end
        self.dev = dev
        self.alphas = torch.cat(
            (
                torch.zeros(self.start),
                torch.linspace(0, self.num_freq, self.end - self.start),
            ),
            dim=0,
        )

        self.register_buffer("alpha_iter", torch.tensor(0))
        self.alpha = self.alphas[self.alpha_iter]
        self.register_buffer("alpha_max_iter", torch.tensor(len(self.alphas)))
        self.populate_barf_weights(self.alpha)

    def populate_barf_weights(self, alpha):
        self.barf_weights = self.compute_barf_weights(
            alpha, self.num_freq, self.input_dims, len(self.periodic_fns), self.dev
        )

    def compute_barf_weights(self, alpha, L, input_dim, sin_dim, dev):
        # alpha in [0, L-1]
        # L: num freq
        # output: weights with len(weights) ==  L
        k = torch.arange(L, dtype=torch.float32, device=dev)
        ak = alpha - k
        weights = torch.clamp(ak, 0, 1)
        cos_idx = torch.logical_and(0 <= ak, ak < 1)
        cos_val = (1 - torch.cos(ak * math.pi)) / 2
        weights[cos_idx] = cos_val[cos_idx]

        weights = weights[:, None].repeat(1, input_dim * sin_dim).view(-1)
        weights = torch.cat((torch.ones(input_dim, device=dev), weights), dim=0)
        return weights

    def step(self):
        self.alpha_iter = min(self.alpha_iter + 1, self.alpha_max_iter - 1)
        self.alpha = self.alphas[self.alpha_iter]

        self.populate_barf_weights(self.alpha)
        """
        print(f'alpha_iter = {self.alpha_iter}')
        print(f'alpha = {self.alpha}')
        print(self.barf_weights)
        """

    def embed(self, inputs):
        encoded_x = super().embed(inputs)
        if not self.no_barf:
            encoded_x = encoded_x * self.barf_weights[None, :]
        return encoded_x

    def eval(self):
        self.no_barf = True


def get_embedder(multires, mode, input_dims=3, barf_s=None, barf_e=None, no_barf=False):
    specs = {
        "include_input": True,
        "input_dims": input_dims,
        "num_freqs": multires,
        "log_sampling": True,
    }

    if mode == "fourier":
        embedder_obj = Embedder(
            input_dims=specs["input_dims"],
            num_freq=specs["num_freqs"],
            include_input=specs["include_input"],
            log_sampling=specs["log_sampling"],
        )
        out_dim = embedder_obj.out_dim
        # assert False, "should only use barf"
    elif mode == "barf":
        embedder_obj = BarfEmbedder(
            input_dims=specs["input_dims"],
            num_freq=specs["num_freqs"],
            include_input=specs["include_input"],
            log_sampling=specs["log_sampling"],
            start=barf_s,
            end=barf_e,
            dev=torch.device("cuda"),
            no_barf=no_barf,
        )
        out_dim = embedder_obj.out_dim
    else:
        assert False, f"Unknown mode {mode}"

    return embedder_obj, out_dim
