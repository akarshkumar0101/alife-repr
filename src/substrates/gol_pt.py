import torch


def int2binary(x, n_bits=18, bits_per_token=1):
    bit_positions = 2 ** torch.arange(n_bits, device=x.device)
    binary = torch.bitwise_and(x, bit_positions) > 0
    tokens = binary.reshape(-1, bits_per_token)
    tokens = (tokens * (2 ** torch.arange(bits_per_token, device=x.device))).sum(dim=-1)
    return tokens

def conv2d_3x3_sum(x):
    """x.shape = (B, H, W)"""
    kernel = torch.ones((1, 1, 3, 3), device=x.device, dtype=x.dtype)
    x_padded = torch.nn.functional.pad(x[:, None], (1, 1, 1, 1), mode="circular")
    return torch.nn.functional.conv2d(x_padded, kernel, padding="valid")[:, 0]


def run_game_of_life(params, batch_size=32, grid_size=64, t_steps=128):
    params = int2binary(params)
    sparsity = torch.rand((batch_size, 1, 1), device=params.device) * 0.35 + 0.05
    state = torch.rand(batch_size, grid_size, grid_size, device=params.device)
    state = torch.floor(state + sparsity).int()

    state_video = []
    for i in range(t_steps):
        state_video.append(state)
        n_neighbors = conv2d_3x3_sum(state.float()).int() - state
        update_idx = state * 9 + n_neighbors
        state = params[update_idx]
    state_video = torch.stack(state_video, dim=1)
    return state_video
