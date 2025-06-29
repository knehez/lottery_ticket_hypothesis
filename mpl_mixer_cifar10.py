from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
from tqdm import trange
import time
import numpy as np
import copy

# --- MLP-Mixer modules ---
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)  # bias visszaállítva
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)  # bias visszaállítva

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, token_dim, channel_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = MLP(num_patches, token_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = MLP(dim, channel_dim)

    def forward(self, x):
        # x: [B, N, D]
        y = self.norm1(x).transpose(1, 2)  # [B, D, N]
        x = x + self.token_mlp(y).transpose(1, 2)  # token_mlp(y): [B, D, N] -> transpose: [B, N, D]
        x = x + self.channel_mlp(self.norm2(x))  # channel_mlp(norm2(x)): [B, N, D]
        # Output: [B, N, D]
        return x

class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=8, dim=192, depth=8, num_classes=10, token_dim=96, channel_dim=768, in_channels=3):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding = nn.Linear(patch_dim, dim)
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(num_patches, dim, token_dim, channel_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)  # [B, C, n_patches_h, n_patches_w, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, n_patches_h, n_patches_w, C, p, p]
        x = x.view(B, -1, C * p * p)  # [B, N, patch_dim], N = num_patches
        x = self.embedding(x)  # [B, N, D]
        x = self.mixer_blocks(x)  # [B, N, D]
        x = self.norm(x)  # [B, N, D]
        x = x.mean(dim=1)  # [B, D]
        return self.classifier(x)  # [B, num_classes]

# --- LTH components ---
def train(model, optimizer, dataloader, device):
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total_infer_time = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            start = time.perf_counter()
            output = model(data)
            total_infer_time += time.perf_counter() - start
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(dataloader.dataset), total_infer_time


def correlation_mask(weight_tensor: torch.Tensor, global_mask=None, relative_margin=0.15, verbose=True, layer_name="layer", axis=0):
    """
    Általános korrelációalapú maszkolás.
    - axis=0: sorok (kimeneti neuronok) prunolása
    - axis=1: oszlopok (bemeneti neuronok) prunolása
    """
    W = weight_tensor.detach().cpu().numpy()

    if global_mask is not None:
        W = W * global_mask.cpu().numpy()

    if axis == 0:
        norms = np.linalg.norm(W, axis=1)
        nonzero_indices = np.where(norms > 1e-6)[0]
        data_for_corr = W[nonzero_indices]
    elif axis == 1:
        norms = np.linalg.norm(W, axis=0)
        nonzero_indices = np.where(norms > 1e-6)[0]
        data_for_corr = W[:, nonzero_indices].T
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    if len(nonzero_indices) < 2:
        if verbose:
            print(f"  [Debug] {layer_name}: kevés nem-nulla {'sor' if axis==0 else 'oszlop'} maradt, nincs értelme korrelációt számolni.")
        return global_mask.clone() if global_mask is not None else torch.ones_like(weight_tensor)

    corr_matrix = np.corrcoef(data_for_corr)
    np.fill_diagonal(corr_matrix, 0.0)

    max_corr = np.nanmax(np.abs(corr_matrix))
    threshold = max(max_corr - relative_margin, 0.0)

    if verbose:
        print(f"  [Debug] {layer_name}: max(abs(corr)) = {max_corr:.4f}, threshold = {threshold:.4f}")

    mask = np.ones_like(W)
    already_pruned = set()

    for i in range(len(nonzero_indices)):
        for j in range(i + 1, len(nonzero_indices)):
            if abs(corr_matrix[i, j]) > threshold:
                j_idx = int(nonzero_indices[j])
                if j_idx not in already_pruned:
                    if axis == 0:
                        mask[j_idx, :] = 0.0
                    else:
                        mask[:, j_idx] = 0.0
                    already_pruned.add(j_idx)

    if verbose:
        print(f"  [Debug] {layer_name}: nullázva ({'sor' if axis==0 else 'oszlop'} indexek): {sorted(list(already_pruned))}")

    final_mask = torch.tensor(mask, dtype=torch.float32, device=weight_tensor.device)
    if global_mask is not None:
        final_mask *= global_mask
    return final_mask

def propagate_mask(prev_mask: torch.Tensor, next_weight: torch.Tensor) -> torch.Tensor:
    row_sums = prev_mask.sum(dim=1)
    zero_rows = (row_sums == 0).nonzero(as_tuple=False).squeeze()
    next_mask = torch.ones_like(next_weight)
    if zero_rows.numel() > 0:
        next_mask[:, zero_rows] = 0.0
    return next_mask

def prune_mixer_model(model, global_masks, relative_margin=0.15, verbose=True):
    """
    Korrelációalapú pruning minden MixerBlock token_mlp és channel_mlp részén.
    Csak az fc1 rétegekben történik korrelációszámítás.
    Az fc2 rétegek maszkolása az fc1 kiesett kimenetei alapján történik.
    """
    new_global_masks = []

    for i, block in enumerate(model.mixer_blocks):
        device = block.token_mlp.fc1.weight.device

        # === TOKEN MLP ===
        mask_token_mlp_fc1 = correlation_mask(
            block.token_mlp.fc1.weight,
            global_mask=global_masks[i]['token_mlp.fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"token_mlp_fc1_block_{i}"
        )
        mask_token_mlp_fc2 = global_masks[i]['token_mlp.fc2'] * propagate_mask(mask_token_mlp_fc1, block.token_mlp.fc2.weight).to(device)

        # === CHANNEL MLP ===
        mask_channel_mlp_fc1 = correlation_mask(
            block.channel_mlp.fc1.weight,
            global_mask=global_masks[i]['channel_mlp.fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"channel_mlp_fc1_block_{i}"
        )
        mask_channel_mlp_fc2 = global_masks[i]['channel_mlp.fc2'] * propagate_mask(mask_channel_mlp_fc1, block.channel_mlp.fc2.weight).to(device)

        # === APPLY ALL MASKS (bias is masked if all weights in a row are zero) ===
        with torch.no_grad():
            # token_mlp.fc1
            block.token_mlp.fc1.weight *= mask_token_mlp_fc1
            if block.token_mlp.fc1.bias is not None:
                for idx in range(block.token_mlp.fc1.weight.shape[0]):
                    if torch.all(block.token_mlp.fc1.weight[idx] == 0):
                        block.token_mlp.fc1.bias[idx] = 0.0
            # token_mlp.fc2
            block.token_mlp.fc2.weight *= mask_token_mlp_fc2
            if block.token_mlp.fc2.bias is not None:
                for idx in range(block.token_mlp.fc2.weight.shape[0]):
                    if torch.all(block.token_mlp.fc2.weight[idx] == 0):
                        block.token_mlp.fc2.bias[idx] = 0.0
            # channel_mlp.fc1
            block.channel_mlp.fc1.weight *= mask_channel_mlp_fc1
            if block.channel_mlp.fc1.bias is not None:
                for idx in range(block.channel_mlp.fc1.weight.shape[0]):
                    if torch.all(block.channel_mlp.fc1.weight[idx] == 0):
                        block.channel_mlp.fc1.bias[idx] = 0.0
            # channel_mlp.fc2
            block.channel_mlp.fc2.weight *= mask_channel_mlp_fc2
            if block.channel_mlp.fc2.bias is not None:
                for idx in range(block.channel_mlp.fc2.weight.shape[0]):
                    if torch.all(block.channel_mlp.fc2.weight[idx] == 0):
                        block.channel_mlp.fc2.bias[idx] = 0.0

        # === UPDATE GLOBAL MASKS ===
        new_global_masks.append({
            'token_mlp.fc1': global_masks[i]['token_mlp.fc1'] * mask_token_mlp_fc1,
            'token_mlp.fc2': global_masks[i]['token_mlp.fc2'] * mask_token_mlp_fc2,
            'channel_mlp.fc1': global_masks[i]['channel_mlp.fc1'] * mask_channel_mlp_fc1,
            'channel_mlp.fc2': global_masks[i]['channel_mlp.fc2'] * mask_channel_mlp_fc2
        })

    return new_global_masks

def build_fully_pruned_mixer_model(model, global_masks):
    """
    Új MLPMixer modellt épít a pruning maszkok alapján, megtartva a neuronok sorrendjét.
    Csak a token_mlp.fc1 és channel_mlp.fc1 output dimenzióit csökkentjük.
    """
    device = next(model.parameters()).device
    new_blocks = []

    for i, block in enumerate(model.mixer_blocks):
        # === Maszkok ===
        token_mlp_fc1_mask = global_masks[i]['token_mlp.fc1']
        token_mlp_fc2_mask = global_masks[i]['token_mlp.fc2']
        channel_mlp_fc1_mask = global_masks[i]['channel_mlp.fc1']
        channel_mlp_fc2_mask = global_masks[i]['channel_mlp.fc2']

        # === Aktív neuronindexek (megőrizzük sorrendet) ===
        token_mlp_fc1_rows = (token_mlp_fc1_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        channel_mlp_fc1_rows = (channel_mlp_fc1_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        # Dimenziók
        num_patches = block.token_mlp.fc1.in_features
        dim = block.channel_mlp.fc1.in_features
        token_dim = len(token_mlp_fc1_rows)
        channel_dim = len(channel_mlp_fc1_rows)

        # === Új blokk létrehozása ===
        new_block = MixerBlock(
            num_patches=num_patches,
            dim=dim,
            token_dim=token_dim,
            channel_dim=channel_dim
        ).to(device)

        # === Súlyok másolása ===
        with torch.no_grad():
            # TOKEN MLP
            new_block.token_mlp.fc1.weight.copy_(block.token_mlp.fc1.weight[token_mlp_fc1_rows])
            new_block.token_mlp.fc1.bias.copy_(block.token_mlp.fc1.bias[token_mlp_fc1_rows])
            new_block.token_mlp.fc2.weight.copy_(block.token_mlp.fc2.weight[:, token_mlp_fc1_rows])
            new_block.token_mlp.fc2.bias.copy_(block.token_mlp.fc2.bias)

            # CHANNEL MLP
            new_block.channel_mlp.fc1.weight.copy_(block.channel_mlp.fc1.weight[channel_mlp_fc1_rows])
            new_block.channel_mlp.fc1.bias.copy_(block.channel_mlp.fc1.bias[channel_mlp_fc1_rows])
            new_block.channel_mlp.fc2.weight.copy_(block.channel_mlp.fc2.weight[:, channel_mlp_fc1_rows])
            new_block.channel_mlp.fc2.bias.copy_(block.channel_mlp.fc2.bias)