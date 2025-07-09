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

import torch
import torch.nn as nn

# --- MLP-Mixer modules ---
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches, dim,
        token_dim, token_out_dim,
        channel_dim, channel_out_dim,
        token_in_dim=None, token_out_proj_dim=None,
        channel_in_dim=None, channel_out_proj_dim=None,
    ):
        super().__init__()
        # --- Token mixing ---
        self.norm1 = nn.LayerNorm(dim)
        token_in = token_in_dim if token_in_dim is not None else num_patches
        token_out_proj = token_out_proj_dim if token_out_proj_dim is not None else num_patches
        self.token_in_proj = nn.Linear(token_in, num_patches) if token_in != num_patches else nn.Identity()
        self.token_mlp = MLP(num_patches, token_dim, token_out_dim)
        self.token_out_proj = nn.Linear(token_out_dim, token_out_proj) if token_out_dim != token_out_proj else nn.Identity()
        # --- Channel mixing ---
        self.norm2 = nn.LayerNorm(dim)
        channel_in = channel_in_dim if channel_in_dim is not None else dim
        channel_out_proj = channel_out_proj_dim if channel_out_proj_dim is not None else dim
        self.channel_in_proj = nn.Linear(channel_in, dim) if channel_in != dim else nn.Identity()
        self.channel_mlp = MLP(dim, channel_dim, channel_out_dim)
        self.channel_out_proj = nn.Linear(channel_out_dim, channel_out_proj) if channel_out_dim != channel_out_proj else nn.Identity()

    def forward(self, x):
        # x: [B, N, D]
        y = self.norm1(x).transpose(1, 2)          # [B, D, N]
        y = self.token_in_proj(y)                  # [B, D, N*]
        token_out = self.token_mlp(y)              # [B, N, token_out_dim]
        token_out = self.token_out_proj(token_out).transpose(1, 2)  # [B, N, D]
        x = x + token_out                          # residual

        z = self.norm2(x)
        z = self.channel_in_proj(z)                # [B, N, D*]
        channel_out = self.channel_mlp(z)          # [B, N, channel_out_dim]
        channel_out = self.channel_out_proj(channel_out)   # [B, N, D]
        x = x + channel_out                        # residual
        return x

class MLPMixer(nn.Module):
    def __init__(
        self,
        patch_size=8,
        in_channels=3,
        embedding_dim=128,
        num_classes=10,
        depth=8,
        token_dim=64,
        channel_dim=768,
        mixer_block_dims=None,  # list of tuples: see below!
    ):
        super().__init__()
        patch_dim = patch_size * patch_size * in_channels
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding = nn.Linear(patch_dim, embedding_dim)
        # Alapértelmezett architektúra (nincs pruning):
        if mixer_block_dims is None:
            num_patches = (32 // patch_size) ** 2  # pl. MNIST: 32x32
            mixer_block_dims = [
                # (num_patches, embedding_dim, token_dim, token_out_dim,
                #  channel_dim, channel_out_dim,
                #  token_in_dim, token_out_proj_dim, channel_in_dim, channel_out_proj_dim)
                (num_patches, embedding_dim, token_dim, num_patches,
                 channel_dim, embedding_dim,
                 None, None, None, None)
                for _ in range(depth)
            ]

        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(
                num_patches=cfg[0],
                dim=cfg[1],
                token_dim=cfg[2],
                token_out_dim=cfg[3],
                channel_dim=cfg[4],
                channel_out_dim=cfg[5],
                token_in_dim=cfg[6],
                token_out_proj_dim=cfg[7],
                channel_in_dim=cfg[8],
                channel_out_proj_dim=cfg[9],
            ) for cfg in mixer_block_dims
        ])
        self.norm = nn.LayerNorm(mixer_block_dims[-1][1])
        self.classifier = nn.Linear(mixer_block_dims[-1][1], num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)             # [B, C, nH, nW, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()      # [B, nH, nW, C, p, p]
        x = x.view(B, -1, C * p * p)                      # [B, num_patches, patch_dim]
        x = self.embedding(x)                             # [B, num_patches, embedding_dim]
        x = self.mixer_blocks(x)                          # [B, num_patches, embedding_dim]
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

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
    Most már a token_mlp.fc2 output (axis=0) is korreláció alapján prune-olódik.
    """
    new_global_masks = []

    for i, block in enumerate(model.mixer_blocks):
        device = block.token_mlp.fc1.weight.device

        # === TOKEN MLP FC1 ===
        mask_token_mlp_fc1 = correlation_mask(
            block.token_mlp.fc1.weight,
            global_mask=global_masks[i]['token_mlp.fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"token_mlp_fc1_block_{i}",
            axis=0
        )

        mask_token_mlp_fc1 = correlation_mask(
            block.token_mlp.fc1.weight,
            global_mask=mask_token_mlp_fc1,
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"token_mlp_fc1_block_{i}",
            axis=1
        )

        # === TOKEN MLP FC2 input: fc1 output alapján prune ===
        mask_token_mlp_fc2 = global_masks[i]['token_mlp.fc2'] * propagate_mask(mask_token_mlp_fc1, block.token_mlp.fc2.weight).to(device)

        # === TOKEN MLP FC2 output: KÜLÖN korrelációs pruning (axis=0) ===
        mask_token_mlp_fc2 = correlation_mask(
            block.token_mlp.fc2.weight,
            global_mask=mask_token_mlp_fc2,
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"token_mlp_fc2_block_{i}",
            axis=0
        )
        # Most mask_token_mlp_fc2 tartalmazza az input- és output-pruningot is (mindkét maszkot megszorozva).

        # === CHANNEL MLP ===
        mask_channel_mlp_fc1 = correlation_mask(
            block.channel_mlp.fc1.weight,
            global_mask=global_masks[i]['channel_mlp.fc1'],
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"channel_mlp_fc1_block_{i}",
            axis=0
        )
        mask_channel_mlp_fc2 = global_masks[i]['channel_mlp.fc2'] * propagate_mask(mask_channel_mlp_fc1, block.channel_mlp.fc2.weight).to(device)
        
        mask_channel_mlp_fc1 = correlation_mask(
            block.channel_mlp.fc1.weight,
            global_mask=mask_channel_mlp_fc1,
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"channel_mlp_fc1_block_{i}",
            axis=1
        )
        
        mask_channel_mlp_fc2 = correlation_mask(
            block.channel_mlp.fc2.weight,
            global_mask=mask_channel_mlp_fc2,
            relative_margin=relative_margin,
            verbose=verbose,
            layer_name=f"channel_mlp_fc2_block_{i}",
            axis=0
        )
        
        bias_mask_token_mlp_fc1 = (mask_token_mlp_fc1.sum(dim=1) != 0).float()
        bias_mask_channel_mlp_fc1 = (mask_channel_mlp_fc1.sum(dim=1) != 0).float()

        # === APPLY ALL MASKS ===
        with torch.no_grad():
            block.token_mlp.fc1.weight *= mask_token_mlp_fc1
            block.token_mlp.fc1.bias *= bias_mask_token_mlp_fc1
            
            block.token_mlp.fc2.weight *= mask_token_mlp_fc2
            
            block.channel_mlp.fc1.weight *= mask_channel_mlp_fc1
            block.channel_mlp.fc1.bias *= bias_mask_channel_mlp_fc1
            
            block.channel_mlp.fc2.weight *= mask_channel_mlp_fc2

        # === UPDATE GLOBAL MASKS ===
        new_global_masks.append({
            'token_mlp.fc1': global_masks[i]['token_mlp.fc1'] * mask_token_mlp_fc1,
            'token_mlp.fc1.bias': global_masks[i]['token_mlp.fc1.bias'] * bias_mask_token_mlp_fc1,
            
            'token_mlp.fc2': global_masks[i]['token_mlp.fc2'] * mask_token_mlp_fc2,
            
            'channel_mlp.fc1': global_masks[i]['channel_mlp.fc1'] * mask_channel_mlp_fc1,
            'channel_mlp.fc1.bias': global_masks[i]['channel_mlp.fc1.bias'] * bias_mask_channel_mlp_fc1,
            
            'channel_mlp.fc2': global_masks[i]['channel_mlp.fc2'] * mask_channel_mlp_fc2
        })

    return new_global_masks


def build_fully_pruned_mixer_model(model, global_masks):
    device = next(model.parameters()).device
    mixer_block_dims = []

    # Kezdeti bemeneti dimenziók (pl. CIFAR10: 32x32 patch -> 16)
    orig_num_patches = (32 // model.patch_size) ** 2
    orig_embedding_dim = model.embedding.out_features

    current_token_dim = orig_num_patches
    current_channel_dim = orig_embedding_dim

    for i, block in enumerate(model.mixer_blocks):
        # --- Token MLP pruning ---
        mask_token_fc1 = global_masks[i]['token_mlp.fc1']
        mask_token_fc2 = global_masks[i]['token_mlp.fc2']
        token_fc1_in = (mask_token_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        token_fc1_out = (mask_token_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        token_fc2_out = (mask_token_fc2.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        # --- Channel MLP pruning ---
        mask_channel_fc1 = global_masks[i]['channel_mlp.fc1']
        mask_channel_fc2 = global_masks[i]['channel_mlp.fc2']
        channel_fc1_in = (mask_channel_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        channel_fc1_out = (mask_channel_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        channel_fc2_out = (mask_channel_fc2.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        # Pruned MLP méretek
        token_dim = len(token_fc1_out)        # token_mlp.fc1 out
        token_out_dim = len(token_fc2_out)    # token_mlp.fc2 out
        token_in_dim = current_token_dim      # residual miatt!
        token_out_proj_dim = current_token_dim  # residual miatt!

        channel_dim = len(channel_fc1_out)        # channel_mlp.fc1 out
        channel_out_dim = len(channel_fc2_out)    # channel_mlp.fc2 out
        channel_in_dim = current_channel_dim      # residual miatt!
        channel_out_proj_dim = current_channel_dim  # residual miatt!

        # Block konfigurációk
        mixer_block_dims.append((
            token_in_dim,            # num_patches (block input token dim)
            channel_in_dim,          # dim (block input channel dim)
            token_dim, token_out_dim,
            channel_dim, channel_out_dim,
            token_in_dim, token_out_proj_dim,
            channel_in_dim, channel_out_proj_dim
        ))

        # NEM változtatjuk current_token_dim/current_channel_dim, residual shape végig fix!

    pruned_mixer = MLPMixer(
        patch_size=model.patch_size,
        in_channels=model.in_channels,
        embedding_dim=model.embedding.out_features,
        num_classes=model.classifier.out_features,
        mixer_block_dims=mixer_block_dims
    ).to(device)

    pruned_mixer.embedding.load_state_dict(model.embedding.state_dict())
    pruned_mixer.norm.load_state_dict(model.norm.state_dict())
    pruned_mixer.classifier.load_state_dict(model.classifier.state_dict())

    # Súlyok másolása prunolt FC-kre!
    for i, block in enumerate(pruned_mixer.mixer_blocks):
        orig_block = model.mixer_blocks[i]
        with torch.no_grad():
            # --- Token MLP ---
            mask_token_fc1 = global_masks[i]['token_mlp.fc1']
            mask_token_fc2 = global_masks[i]['token_mlp.fc2']
            token_fc1_in = (mask_token_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            token_fc1_out = (mask_token_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            token_fc2_out = (mask_token_fc2.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

            # FC1 (pruned in-dim ellenőrzés)
            orig_in_dim = orig_block.token_mlp.fc1.in_features
            pruned_in_dim = block.token_mlp.fc1.in_features
            if pruned_in_dim == len(token_fc1_in):
                block.token_mlp.fc1.weight.copy_(
                    orig_block.token_mlp.fc1.weight[token_fc1_out][:, token_fc1_in]
                )
            else:
                block.token_mlp.fc1.weight.copy_(
                    orig_block.token_mlp.fc1.weight[token_fc1_out]
                )
            block.token_mlp.fc1.bias.copy_(orig_block.token_mlp.fc1.bias[token_fc1_out])

            # FC2 (mindig out, in: az előző FC1 out-ja)
            block.token_mlp.fc2.weight.copy_(
                orig_block.token_mlp.fc2.weight[token_fc2_out][:, token_fc1_out]
            )
            block.token_mlp.fc2.bias.copy_(orig_block.token_mlp.fc2.bias[token_fc2_out])

            # --- Channel MLP ---
            mask_channel_fc1 = global_masks[i]['channel_mlp.fc1']
            mask_channel_fc2 = global_masks[i]['channel_mlp.fc2']
            channel_fc1_in = (mask_channel_fc1.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            channel_fc1_out = (mask_channel_fc1.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            channel_fc2_out = (mask_channel_fc2.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

            # FC1 (pruned in-dim ellenőrzés)
            orig_in_dim = orig_block.channel_mlp.fc1.in_features
            pruned_in_dim = block.channel_mlp.fc1.in_features
            if pruned_in_dim == len(channel_fc1_in):
                block.channel_mlp.fc1.weight.copy_(
                    orig_block.channel_mlp.fc1.weight[channel_fc1_out][:, channel_fc1_in]
                )
            else:
                block.channel_mlp.fc1.weight.copy_(
                    orig_block.channel_mlp.fc1.weight[channel_fc1_out]
                )
            block.channel_mlp.fc1.bias.copy_(orig_block.channel_mlp.fc1.bias[channel_fc1_out])

            # FC2 (mindig out, in: az előző FC1 out-ja)
            block.channel_mlp.fc2.weight.copy_(
                orig_block.channel_mlp.fc2.weight[channel_fc2_out][:, channel_fc1_out]
            )
            block.channel_mlp.fc2.bias.copy_(orig_block.channel_mlp.fc2.bias[channel_fc2_out])

            # Norm rétegek
            block.norm1.load_state_dict(orig_block.norm1.state_dict())
            block.norm2.load_state_dict(orig_block.norm2.state_dict())
            # Projekciós layereket nem kell másolni, random marad!

    print("Num of parameters in pruned model:", sum(p.numel() for p in pruned_mixer.parameters()))
    print("Num of parameters in original model:", sum(p.numel() for p in model.parameters()))
    return pruned_mixer



def lottery_ticket_mixer_cycle(prune_steps=7, relative_margin=0.15):
    # --- Data ---
    transform = transforms.ToTensor()
    train_loader = DataLoader(datasets.CIFAR10('.', train=True, download=True, transform=transform),
                            batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('.', train=False, download=True, transform=transform),
                            batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPMixer().to(device)
    print(model)
    initial_state = copy.deepcopy(model.state_dict())

    # Globális maszk inicializálása minden blokkra
    global_masks = []
    for block in model.mixer_blocks:
        m = {
            'token_mlp.fc1': torch.ones_like(block.token_mlp.fc1.weight),
            'token_mlp.fc1.bias': torch.ones_like(block.token_mlp.fc1.bias),
            
            'token_mlp.fc2': torch.ones_like(block.token_mlp.fc2.weight),

            'channel_mlp.fc1': torch.ones_like(block.channel_mlp.fc1.weight),
            'channel_mlp.fc1.bias': torch.ones_like(block.channel_mlp.fc1.bias),
            
            'channel_mlp.fc2': torch.ones_like(block.channel_mlp.fc2.weight),
        }
        global_masks.append(m)

    for step in range(prune_steps):
        print(f"\n=== MIXER PRUNING STEP {step + 1} ===")

        model = MLPMixer().to(device)
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.Adam(model.parameters())

        # Maszkolás alkalmazása a súlyokra
        with torch.no_grad():
            for i, block in enumerate(model.mixer_blocks):
                block.token_mlp.fc1.weight *= global_masks[i]['token_mlp.fc1'].to(device)
                block.token_mlp.fc1.bias *= global_masks[i]['token_mlp.fc1.bias'].to(device)
                
                block.token_mlp.fc2.weight *= global_masks[i]['token_mlp.fc2'].to(device)
                
                block.channel_mlp.fc1.weight *= global_masks[i]['channel_mlp.fc1'].to(device)
                block.channel_mlp.fc1.bias *= global_masks[i]['channel_mlp.fc1.bias'].to(device)
                
                block.channel_mlp.fc2.weight *= global_masks[i]['channel_mlp.fc2'].to(device)

        for epoch in range(3):
            train(model, optimizer, train_loader, device)
            acc, elapsed = test(model, test_loader, device)
            print(f"Epoch {epoch + 1}: accuracy={acc:.4f} - {elapsed:.4f} seconds")

        # Mentés az utolsó iterációban
        if step == prune_steps - 1:
            torch.save(model.state_dict(), "original_mixer_model.pt")
            print("Final pruned Mixer model saved as 'mixer_pruned_model.pt'")
            pruned_mixer = build_fully_pruned_mixer_model(model, global_masks)
            print(pruned_mixer)
            # --- Finomhangolás (fine-tune) pruned_mixer-en ---
            pruned_optimizer = torch.optim.Adam(pruned_mixer.parameters())
            for epoch in range(5):
                train(pruned_mixer, pruned_optimizer, train_loader, device)
                acc, elapsed = test(pruned_mixer, test_loader, device)
                print(f"[Fine-tune] Epoch {epoch + 1}: accuracy={acc:.4f} - {elapsed:.4f} seconds")
            
            # Mérés a végleges pruned modellen
            acc, elapsed = test(pruned_mixer, test_loader, device)
            print(f"[Eval] pruned modell - Inference time on test set: {elapsed:.4f} seconds - accuracy={acc:.4f}")
            torch.save(pruned_mixer.state_dict(), "pruned_mixer_model.pt")
            dummy_input = torch.randn(1, 3, 32, 32).to(device)
            # ONNX exportálás
            torch.onnx.export(
                pruned_mixer,
                dummy_input,
                "pruned_model.onnx",
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=20
            )
            break

        # Új maszkolás
        global_masks = prune_mixer_model(model, global_masks, relative_margin=relative_margin)
        
        # Log pruning statistics
        for i, block in enumerate(model.mixer_blocks):
            for name, weight in [
                ('token_mlp.fc1', block.token_mlp.fc1.weight),
                ('token_mlp.fc2', block.token_mlp.fc2.weight),
                ('channel_mlp.fc1', block.channel_mlp.fc1.weight),
                ('channel_mlp.fc2', block.channel_mlp.fc2.weight)
            ]:
                num_zero = torch.sum(weight == 0).item()
                total = weight.numel()
                sparsity = 100.0 * num_zero / total
                print(f"[Pruning Info] Block {i} {name} nullázott súlyok: {num_zero}/{total} ({sparsity:.2f}%)")
    
# === Futtatás ===
lottery_ticket_mixer_cycle()