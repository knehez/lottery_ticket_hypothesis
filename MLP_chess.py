import chess
import numpy as np
import torch.nn as nn
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from trainer import online_training_loop
from trainer import fen_to_tensor
# --- MLP-Mixer model ---
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x = self.fc2(self.act(self.fc1(x)))
        return x.view(*orig_shape)

class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, token_dim, channel_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = MLP(num_patches, token_dim, num_patches)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = MLP(dim, channel_dim, dim)

    def forward(self, x):
        # x shape: [B, N, C]
        y = self.norm1(x).transpose(1, 2)        # [B, C, N]
        B, C, N = y.shape
        y = y.reshape(B * C, N)                  # [B*C, N]
        y = self.token_mlp(y)                    # [B*C, N]
        y = y.reshape(B, C, N).transpose(1, 2)   # [B, N, C]
        x = x + y
        x = x + self.channel_mlp(self.norm2(x))  # [B, N, C]
        return x


class MLPMixer(nn.Module):
    def __init__(self, image_size=28, patch_size=4, dim=64, depth=4, num_classes=10, token_dim=32, channel_dim=128, in_channels=1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.embedding = nn.Linear(patch_dim, dim)
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(num_patches, dim, token_dim, channel_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dim = dim
        self.in_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # Patch-elés: minden patch tartalmazza az összes csatornát
        x = x.unfold(2, p, p).unfold(3, p, p)  # [B, C, H//p, W//p, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, H//p, W//p, C, p, p]
        x = x.view(B, self.num_patches, C * p * p)    # [B, num_patches, patch_dim]
        x = self.embedding(x)
        x = self.mixer_blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

mixer = MLPMixer(
    image_size=8,
    patch_size=1,
    dim=64,
    depth=4,
    num_classes=1,       # csak placeholder, mert policy+value head külön van
    token_dim=32,
    channel_dim=128,
    in_channels=20  # Állítsd be a bemeneti csatornák számát a fen_to_tensor alapján (20)
)

class ChessDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states  # list of [C, 8, 8] numpy arrays
        self.policies = policies  # list of [64, 64] numpy arrays
        self.values = values  # list of scalar values (-1, 0, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        x = torch.tensor(self.states[idx], dtype=torch.float32)
        pi = torch.tensor(self.policies[idx], dtype=torch.float32)
        z = torch.tensor(self.values[idx], dtype=torch.float32)
        return x, pi, z

class MixerPolicyValueHead(nn.Module):
    def __init__(self, mlp_mixer: nn.Module):
        super().__init__()
        self.backbone = mlp_mixer
        self.policy_head = nn.Linear(mlp_mixer.classifier.out_features, 64 * 64)
        self.value_head = nn.Sequential(
            nn.Linear(mlp_mixer.classifier.out_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.backbone(x)
        policy_logits = self.policy_head(features).view(-1, 64, 64)
        value = self.value_head(features).squeeze(1)
        return policy_logits, value

def train_step(model, batch, optimizer):
    x, pi_star, z = batch
    optimizer.zero_grad()
    policy_logits, value = model(x)
    policy_log_probs = F.log_softmax(policy_logits.view(x.size(0), -1), dim=1)
    target_probs = pi_star.view(x.size(0), -1)
    loss_policy = F.kl_div(policy_log_probs, target_probs, reduction='batchmean')
    loss_value = F.mse_loss(value, z)
    loss = loss_policy + loss_value
    loss.backward()
    optimizer.step()
    return loss.item(), loss_policy.item(), loss_value.item()

class MCTSNode:
    def __init__(self, board: chess.Board):
        self.board = board
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.prior = {}

    def value(self):
        return 0 if self.visits == 0 else self.value_sum / self.visits

def ucb_score(parent_visits, child_visits, q_value, prior, c_puct=1.0):
    if child_visits == 0:
        return float('inf')
    return q_value + c_puct * prior * np.sqrt(parent_visits) / (1 + child_visits)

def select_child(node):
    best_score = -float('inf')
    best_move = None
    best_child = None

    for move, child in node.children.items():
        score = ucb_score(
            parent_visits=node.visits,
            child_visits=child.visits,
            q_value=child.value(),
            prior=node.prior.get(move, 1.0)
        )
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child

def expand_node(node):
    legal_moves = list(node.board.legal_moves)
    prior_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
    for move in legal_moves:
        new_board = node.board.copy()
        new_board.push(move)
        node.children[move] = MCTSNode(new_board)
        node.prior[move] = prior_prob

def simulate_game(board: chess.Board):
    sim_board = board.copy()
    for _ in range(20):
        if sim_board.is_game_over():
            break
        move = random.choice(list(sim_board.legal_moves))
        sim_board.push(move)

    if sim_board.is_checkmate():
        return -1
    elif sim_board.is_stalemate() or sim_board.is_insufficient_material():
        return 0
    return 0

def backpropagate(path, value):
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value

def run_mcts(board: chess.Board, simulations=100):
    root = MCTSNode(board)

    for _ in range(simulations):
        node = root
        path = [node]

        while node.children:
            move, node = select_child(node)
            path.append(node)

        if not node.board.is_game_over():
            expand_node(node)
            if node.children:
                move = random.choice(list(node.children.keys()))
                node = node.children[move]
                path.append(node)

        value = simulate_game(node.board)
        backpropagate(path, value)

    pi_star = np.zeros((64, 64), dtype=np.float32)
    for move, child in root.children.items():
        pi_star[move.from_square, move.to_square] = child.visits

    total = np.sum(pi_star)
    if total > 0:
        pi_star /= total
    return pi_star

def self_play_game(model, max_moves=60, simulations=100):
    board = chess.Board()
    positions = []
    policies = []
    result = None

    for _ in range(max_moves):
        if board.is_game_over():
            break

        pi_star = run_mcts_with_model(board, model, simulations)
        positions.append(board.fen())
        policies.append(pi_star)

        legal_moves = list(board.legal_moves)
        move_probs = [(move, pi_star[move.from_square][move.to_square]) for move in legal_moves]
        total_prob = sum(prob for _, prob in move_probs)
        move_probs = [(m, p / total_prob) for m, p in move_probs if p > 0]

        if not move_probs:
            move = random.choice(legal_moves)
        else:
            moves, probs = zip(*move_probs)
            move = random.choices(moves, probs)[0]

        board.push(move)

    if board.result() == '1-0':
        result = 1
    elif board.result() == '0-1':
        result = -1
    else:
        result = 0

    return positions, policies, result

if __name__ == "__main__":
    from mcts_with_model import run_mcts_with_model
    model = MixerPolicyValueHead(mixer)
    online_training_loop(
        model=model,
        train_step=train_step,
        self_play_game=self_play_game,
        fen_to_tensor=fen_to_tensor,
        ChessDataset=ChessDataset,
        iterations=5,
        games_per_iter=10
    )
