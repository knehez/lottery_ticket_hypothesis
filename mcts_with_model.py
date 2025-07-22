import chess
import numpy as np
import torch
from trainer import fen_to_tensor

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

def expand_node_with_model(node, model):
    x = torch.tensor(fen_to_tensor(node.board.fen()), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value = model(x)
        probs = torch.softmax(policy_logits.view(1, 64 * 64), dim=1).view(64, 64)
        value = value.item()

    legal_moves = list(node.board.legal_moves)
    total_p = 0.0
    for move in legal_moves:
        p = probs[move.from_square][move.to_square].item()
        node.prior[move] = p
        total_p += p
        new_board = node.board.copy()
        new_board.push(move)
        node.children[move] = MCTSNode(new_board)

    # Normalize priors
    if total_p > 0:
        for move in legal_moves:
            node.prior[move] /= total_p

    return value

def backpropagate(path, value):
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value  # alternate perspective

def run_mcts_with_model(board: chess.Board, model, simulations=100):
    from MLP_chess import MixerPolicyValueHead, mixer
    root = MCTSNode(board)

    for _ in range(simulations):
        node = root
        path = [node]

        while node.children:
            move, node = select_child(node)
            path.append(node)

        if not node.board.is_game_over():
            value = expand_node_with_model(node, model)
        else:
            result = node.board.result()
            if result == '1-0':
                value = 1
            elif result == '0-1':
                value = -1
            else:
                value = 0

        backpropagate(path, value)

    pi_star = np.zeros((64, 64), dtype=np.float32)
    for move, child in root.children.items():
        pi_star[move.from_square, move.to_square] = child.visits

    total = np.sum(pi_star)
    if total > 0:
        pi_star /= total
    return pi_star

# Optional test
if __name__ == "__main__":
    model = MixerPolicyValueHead(mixer)
    model.load_state_dict(torch.load("mixer_policyvalue_iter5.pt"))
    model.eval()

    board = chess.Board()
    pi = run_mcts_with_model(board, model, simulations=50)

    for move in board.legal_moves:
        print(f"{move.uci()}: {pi[move.from_square, move.to_square]:.4f}")
