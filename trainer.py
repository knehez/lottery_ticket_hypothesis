import chess
import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_self_play_data(model, n_games=25, simulations=100, max_moves=60, self_play_game=None, fen_to_tensor=None):
    all_states, all_policies, all_values = [], [], []
    for _ in range(n_games):
        positions, policies, result = self_play_game(model, simulations=simulations, max_moves=max_moves)
        states = [fen_to_tensor(fen) for fen in positions]
        values = [result] * len(states)
        all_states.extend(deepcopy(states))
        all_policies.extend(deepcopy(policies))
        all_values.extend(deepcopy(values))
    return all_states, all_policies, all_values

def save_self_play_data(filename, states, policies, values):
    torch.save({
        'states': states,
        'policies': policies,
        'values': values
    }, filename)

def load_self_play_data(filename, ChessDataset):
    data = torch.load(filename)
    return ChessDataset(data['states'], data['policies'], data['values'])

def evaluate_position(model, fen, fen_to_tensor):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(device)  # [1, C, 8, 8]
        model.to(device)
        policy_logits, value = model(x)
        probs = torch.softmax(policy_logits.view(64 * 64), dim=1).reshape(64, 64)
        return probs.cpu().numpy(), value.item()

def train_loop(model, dataset, train_step, epochs=5, batch_size=32, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history, policy_loss_history, value_loss_history = [], [], []

    model.to(device)
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            # batch elemeket GPU-ra viszünk
            batch = tuple(b.to(device) if hasattr(b, 'to') else b for b in batch)
            loss, lp, lv = train_step(model, batch, optimizer)
            loss_history.append(loss)
            policy_loss_history.append(lp)
            value_loss_history.append(lv)
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss:.4f}, Policy: {lp:.4f}, Value: {lv:.4f}")

def online_training_loop(model, train_step, self_play_game, fen_to_tensor, ChessDataset, iterations=5, games_per_iter=10):
    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===")
        states, policies, values = generate_self_play_data(
            model,
            n_games=games_per_iter,
            self_play_game=self_play_game,
            fen_to_tensor=fen_to_tensor
        )
        #save_self_play_data(f'selfplay_iter_{i+1}.pt', states, policies, values)
        dataset = ChessDataset(states, policies, values)
        train_loop(model, dataset, train_step, epochs=1, batch_size=16)
        torch.save(model.state_dict(), f"mixer_policyvalue_iter{i+1}.pt")

def fen_to_tensor(fen):
    """Egyszerű bitboard + metaadat encoder (12 bábu, 1 aktív fél, 4 sáncolás, 1 en passant)."""
    board = chess.Board(fen)
    planes = np.zeros((20, 8, 8), dtype=np.float32)

    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_map[piece.symbol()]
            row = 7 - square // 8
            col = square % 8
            planes[idx, row, col] = 1.0

    # Aktív fél (1 ha világos jön)
    planes[12] = (np.ones((8, 8), dtype=np.float32)
                  if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32))

    # Sáncolási jogok
    if board.has_kingside_castling_rights(chess.WHITE): planes[13] = 1
    if board.has_queenside_castling_rights(chess.WHITE): planes[14] = 1
    if board.has_kingside_castling_rights(chess.BLACK): planes[15] = 1
    if board.has_queenside_castling_rights(chess.BLACK): planes[16] = 1

    # En passant mező
    if board.ep_square:
        row = 7 - board.ep_square // 8
        col = board.ep_square % 8
        planes[17, row, col] = 1.0

    # Lépésszámláló (fél lépések mod 100)
    planes[18] = board.halfmove_clock / 100.0
    planes[19] = board.fullmove_number / 100.0

    return planes