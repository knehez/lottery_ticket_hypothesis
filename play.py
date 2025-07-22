
import tkinter as tk
import chess
import chess.svg
import torch
from PIL import Image, ImageTk
import io
from MLP_chess import MixerPolicyValueHead, mixer, fen_to_tensor

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP Mixer Chess GUI")
        self.board = chess.Board()
        self.canvas = tk.Canvas(root, width=520, height=520)
        self.canvas.pack()

        self.model = MixerPolicyValueHead(mixer)
        self.model.load_state_dict(torch.load("mixer_policyvalue_iter5.pt"))
        self.model.eval()

        self.selected_square = None
        self.draw_board()

        self.canvas.bind("<Button-1>", self.on_click)

    def draw_board(self):
        svg_data = chess.svg.board(self.board, lastmove=None, size=520)
        img = Image.open(io.BytesIO(svg_data.encode("utf-8")))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def on_click(self, event):
        file = event.x // 65
        rank = 7 - (event.y // 65)
        square = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                self.root.after(300, self.engine_move)
            else:
                self.selected_square = None

    def engine_move(self):
        move = self.predict_best_move()
        if move:
            self.board.push(move)
            self.draw_board()

    def predict_best_move(self):
        with torch.no_grad():
            x = torch.tensor(fen_to_tensor(self.board.fen()), dtype=torch.float32).unsqueeze(0)
            policy_logits, _ = self.model(x)
            probs = torch.softmax(policy_logits.view(1, 64 * 64), dim=1).view(64, 64)

            legal_moves = list(self.board.legal_moves)
            scores = {m: probs[m.from_square][m.to_square].item() for m in legal_moves}
            return max(scores, key=scores.get) if scores else None

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()
