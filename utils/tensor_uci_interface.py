import chess
import torch

def board_uci_to_tensor(uci_board: chess.Board) -> torch.tensor:
    # TODO
    tensor_board = uci_board

def board_tensor_to_uci(tensor_board: torch.tensor) -> chess.Board:
    # TODO
    uci_board = tensor_board 

def moves_uci_to_tensor(uci_move: chess.Move) -> torch.tensor:
    tensor_move = uci_move