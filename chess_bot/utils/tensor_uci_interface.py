import chess
import torch


def board_to_tensor(board: chess.Board) -> torch.tensor:
    """
    Converts a chess.Board object to a (12, 8, 8) tensor representation.

    Args:
        board: The chess.Board object.

    Returns:
        A PyTorch tensor representing the board state.
    """

    piece_to_channel = {
        "K": 0,
        "Q": 1,
        "R": 2,
        "B": 3,
        "N": 4,
        "P": 5,
        "k": 6,
        "q": 7,
        "r": 8,
        "b": 9,
        "n": 10,
        "p": 11,
    }

    tensor_board = torch.zeros((12, 8, 8))  # C, Rank, File

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[piece.symbol()]

            rank = chess.square_rank(square)
            file = chess.square_file(square)

            tensor_board[channel, rank, file] = 1

    return tensor_board


def moves_uci_to_tensor(uci_move: chess.Move) -> torch.tensor:
    tensor_move = uci_move


def moves_tensor_to_uci(tensor_move: torch.tensor) -> chess.Move:
    uci_move = tensor_move


def replace_digits_with_zeros(input_string: str) -> str:
    """
    Replaces any single digits in a string with that many '0's.

    Args:
        input_string: The string to process.

    Returns:
        The new string with digits replaced by zeros.
    """
    output_string = ""
    for char in input_string:
        if char.isdigit():
            num_zeros = int(char)
            output_string += "0" * num_zeros
        else:
            output_string += char
    return output_string
