import chess
import torch
import torch.nn.functional as F


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
    move_index = get_move_index(uci_move)
    tensor_move = torch.zeros(4096, dtype=torch.float32)
    tensor_move[move_index] = 1.0
    return tensor_move


def moves_tensor_to_uci(tensor_move: torch.tensor) -> chess.Move:
    move_index = int(torch.argmax(tensor_move).item())

    from_square_index = move_index // 64
    to_square_index = move_index % 64
    uci_move = chess.Move(from_square_index, to_square_index)

    return uci_move


def legal_move_mask(legal_moves: chess.Board.legal_moves) -> torch.tensor:
    move_mask = torch.zeros(4096, dtype=torch.float32)
    for move in legal_moves:
        move_index = get_move_index(move)
        move_mask[move_index] = 1.0

    return move_mask


def check_promotion_move(board: chess.Board, move: chess.Move) -> chess.Move:
    piece = board.piece_at(move.from_square)
    if piece.piece_type == chess.PAWN:
        if (
            chess.square_rank(move.to_square) == 0
            or chess.square_rank(move.to_square) == 7
        ):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

    return move


def pick_tensor_move(
    logits: torch.tensor, legal_move_mask: torch.tensor
) -> torch.tensor:
    if not legal_move_mask.any():
        return None  # No legal moves available

    masked_logits = logits.masked_fill(legal_move_mask == 0, -1e9)
    move_probs = F.softmax(masked_logits, dim=-1)
    move_index = torch.multinomial(move_probs, num_samples=1)
    tensor_move = F.one_hot(move_index, logits.squeeze().shape[-1])
    return tensor_move


def get_move_index(uci_move: chess.Move) -> int:
    return uci_move.from_square * 64 + uci_move.to_square


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
