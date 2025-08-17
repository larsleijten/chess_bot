import chess.pgn
import zstandard
import io
import torch
import argparse
import chess_bot.utils.tensor_uci_interface as interface
from typing import Generator
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pgn_zst_file", type=str, required=True, help="Location of PGN ZST file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for tensor chunks",
    )
    return parser.parse_args()


def pgn_stream_generator(file_path: str) -> Generator[chess.pgn.Game, None, None]:
    """
    Creates a generator that yields chess games from a .pgn.zst file.
    """
    with open(file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                yield game


def process_and_save_chunks(pgn_path: str, output_dir: str, chunk_size: int = 100000):
    """
    Processes a large PGN file and saves the data in chunks.

    Args:
        pgn_path: Path to the .pgn.zst file.
        output_dir: Directory to save the tensor chunks.
        chunk_size: The number of (board, move) pairs to save in each chunk.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    board_chunks_white = []
    board_chunks_black = []
    move_chunks_white = []
    move_chunks_black = []
    chunk_num_white = 0
    chunk_num_black = 0
    total_positions = 0

    game_generator = pgn_stream_generator(pgn_path)

    for game in game_generator:
        board = game.board()
        last_move_white = True
        for move in game.mainline_moves():
            if last_move_white:
                player = "white"
            else:
                player = "black"

            board_tensor = interface.board_to_tensor(board)
            move.promotion = None
            move_tensor = interface.moves_uci_to_tensor(move)

            if player == "white":
                board_chunks_white.append(board_tensor)
                move_chunks_white.append(move_tensor)
            else:
                board_chunks_black.append(board_tensor)
                move_chunks_black.append(move_tensor)

            board.push(move)
            total_positions += 1

            if len(board_chunks_white) >= chunk_size:
                boards_tensor = torch.stack(board_chunks_white)
                moves_tensor = torch.stack(move_chunks_white)
                chunk_path = Path(output_dir) / f"white/chunk_{chunk_num_white}.pt"
                torch.save({"boards": boards_tensor, "moves": moves_tensor}, chunk_path)

                board_chunks_white = []
                move_chunks_white = []
                chunk_num_white += 1

            if len(board_chunks_black) >= chunk_size:
                boards_tensor = torch.stack(board_chunks_black)
                moves_tensor = torch.stack(move_chunks_black)
                chunk_path = Path(output_dir) / f"black/chunk_{chunk_num_black}.pt"
                torch.save({"boards": boards_tensor, "moves": moves_tensor}, chunk_path)

                board_chunks_black = []
                move_chunks_black = []
                chunk_num_black += 1

            last_move_white = not last_move_white

    if board_chunks_white:
        boards_tensor = torch.stack(board_chunks_white)
        moves_tensor = torch.stack(move_chunks_white)
        chunk_path = Path(output_dir) / f"white/chunk_{chunk_num_white}.pt"
        torch.save({"boards": boards_tensor, "moves": moves_tensor}, chunk_path)

    if board_chunks_black:
        boards_tensor = torch.stack(board_chunks_black)
        moves_tensor = torch.stack(move_chunks_black)
        chunk_path = Path(output_dir) / f"black/chunk_{chunk_num_black}.pt"
        torch.save({"boards": boards_tensor, "moves": moves_tensor}, chunk_path)

    print(f"\nProcessing complete. Total positions processed: {total_positions}")


if __name__ == "__main__":
    args = parse_args()
    process_and_save_chunks(args.pgn_zst_file, args.output_dir)
