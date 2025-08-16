import chess.pgn
import zstandard
import io
import argparse


def process_pgn_zst(file_path: str):
    """
    Reads and processes a Zstandard compressed PGN file.

    This function decompresses the file and iterates through each game,
    printing the event and the final board position as an example.

    Args:
        file_path: The path to the .pgn.zst file.
    """
    game_count = 0
    with open(file_path, "rb") as compressed_file:
        # Create a decompressor
        dctx = zstandard.ZstdDecompressor()

        # Create a stream reader to handle decompression on the fly
        with dctx.stream_reader(compressed_file) as reader:
            # Wrap the binary stream in a text stream
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            # Use python-chess to read games from the stream
            while True:
                # The read_game function will process one game at a time
                game = chess.pgn.read_game(text_stream)

                # If no more games are found, break the loop
                if game is None:
                    break

                # --- This is where you process each game ---
                # For example, you can iterate through moves to create training data
                board = game.board()
                for move in game.mainline_moves():
                    # Here you'd get your (board_tensor, move_tensor) pair
                    board.push(move)

                game_count += 1
                if game_count % 1000 == 0:
                    print(f"Processed {game_count} games...")

    print(f"\nFinished processing. Total games found: {game_count}")


# --- How to use it ---
if __name__ == "__main__":
    # Replace with the actual path to your downloaded file
    lichess_db_path = "path/to/your/lichess_db_standard_rated_2023-01.pgn.zst"
    try:
        process_pgn_zst(lichess_db_path)
    except FileNotFoundError:
        print(f"Error: File not found at {lichess_db_path}")
        print(
            "Please update the 'lichess_db_path' variable with the correct file path."
        )
