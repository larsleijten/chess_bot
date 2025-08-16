import chess
import sys
import random
from pathlib import Path

def main():
    """
    Main function to run the UCI chess bot.
    This bot communicates with a GUI using the UCI protocol.
    It makes random legal moves.
    """
    # Initialize the board
    board = chess.Board()
    log_folder = Path("/home/larsleijten/repositories/chess_bot/logs")
    # A simple log for debugging
    with open(log_folder / "bot_log.txt", "w") as log_file:
        log_file.write("Bot started\n")

    # Main loop to process UCI commands
    while True:
        # Read a command from the GUI
        line = sys.stdin.readline().strip()
        
        # Log the received command
        with open(log_folder / "bot_log.txt", "a") as log_file:
            log_file.write(f"Received: {line}\n")
        
        # Exit if we receive an empty line (e.g., GUI closed)
        if not line:
            break

        parts = line.split()
        command = parts[0]

        if command == "uci":
            # Respond to the 'uci' command with engine identity and options
            # This is the first command sent by the GUI
            sys.stdout.write("id name MyRandomBot\n")
            sys.stdout.write("id author YourName\n")
            # Add any options your bot supports here, if any
            sys.stdout.write("uciok\n")
            sys.stdout.flush()

        elif command == "isready":
            # Respond to 'isready' to confirm the bot is ready to receive more commands
            sys.stdout.write("readyok\n")
            sys.stdout.flush()

        elif command == "ucinewgame":
            # Reset the board for a new game
            board.reset()

        elif command == "position":
            # Set up the board position based on the GUI's command
            # Example: position startpos moves e2e4 e7e5
            if "startpos" in parts:
                board.reset()
                moves_index = parts.index("moves") if "moves" in parts else -1
            elif "fen" in parts:
                fen_parts = parts[parts.index("fen")+1:]
                fen_str = " ".join(fen_parts)
                # Find where the moves part starts, if it exists
                if "moves" in fen_parts:
                    fen_str = " ".join(fen_parts[:fen_parts.index("moves")])
                board.set_fen(fen_str)
                moves_index = parts.index("moves") if "moves" in parts else -1
            else:
                moves_index = -1

            if moves_index != -1:
                moves_list = parts[moves_index + 1:]
                for move_uci in moves_list:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                    except ValueError:
                        # Invalid move format, log and ignore
                        with open(log_folder / "bot_log.txt", "a") as log_file:
                            log_file.write(f"Invalid move UCI: {move_uci}\n")


        elif command == "go":
            # The GUI wants the bot to think and make a move
            
            # --- THIS IS WHERE YOU ADD YOUR BOT'S LOGIC ---
            # For now, we just pick a random legal move.
            
            if not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    # Choose a random move
                    best_move = random.choice(legal_moves)
                    
                    # Send the best move back to the GUI
                    sys.stdout.write(f"bestmove {best_move.uci()}\n")
                    sys.stdout.flush()
            # ----------------------------------------------

        elif command == "quit":
            # The GUI is shutting down the engine
            break

if __name__ == "__main__":
    main()

