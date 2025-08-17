import pygame
import chess
import time
import torch
from typing import Optional, Tuple
from chess_bot.bot.cnn_chessbot import ChessPolicyNet, ChessUNET
import chess_bot.utils.tensor_uci_interface as interface

# --- Configurable Parameters ---
MOVE_DELAY_MS = 0  # Set the delay between moves in milliseconds

# --- Pygame Setup ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BOARD_SIZE = 512
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_OFFSET_X = (SCREEN_WIDTH - BOARD_SIZE) // 2
BOARD_OFFSET_Y = (SCREEN_HEIGHT - BOARD_SIZE) // 2

# Colors
WHITE_SQUARE = (240, 217, 181)
BLACK_SQUARE = (181, 136, 99)


def draw_board(screen: pygame.Surface):
    """Draws the chessboard squares."""
    for r in range(8):
        for f in range(8):
            color = WHITE_SQUARE if (r + f) % 2 == 0 else BLACK_SQUARE
            rect = pygame.Rect(
                BOARD_OFFSET_X + f * SQUARE_SIZE,
                BOARD_OFFSET_Y + r * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE,
            )
            pygame.draw.rect(screen, color, rect)


def draw_pieces(screen: pygame.Surface, board: chess.Board, font: pygame.font.Font):
    """Draws the pieces on the board using Unicode characters."""
    piece_map = {
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece_map[piece.symbol()]
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            text_surface = font.render(symbol, True, (0, 0, 0))
            text_rect = text_surface.get_rect(
                center=(
                    BOARD_OFFSET_X + file * SQUARE_SIZE + SQUARE_SIZE // 2,
                    BOARD_OFFSET_Y + (7 - rank) * SQUARE_SIZE + SQUARE_SIZE // 2,
                )
            )
            screen.blit(text_surface, text_rect)


def get_computer_move(board: chess.Board, bot: ChessPolicyNet) -> Optional[chess.Move]:
    """
    Gets the computer's move by calling the provided bot instance.
    """
    tensor_board = interface.board_to_tensor(board).unsqueeze(0)
    logits = bot(tensor_board)
    legal_move_mask = interface.legal_move_mask(board.legal_moves)
    tensor_move = interface.pick_tensor_move(logits, legal_move_mask).squeeze()
    uci_move = interface.moves_tensor_to_uci(tensor_move)
    uci_move = interface.check_promotion_move(board, uci_move)
    return uci_move


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Bot vs. Bot Exhibition")

    # Use a font that supports Unicode chess pieces
    try:
        font = pygame.font.Font(
            "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/dejavu-sans/DejaVuSans.ttf",
            60,
        )
    except FileNotFoundError:
        print(
            "Custom font not found. Using default font (chess pieces may not render)."
        )
        font = pygame.font.Font(None, 80)

    board = chess.Board()

    # Initialize two separate bot instances
    bot_white = ChessUNET()

    bot_black = ChessUNET()
    bot_black.load_state_dict(
        torch.load(
            "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/train/unet_black.pth"
        )
    )

    running = True
    while running and not board.is_game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Determine which bot's turn it is
        current_bot = bot_white if board.turn == chess.WHITE else bot_black

        # Get the move from the current bot
        move = get_computer_move(board, current_bot)

        if move:
            board.push(move)
        else:
            # No legal moves, game must be over
            break

        # Drawing
        draw_board(screen)
        draw_pieces(screen, board, font)
        pygame.display.flip()

        # Wait for the specified delay
        pygame.time.wait(MOVE_DELAY_MS)

    # Game is over
    if board.is_game_over():
        print("--- Game Over ---")
        pygame.time.wait(5000)
        print("Result:", board.result())
        # Keep the final board position on screen
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

    pygame.quit()


if __name__ == "__main__":
    main()
