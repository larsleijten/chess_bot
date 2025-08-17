import pygame
import chess
import random
import torch
from typing import Optional, Tuple
from chess_bot.bot.cnn_chessbot import ChessUNET, ChessPolicyNet
import chess_bot.utils.tensor_uci_interface as interface


# --- Pygame Setup ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BOARD_SIZE = 512  # The size of the board image
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_OFFSET_X = (SCREEN_WIDTH - BOARD_SIZE) // 2
BOARD_OFFSET_Y = (SCREEN_HEIGHT - BOARD_SIZE) // 2

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT_COLOR = (255, 255, 51, 150)  # Yellow with transparency


def get_square_from_pos(pos: Tuple[int, int]) -> Optional[int]:
    """Converts a mouse position to a chess square index."""
    x, y = pos
    if not (
        BOARD_OFFSET_X <= x < BOARD_OFFSET_X + BOARD_SIZE
        and BOARD_OFFSET_Y <= y < BOARD_OFFSET_Y + BOARD_SIZE
    ):
        return None

    file = (x - BOARD_OFFSET_X) // SQUARE_SIZE
    rank = 7 - ((y - BOARD_OFFSET_Y) // SQUARE_SIZE)
    return chess.square(file, rank)


def draw_board(screen: pygame.Surface):
    """Draws the chessboard squares."""
    for r in range(8):
        for f in range(8):
            color = WHITE if (r + f) % 2 == 0 else BLACK
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


def highlight_square(screen: pygame.Surface, square: int):
    """Highlights a single square."""
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    highlight_surface.fill(HIGHLIGHT_COLOR)
    screen.blit(
        highlight_surface,
        (
            BOARD_OFFSET_X + file * SQUARE_SIZE,
            BOARD_OFFSET_Y + (7 - rank) * SQUARE_SIZE,
        ),
    )


def get_computer_move(board: chess.Board, bot: ChessPolicyNet) -> Optional[chess.Move]:
    """
    Gets the computer's move.

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
    pygame.display.set_caption("Chess")

    # Use a font that supports Unicode chess pieces
    font = pygame.font.Font(
        "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/dejavu-sans/DejaVuSans.ttf",
        80,
    )

    board = chess.Board()
    bot = ChessUNET()
    bot.load_state_dict(
        torch.load(
            "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/train/unet_black.pth"
        )
    )
    selected_square: Optional[int] = None
    player_turn = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if player_turn and event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                square = get_square_from_pos(pos)

                if selected_square is not None:
                    # A piece is already selected, try to make a move
                    move = chess.Move(selected_square, square)
                    move = interface.check_promotion_move(board, move)
                    if move in board.legal_moves:
                        board.push(move)
                        player_turn = False  # Switch to computer's turn
                    selected_square = None  # Deselect
                elif square is not None and board.piece_at(square) is not None:
                    # Select a piece
                    if board.piece_at(square).color == board.turn:
                        selected_square = square

        # Drawing
        draw_board(screen)
        if selected_square is not None:
            highlight_square(screen, selected_square)
        draw_pieces(screen, board, font)
        pygame.display.flip()

        # Computer's Turn
        if not player_turn and not board.is_game_over():
            pygame.time.wait(500)  # Give a small delay
            move = get_computer_move(board, bot)
            if move:
                board.push(move)
            player_turn = True

        # Check for game over
        if board.is_game_over():
            print("Game Over:", board.result())
            pygame.time.wait(3000)  # Show final board for 3 seconds
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
