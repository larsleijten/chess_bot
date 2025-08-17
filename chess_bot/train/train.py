import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict
from chess_bot.bot.cnn_chessbot import ChessPolicyNet, ChessUNET
from chess_bot.train.datasets import ChunkedChessDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a chess bot using chunked datasets."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    return parser.parse_args()


def train_model(
    args, model: nn.Module, epochs: int, batch_size: int, learning_rate: float = 0.001
):
    """
    A simple training loop for the chess bot.

    Args:
        data_dir: Directory containing the chunked dataset files.
        model: The neural network model to train.
        epochs: The number of times to iterate over the entire dataset.
        batch_size: The number of samples per batch.
        learning_rate: The learning rate for the optimizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    dataset = ChunkedChessDataset(directory=args.data_dir)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (boards, moves) in enumerate(data_loader):
            boards = boards.to(device)
            moves = moves.to(device)
            optimizer.zero_grad()
            outputs = model(boards)

            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(data_loader)}], Loss: {running_loss / 100:.4f}"
                )
                running_loss = 0.0

    torch.save(model.state_dict(), args.checkpoint_path)
    print("Finished Training")


if __name__ == "__main__":
    args = parse_args()

    bot = ChessUNET()
    train_model(args=args, model=bot, epochs=args.epochs, batch_size=args.batch_size)
