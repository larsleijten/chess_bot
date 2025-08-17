import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple


class ChunkedChessDataset(Dataset):
    """
    A PyTorch Dataset for loading chess data stored in chunked .pt files.

    This class is designed to handle datasets that are too large to fit in
    memory. It works by discovering all chunk files in a directory and only

    loading one chunk into memory at a time as needed.
    """

    def __init__(self, directory: str):
        """
        Initializes the dataset.

        Args:
            directory: The path to the directory containing the chunk files.
        """
        self.directory = Path(directory)
        self.chunk_files = sorted(self.directory.glob("chunk_*.pt"))

        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found in directory: {directory}")

        self.chunk_sizes = [100000 for f in self.chunk_files]
        cumulative_sizes_tensor = torch.cumsum(torch.tensor(self.chunk_sizes), 0)
        self.cumulative_sizes = torch.cat((torch.tensor([0]), cumulative_sizes_tensor))

        self.total_size = self.cumulative_sizes[-1].item()

        # Cache for the currently loaded chunk to avoid repeated file I/O
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cached_chunk_index: int = -1

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.total_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx: The global index of the sample to retrieve.

        Returns:
            A tuple containing the board tensor and the move tensor.
        """
        if not (0 <= idx < self.total_size):
            raise IndexError("Index out of range")

        chunk_index = torch.searchsorted(self.cumulative_sizes, idx + 1).item() - 1

        local_index = idx - self.cumulative_sizes[chunk_index].item()

        if chunk_index != self._cached_chunk_index:
            self._cache = torch.load(self.chunk_files[chunk_index])
            self._cached_chunk_index = chunk_index

        board = self._cache["boards"][local_index]
        move_one_hot = self._cache["moves"][local_index]
        move = move_one_hot.argmax()

        return board, move
