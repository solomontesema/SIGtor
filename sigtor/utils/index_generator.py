import numpy as np


class SourceIndexGenerator:
    def __init__(self, total_indices, shuffle=True):
        """
        Initializes the SourceIndexGenerator.

        Args:
            total_indices (int): The total number of indices available.
            shuffle (bool): Whether to shuffle the indices.
        """
        self.total_indices = total_indices
        self.shuffle = shuffle
        self.indices = list(range(total_indices))
        if shuffle:
            np.random.shuffle(self.indices)
        self.current = 0

    def __iter__(self):
        """
        Returns the iterator object itself.

        Returns:
            SourceIndexGenerator: The iterator object.
        """
        return self

    def __next__(self):
        """
        Returns the next index in the sequence.

        Returns:
            int: The next index.

        Raises:
            StopIteration: When all indices have been returned.
        """
        if self.current >= self.total_indices:
            self.current = 0  # Reset for circular behavior
            if self.shuffle:
                np.random.shuffle(self.indices)

        index = self.indices[self.current]
        self.current += 1
        return index

