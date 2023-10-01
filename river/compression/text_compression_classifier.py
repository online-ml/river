from __future__ import annotations
from collections import deque

import zstandard  # type: ignore

from river import base


class TextCompressionClassifier(base.Classifier):
    def __init__(self, compression_level=3, k=150):
        self.compression_level = compression_level
        self.k = k
        self.label_documents = {}  # Concatenated documents for each label
        self.compression_contexts = {}  # Zstd compression contexts for each label

    def learn_one(self, x, y):
        # Convert your input 'x' to a string representation if it's not already
        # For the sake of example, let's assume 'x' is a dictionary of features
        x_str = str(x)

        # Initialize if label is new
        if y not in self.label_documents:
            self.label_documents[y] = deque(maxlen=self.k)

        # Append the new document and remove the oldest if length > k
        self.label_documents[y].append(x_str)

        # Concatenate documents in the deque into a single string
        concatenated_documents = " ".join(self.label_documents[y])

        # Create a dictionary with encoded concatenated text
        compression_dict = zstandard.ZstdCompressionDict(concatenated_documents.encode("utf-8"))

        # Create a Zstandard compression context for this label using the dictionary
        zstd_compressor = zstandard.ZstdCompressor(
            level=self.compression_level, dict_data=compression_dict
        )

        # Update the compression context for this label with the new compressor
        self.compression_contexts[y] = zstd_compressor

        return self

    def predict_one(self, x):
        """Predict the output label for the input `x`.

        Parameters
        ----------
        x : any type that can be converted to string
            The input to be classified.

        Returns
        -------
        best_label: string
            The label corresponding to the smallest increase in compressed size after
            adding the new data `x`.
        """
        min_size_increase = float("inf")
        best_label = None

        # Convert your input 'x' to a string representation if it's not already
        x_str = str(x)

        for label, compressor in self.compression_contexts.items():
            concatenated_doc = (" ".join(self.label_documents[label]) + " " + x_str).encode("utf-8")
            compressed_size = len(compressor.compress(concatenated_doc))

            previous_size = len(
                compressor.compress(" ".join(self.label_documents[label]).encode("utf-8"))
            )

            size_increase = compressed_size - previous_size

            if size_increase < min_size_increase:
                min_size_increase = size_increase
                best_label = label

        return best_label
