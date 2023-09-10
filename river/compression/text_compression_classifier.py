from __future__ import annotations

import zstandard  # type: ignore

from river import base


class TextCompressionClassifier(base.Classifier):
    def __init__(self, compression_level=3):
        self.compression_level = compression_level
        self.label_documents = {}  # Concatenated documents for each label
        self.compression_contexts = {}  # Zstd compression contexts for each label

    def learn_one(self, x, y):
        # Convert your input 'x' to a string representation if it's not already
        # For the sake of example, let's assume 'x' is a dictionary of features
        x_str = str(x)

        # Concatenate the new example to the existing document for this label
        self.label_documents[y] = self.label_documents.get(y, "") + x_str

        # Compress new document
        zstd_compressor = zstandard.ZstdCompressor(level=self.compression_level).compress(x_str.encode("utf-8"))
            
        # Create a dictionary using the compressed document
        if y not in self.compression_contexts:
            data_source = zstd_compressor
        else:
            data_source = self.compression_contexts[y].as_bytes() + zstd_compressor
        zstd_dict = zstandard.ZstdCompressionDict(data_source)

        # Update the compression context for this label with the new dictionary
        self.compression_contexts[y] = zstd_dict

        return self

    def predict_one(self, x):
        min_size_increase = float("inf")
        best_label = None

        # Convert your input 'x' to a string representation if it's not already
        x_str = str(x)

        for label, compressor in self.compression_contexts.items():
            # Concatenate and compress
            concatenated_doc = (self.label_documents[label] + x_str).encode("utf-8")
            new_compressed_size = len(zstandard.ZstdCompressor(level=self.compression_level).compress(concatenated_doc))

            # Calculate size increase (you can define your own metric here)
            size_increase = new_compressed_size - len(compressor.as_bytes())

            if size_increase < min_size_increase:
                min_size_increase = size_increase
                best_label = label

        return best_label

    def test(self, arg):
        print("estou na classe !! ", arg)
