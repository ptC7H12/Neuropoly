class IncrementalSplitter:
    def __init__(self, data, chunk_size):
        self.data = data
        self.chunk_size = chunk_size

    def split(self):
        """
        Splits the data into chunks using a walk-forward method.
        """
        for start in range(0, len(self.data), self.chunk_size):
            end = start + self.chunk_size
            yield self.data[start:end]

# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Example dataset
    data = pd.DataFrame({'value': range(1000)})
    chunk_size = 100

    splitter = IncrementalSplitter(data, chunk_size)
    for chunk in splitter.split():
        print(chunk)