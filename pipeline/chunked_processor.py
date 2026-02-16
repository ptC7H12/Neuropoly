import pandas as pd

class ChunkedProcessor:
    def __init__(self, file_path, chunk_size=10000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def process_chunk(self, chunk):
        # Perform processing on the chunk of data
        # For example, just printing the size of the chunk
        print(f"Processing chunk of size: {len(chunk)}")
        # You can include your processing logic here

    def process(self):
        # Read the dataset in chunks
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            self.process_chunk(chunk)

# Usage example:
# processor = ChunkedProcessor('path_to_large_file.csv')
# processor.process()