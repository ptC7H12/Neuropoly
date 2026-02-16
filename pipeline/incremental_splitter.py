def chunked_walk_forward_split(data, chunk_size):
    """
    Generator that yields training data in chunks for walk-forward validation.

    Parameters:
    - data: A large dataset as a pandas DataFrame.
    - chunk_size: The size of each chunk to yield.

    Yields:
    - DataFrame: A chunk of the original DataFrame.
    """
    for start in range(0, len(data), chunk_size):
        end = min(start + chunk_size, len(data))
        yield data[start:end]

# Example usage (uncomment for testing):
# import pandas as pd
# data = pd.DataFrame({'column1': range(144000000), 'column2': range(144000000)})
# for chunk in chunked_walk_forward_split(data, 1000000):
#     print(chunk)

