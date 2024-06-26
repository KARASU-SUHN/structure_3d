import os
import numpy as np
import pandas as pd

# Example filling_date function (you should replace this with your actual function)
def filling_date(npy_data):
    # Placeholder for your actual filling_date function implementation
    # This example just returns the sum of the npy_data as a dummy implementation
    return np.sum(npy_data)

# Function to read .npy files, process them, and match with DataFrame paths
def process_npy_files_and_match(df, folder_path):
    # Dictionary to store results
    results = {}

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Loop through the files and read each .npy file
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            npy_data = np.load(file_path)
            
            # Process the .npy data using the filling_date function
            filled_data = filling_date(npy_data)

            # Extract the filename from the file path
            filename = os.path.basename(file_path)

            # Check if the filename matches any entry in the fpath column of the DataFrame
            match = df[df['fpath'].apply(lambda x: os.path.basename(x)) == filename]
            if not match.empty:
                # Get the corresponding value from the value column
                value = match.iloc[0]['value']
                results[file_path] = (value, filled_data)

    return results

# Example usage
data = {
    'value': [10, 20, 30, 40, 50],
    'fpath': [
        './geometry/file1.npy',
        './geometry/file2.npy',
        './geometry/file3.npy',
        './geometry/file4.npy',
        './geometry/file5.npy'
    ]
}

df = pd.DataFrame(data)
folder_path = 'path/to/your/folder/geometry'

results = process_npy_files_and_match(df, folder_path)
print(results)
