import h5py
import numpy as np
import matplotlib.pyplot as plt

def perform_fft_on_hdf5_dataset(hdf5_file_path, dataset_name):
    """
    Opens an HDF5 file, selects a dataset, and performs an FFT on it.

    Args:
        hdf5_file_path (str): The path to the HDF5 file.
        dataset_name (str): The name of the dataset within the HDF5 file.

    Returns:
        tuple: A tuple containing the frequency array and the FFT magnitude array,
               or None if an error occurs.
    """
    try:
        # Open the HDF5 file in read mode
        with h5py.File(hdf5_file_path, 'r') as hdf5_file:
            # Check if the dataset exists
            if dataset_name not in hdf5_file:
                print(f"Error: Dataset '{dataset_name}' not found in the HDF5 file.")
                return None

            # Select the dataset
            dataset = hdf5_file[dataset_name]

            # Read the data from the dataset
            data = dataset[:]

            # Perform FFT
            fft_result = np.fft.fft(data)

            # Calculate the magnitude of the FFT
            fft_magnitude = np.abs(fft_result)

            # Calculate the frequencies
            n = len(data)
            frequencies = np.fft.fftfreq(n)

            return frequencies, fft_magnitude

    except FileNotFoundError:
        print(f"Error: File not found at '{hdf5_file_path}'.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage 
#file_path = 'your_file.hdf5'
#dataset_name = 'your_dataset'
#frequencies, fft_magnitude = perform_fft_on_hdf5_dataset(file_path, dataset_name)

#if frequencies is not None and fft_magnitude is not None:
    # Plot the FFT magnitude
    #plt.plot(frequencies, fft_magnitude)
    #plt.xlabel('Frequency')
