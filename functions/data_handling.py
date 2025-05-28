import os 
import pandas as pd
import numpy as np

def load_preprocess_data(data_path, cosmic_data_path, sep1, sep2, output_folder, output_filename):    
    
    if not os.path.exists(os.path.join(output_folder, output_filename)):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load data and COSMIC data 
        data = pd.read_csv(data_path, sep=sep1) 
        cosmic_data = pd.read_csv(cosmic_data_path, sep=sep2)

        # Set the index of the data to the first column (signature types)
        cosmic_data = cosmic_data.set_index(cosmic_data.columns[0])

        cosmic_aligned = cosmic_data.loc[cosmic_data.index.sort_values()]
        data_aligned = data.loc[cosmic_aligned.index]

        # Check if indices are perfectly aligned (optional)
        assert (cosmic_aligned.index == data_aligned.index).all(), "Indices are not perfectly aligned!"

        # Save the aligned data to a new file
        data_aligned.to_csv(os.path.join(output_folder, output_filename))
        print("Data saved to", os.path.join(output_folder, output_filename))

        print("Data loaded and aligned successfully!")
    
    else:
        print("Data already exists in ", os.path.join(output_folder, output_filename))



def data_augmentation(X: pd.DataFrame, augmentation: int = 5) -> pd.DataFrame:
    '''
    Performs data augmentation by bootstrapping each tumour (column) `augmentation` times using a multinomial 
    distribution M(N, p), where:
    - N is the total mutation count for the tumour.
    - p is the relative frequency of each of the 96 mutational classes.

    Parameters:
    X (pd.DataFrame): Input count data (96 mutational signatures as rows, patients as columns).
    augmentation (int): Number of bootstrap samples to generate per tumour.

    Returns:
    pd.DataFrame: A new DataFrame containing only the augmented data (96 rows × (patients * augmentation) columns).
    '''

    augmented_columns = []

    for i in range(augmentation):
        X_bootstrapped = X.copy()  # Copy structure

        for col in X.columns:  # Iterate over patients
            N = np.sum(X[col])  # Total number of mutations for this patient
            if N == 0:
                # If no mutations, return zero vector instead of multinomial sampling
                X_bootstrapped[col] = np.zeros_like(X[col])
            else:
                p = X[col] / np.sum(X[col])  # Properly normalized probabilities
                X_bootstrapped[col] = np.random.multinomial(N, p)

        # Rename columns to indicate augmentation round
        X_bootstrapped.columns = [str(col) + '_aug_' + str(i) for col in X.columns]
        augmented_columns.append(X_bootstrapped)

    # Concatenate all augmented versions **horizontally**
    X_augmented = pd.concat(augmented_columns, axis=1)

    return X_augmented


def data_normalization(X : pd.DataFrame) -> pd.DataFrame:
    '''
    A function that normalizes the input data. Here X is expected to be a pandas DataFrame that contains
    the count data, rows represent signatures and columns patients.

    Parameters:
    X (pd.DataFrame): The input data to be normalized
    returns (pd.DataFrame): The normalized data
    '''

    # Calculate the total number of mutations per patient

    total_mutations = X.sum(axis=1)

    # Repeat the total number of mutations for each patient

    total_mutations = pd.concat([total_mutations] * X.shape[1], axis=1)

    # Set the column names of the total_mutations DataFrame to match the input data

    total_mutations.columns = X.columns

    # Normalize the data using the log-ratio transformation (Count data follows a Poisson distribution, in theory, so
    # the log-ratio transformation feels appropriate)

    norm_data = X / total_mutations * np.log2(total_mutations)

    return np.array(norm_data, dtype='float64')
    


