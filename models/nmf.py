import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF as nmf_sklearn # Just for comparison
import os

def nmf(catalog_matrix : np.ndarray, num_sign : int, tol : float = 1e-6, max_iter : int = 10_000_000) -> tuple:
    """
    Performs Non-negative Matrix Factorization with a given tolerance level.

    Parameters:
    catalog_matrix (numpy.ndarray): Catalog matrix of shape m x n where m is the number of SBS mutation types (96) and n is the number of patiens
    num_sign (int): Number of signatures to be extracted
    tol (float): Tolerance level for convergence
    max_iter (int): Maximum number of iterations before analysis is disrupted (default is 10e8).

    
    Returns:
    S (numpy.ndarray): Signature matrix of shape (m, num_sign). (Base)
    E (numpy.ndarray): Exposure matrix of shape (num_sign, n). (Weights)
    losses (list): List of loss values at each iteration.
    """

    # Make sure catalog_matrix is a numpy array

    if not isinstance(catalog_matrix, np.ndarray):
        # Convert to numpy array
        catalog_matrix = np.array(catalog_matrix)
    
    m,n = catalog_matrix.shape
    losses = []
    # Initialize the signature and exposure matrices randomly
    S = np.random.rand(m, num_sign)
    E = np.random.rand(num_sign, n)

    # Compute the loss (Frobenius norm squared)
    loss = np.linalg.norm(catalog_matrix - S@E, ord = 'fro')
    losses.append(loss)
    
    diff = float('inf')
    n_iter = 0
    
    while(diff > tol and n_iter < max_iter):
        n_iter += 1 


        E = E*(np.divide(S.T@catalog_matrix, S.T@S@E))
        S = S*(np.divide(catalog_matrix@(E.T), S@E@(E.T)))
        
        loss = np.linalg.norm(catalog_matrix - S@E, ord = 'fro')
        losses.append(loss)

        diff = abs(losses[-1] - losses[-2])
        

        #if n_iter%1000 == 0:
            #print(f"Iteration: {n_iter}, Loss: {losses[-1]}")

    return S, E, losses


def refit_NMF(catalog_matrix: np.ndarray, signature_matrix: np.ndarray, tol : float = 1e-6, max_iter : int = 10_000_000) -> list:
    '''
    Performs NMF on a catalog matrix using a fixed signature matrix. This function is used on the test dataset to assess if the
    signature matrix extracted from the train data is good.

    Parameters:
    catalog_matrix (numpy.ndarray): Catalog matrix of shape m x n where m is the number of SBS mutation types (96) and n is the number of patiens
    signature_matrix (numpy.ndarray): Signature matrix of shape (m, num_sign). (Base)
    tol (float): Tolerance level for convergence
    max_iter (int): Maximum number of iterations before analysis is disrupted (default is 10e8).

    Returns:
    losses (list): List of loss values at each iteration.
    '''

    n = catalog_matrix.shape[1]
    losses = []

    num_sign = signature_matrix.shape[1]

    E = np.random.rand(num_sign, n)

    # Compute the loss (Frobenius norm squared)
    loss = np.linalg.norm(catalog_matrix - signature_matrix@E, ord = 'fro')
    losses.append(loss)

    diff = float('inf')
    n_iter = 0

    while(diff > tol and n_iter < max_iter):
        n_iter += 1 

        E = E*(np.divide(signature_matrix.T@catalog_matrix, signature_matrix.T@signature_matrix@E))
        
        loss= np.linalg.norm(catalog_matrix - signature_matrix@E, ord = 'fro')
        losses.append(loss)

        diff = abs(losses[-1] - losses[-2])

        # if n_iter%1000 == 0:
        #     print(f"Iteration: {n_iter}, Loss: {losses[-1]}")

    return losses


if __name__ == '__main__':

    
    print("############ Testing the NMF function ############")

    # Test the NMF function on a dummy dataset
    catalog_matrix = np.random.rand(96, 532) # Ovary cancer dataset dummy matrix
    num_sign = 4
    S, E, losses_NMF = nmf(catalog_matrix, num_sign)
    print(S.shape, E.shape, len(losses_NMF))

    rec = np.round(S@E,2)

    plt.plot(losses_NMF)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration Ovary dummy dataset')

    model = nmf_sklearn(n_components = num_sign, init = 'random', max_iter = int(10e8))

        
    S = model.fit_transform(catalog_matrix)
    E = model.components_
    loss_sklearn = model.reconstruction_err_


    plt.plot(loss_sklearn)


    plt.legend(['NMF', 'NMF_sklearn'])

    plt.show()

    # Print the last loss values

    print("Loss value of the last iteration of NMF: ", losses_NMF[-1])
    print("Loss value of the last iteration of NMF_sklearn: ", loss_sklearn)


    # Test the NMF function on the real dataset

    os.chdir("..")

    catalog_matrix = pd.read_csv("data/catalogues_Ovary_SBS.tsv", sep = '\t')

    # Extract 418 patients from the dataset

    catalog_matrix = catalog_matrix.iloc[:, 1:419].values

    num_sign = 4
    S, E, losses_NMF = nmf(catalog_matrix, num_sign)
    print(S.shape, E.shape, len(losses_NMF))

    rec = S@E

    plt.plot(losses_NMF)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration Ovary real dataset')

    model = nmf_sklearn(n_components = num_sign, init = 'random', max_iter = int(10e8), solver = 'mu', tol = 1e-6)

        
    S = model.fit_transform(catalog_matrix)
    E = model.components_
    loss_sklearn = model.reconstruction_err_


    plt.plot(loss_sklearn)


    plt.legend(['NMF', 'NMF_sklearn'])

    plt.show()


    print("Loss value of the last iteration of NMF: ", losses_NMF[-1])
    print("Loss value of the last iteration of NMF_sklearn: ", loss_sklearn)

    print("############ End of testing ############")
