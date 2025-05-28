import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import pandas as pd
#from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker


# Function to plot Kmedoids clusters reduced with PCA
def plot_clusters(reduced_signatures, labels, medoid_indices, LATENT_DIM : int, title : str = "AENMF Signatures"):

    plt.figure(figsize=(10, 6))
    # Plot all points colored by cluster
    for cluster in range(LATENT_DIM):
        cluster_points = reduced_signatures[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

    # Highlight medoids
    medoid_points = reduced_signatures[medoid_indices]
    plt.scatter(medoid_points[:, 0], medoid_points[:, 1], 
                color='black', marker='+', s=100, label="Medoids")

    plt.title(title)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend()
    plt.show()


def base_plot_signature(array, axs, index, ylim=1):

    color = ((0.196,0.714,0.863),)*16 + ((0.102,0.098,0.098),)*16 + ((0.816,0.180,0.192),)*16 + \
            ((0.777,0.773,0.757),)*16 + ((0.604,0.777,0.408),)*16 + ((0.902,0.765,0.737),)*16
    
    color = list(color)

    width = max(array.shape)
    x = np.arange(width)
    if axs == None:
        f,axs = plt.subplots(1,figsize=(20,10))
    bars = axs.bar(x, array, edgecolor='black', color=color)

    plt.ylim(0, ylim)
    plt.yticks(fontsize=10)
    axs.set_xlim(-0.5, width) 
    axs.set_ylabel('Probability of mutation \n', fontsize=12)
    axs.set_xticks([])
    axs.set_xticks(x)  
    axs.set_xticklabels(index, rotation=90, fontsize=7)
    


def plot_signature(signatures, model : str = "Autoencoder"):
    """
    Visualize mutational signatures directly in a Jupyter Notebook.
    
    Parameters:
    - signatures: DataFrame containing mutational signatures (96 rows x n columns).
    - name: Base name for the plot titles (default: 'DeNovo_Signatures').
    """
    # Define the mutation types (96 trinucleotide contexts)
    index = ['A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T','C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T',
             'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T', 'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T',
             'A[C>G]A', 'A[C>G]C', 'A[C>G]G', 'A[C>G]T', 'C[C>G]A', 'C[C>G]C', 'C[C>G]G', 'C[C>G]T',
             'G[C>G]A', 'G[C>G]C', 'G[C>G]G', 'G[C>G]T', 'T[C>G]A', 'T[C>G]C', 'T[C>G]G', 'T[C>G]T',
             'A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T', 'C[C>T]A', 'C[C>T]C', 'C[C>T]G', 'C[C>T]T',
             'G[C>T]A', 'G[C>T]C', 'G[C>T]G', 'G[C>T]T', 'T[C>T]A', 'T[C>T]C', 'T[C>T]G', 'T[C>T]T',
             'A[T>A]A', 'A[T>A]C', 'A[T>A]G', 'A[T>A]T', 'C[T>A]A', 'C[T>A]C', 'C[T>A]G', 'C[T>A]T',
             'G[T>A]A', 'G[T>A]C', 'G[T>A]G', 'G[T>A]T', 'T[T>A]A', 'T[T>A]C', 'T[T>A]G', 'T[T>A]T', 
             'A[T>C]A', 'A[T>C]C', 'A[T>C]G', 'A[T>C]T', 'C[T>C]A', 'C[T>C]C', 'C[T>C]G', 'C[T>C]T',
             'G[T>C]A', 'G[T>C]C', 'G[T>C]G', 'G[T>C]T', 'T[T>C]A', 'T[T>C]C', 'T[T>C]G', 'T[T>C]T',
             'A[T>G]A', 'A[T>G]C', 'A[T>G]G', 'A[T>G]T', 'C[T>G]A', 'C[T>G]C', 'C[T>G]G', 'C[T>G]T',
             'G[T>G]A', 'G[T>G]C', 'G[T>G]G', 'G[T>G]T', 'T[T>G]A', 'T[T>G]C', 'T[T>G]G', 'T[T>G]T']

    n_signatures = signatures.shape[1]  # Number of signatures
    
    for signature in range(n_signatures):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Extract the signature values
        s = signatures.loc[index].values[:, signature]
        
        base_plot_signature(s, axs=ax, index=index, ylim=max(s) + 0.05)
        
        # Define legend patches
        l1 = mpatches.Patch(color=(0.196, 0.714, 0.863), label='C>A')
        l2 = mpatches.Patch(color=(0.102, 0.098, 0.098), label='C>G')
        l3 = mpatches.Patch(color=(0.816, 0.180, 0.192), label='C>T')
        l4 = mpatches.Patch(color=(0.777, 0.773, 0.757), label='T>A')
        l5 = mpatches.Patch(color=(0.604, 0.777, 0.408), label='T>C')
        l6 = mpatches.Patch(color=(0.902, 0.765, 0.737), label='T>G')
        
        # Add title and legend
        ax.text(0.01, 0.94, f'{model}-SBS {chr(64 + signature + 1)}\n', transform=ax.transAxes, 
                fontsize=15, fontweight='bold', va='top', ha='left')
        ax.legend(handles=[l1, l2, l3, l4, l5, l6], loc='upper center', ncol=6, 
                  bbox_to_anchor=(0.5, 1.1), fontsize=18)
        
        plt.show()



  
def plot_signature_frequency(all_matches: pd.DataFrame, title: str = "Signature Appearances", figsize: tuple = (12, 7), color: str = 'tab20') -> None:
    all_signatures = all_matches.iloc[:, 0::2]

    # Count occurrences of each signature across the entire dataset
    signature_counts = all_signatures.stack().value_counts().sort_values(ascending=False)

    custom_cmap = plt.cm.get_cmap(color, 20)
    sig_to_color = {
        'SBS1' : custom_cmap(16),
        'SBS3' : custom_cmap(12),
        'SBS5' : custom_cmap(11),
        'SBS6' :custom_cmap(9),
        'SBS8' : custom_cmap(1),
        'SBS10a' : custom_cmap(5),
        'SBS10c' : custom_cmap(3),
        'SBS12' : custom_cmap(13),
        'SBS13' : custom_cmap(0),
        'SBS15' : custom_cmap(2),
        'SBS26' : custom_cmap(4),
        'SBS40a': custom_cmap(18),
        'SBS40c' : custom_cmap(19),
        'SBS44' : custom_cmap(7),
        'SBS52' : custom_cmap(8),
        'SBS57' : custom_cmap(10),
        'SBS56' : custom_cmap(14),
        'SBS28' : custom_cmap(15),
        'SBS2': custom_cmap(17),
        'SBS10d': custom_cmap(6),
        }
    # Assign colors based on sig_to_color, fallback to default color if not in dict
    bar_colors = [sig_to_color.get(sig, custom_cmap(2)) for sig in signature_counts.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(signature_counts.index, signature_counts.values, color=bar_colors, edgecolor='black', alpha=0.7)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Signatures", fontsize=16)
    ax.set_ylabel("Number of Occurrences", fontsize=16)

    ax.set_xticks(range(len(signature_counts.index)))
    ax.set_xticklabels(signature_counts.index, rotation=0, ha='center', fontsize=12)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_cosine_similarity_matrix(all_matches: pd.DataFrame, title : str ="Cosine similarity matrix", figsize : tuple =(7, 7), legend_colums : int =4, palette = 'tab20') -> None:
    all_signatures = all_matches.iloc[:, 0::2]
    all_similarities = all_matches.iloc[:, 1::2]

    # Find the unique signatures
    unique_signatures = sorted(all_signatures.stack().unique())

    # Custom colormap
    #custom_colors = [

    # Create a ListedColormap
    ##custom_cmap = ListedColormap(custom_colors)
    #ustom_cmap
    
    custom_cmap = plt.cm.get_cmap(palette, 20)
    # Map each signature to an RGBA color
    sig_to_color = {
        'SBS1' : custom_cmap(16),
        'SBS3' : custom_cmap(12),
        'SBS5' : custom_cmap(11),
        'SBS6' :custom_cmap(9),
        'SBS8' : custom_cmap(1),
        'SBS10a' : custom_cmap(5),
        'SBS10c' : custom_cmap(3),
        'SBS12' : custom_cmap(13),
        'SBS13' : custom_cmap(0),
        'SBS15' : custom_cmap(2),
        'SBS26' : custom_cmap(4),
        'SBS40a': custom_cmap(18),
        'SBS40c' : custom_cmap(19),
        'SBS44' : custom_cmap(7),
        'SBS52' : custom_cmap(8),
        'SBS57' : custom_cmap(10),
        'SBS56' : custom_cmap(14),
        'SBS28' : custom_cmap(15),
        'SBS2': custom_cmap(17),
        'SBS10d': custom_cmap(6),
        }
    
   # Create a color matrix
    N, M = all_signatures.shape
    color_matrix = np.zeros((N, M, 4))  # 4 for RGBA

    
    
    for i in range(N):
        for j in range(M):
            color_matrix[i, j] = sig_to_color[all_signatures.iloc[i, j]]

    # Plot colored by signature
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(color_matrix, interpolation='nearest')



    # Overlay similarity values
    for i in range(N):
        for j in range(M):

            ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color='white', linewidth=3)  # Left edge
            ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color= 'white', linewidth=3)  # Right edge

            val = all_similarities.iloc[i, j]
            ax.text(
                j, i,                
                f"{val:.2f}",        
                ha='center',
                va='center',
                color='black',
                fontsize=8
            )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_title(title)
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor=sig_to_color[sig], edgecolor='black', label=sig)
        for sig in unique_signatures if sig in sig_to_color
    ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, -0.1),
        loc='upper center',
        ncol=legend_colums,  # how many columns to use in the legend
        title="Matched signatures"
    )

    plt.show()
