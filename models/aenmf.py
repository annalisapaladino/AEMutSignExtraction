import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def frobenius_loss(output, target):
    return torch.norm(output - target, p='fro')


class aenmf(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization

    """
    def __init__(self, input_dim, latent_dim):
        '''
        Constructor for the Autoencoder for Non-negative Matrix Factorization (AENMF) model.

        Parameters:
        - input_dim: Dimension of the input data
        - latent_dim: Dimension of the latent space
        '''
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.activation = nn.Identity()
        

        ''' Encoder and Decoder layers '''
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            self.activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dim),
            self.activation
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=0.0, b=1.0)
                module.weight.data.clamp_(min=0)
    
    def load_custom_weights(self, signature, encoder_bias=None):
        """
        Load custom weights (and optionally bias) into the encoder.
        
        Parameters:
        - signature: torch.Tensor of shape (input_dim, latent_dim) (should be signature.T from `train_aenmf`)
        - encoder_bias (optional): torch.Tensor of shape (latent_dim,)
        """
        if signature.shape != (self.input_dim, self.latent_dim):
            raise ValueError(f"Expected signature of shape ({self.input_dim}, {self.latent_dim}), but got {signature.shape}")
        
        with torch.no_grad():
            self.encoder[0].weight.copy_(signature.T)  # Load weights directly
            if encoder_bias is not None:
                self.encoder[0].bias.copy_(encoder_bias)

        # Ensure non-negative constraint
        self.encoder[0].weight.data.clamp_(min=0)

    def freeze_encoder(self):
        """Freeze encoder weights for inference or decoder-only training."""
        for param in self.encoder.parameters():
            param.requires_grad = False



    def forward(self, x):
        '''
        Forward pass of the Autoencoder for Non-negative Matrix Factorization

        '''
        x = self.encoder(x)
        x = self.decoder(x)
    
        return x
    
    
def train_aenmf(model, training_data, optimizer, tol = 1e-3, relative_tol = True, max_iter = 100_000_000):
    '''
    Function to train the Autoencoder for Non-negative Matrix Factorization (AENMF) model.

    Parameters:
    - model: AENMF model
    - training_data: Training data (Note, we assume to reconstruct X^T = E^TS^T so the input data should have shape n x m (m being 96))
    - criterion: Loss function
    - optimizer: Optimizer
    - tol: Tolerance for convergence
    - relative_tol: If True, the tolerance is relative. If False, the tolerance is absolute.
    - max_iter: Maximum number of iterations
    
    Returns:
    - model: Trained model
    - training_loss: Training loss
    - signature: Signature matrix
    - exposure: Exposure matrix
    '''

    training_data_tensor = torch.tensor(training_data.values, dtype = torch.float32)

    training_loss = []
    diff = float('inf')

    iters = 0
    while diff > tol and iters < max_iter: # Convergence criterion
        optimizer.zero_grad() 
        output = model(training_data_tensor)
        loss = frobenius_loss(output, training_data_tensor)
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            # Clamp the weights to non-negative values
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.clamp_(min=0)  # Ensure non-negative weights

        training_loss.append(loss.item())


        if len(training_loss) > 1:
            if relative_tol:
                diff = abs(training_loss[-1] - training_loss[-2])/training_loss[-2]
            else:
                diff = abs(training_loss[-1] - training_loss[-2])

        
        # if(iters % 1000 == 0):
        #     print(f"Iteration {iters}: Loss = {loss.item()}")

        # Go to next iteration
        iters += 1
    

    # Get the encoder and decoder weights

    enc_weights = model.encoder[0].weight.data.T
    dec_weights = model.decoder[0].weight.data.T

    if(torch.any(enc_weights < 0)):
        raise ValueError("Negative values present in the encoder weights")
    if(torch.any(dec_weights < 0)):
        raise ValueError("Negative values present in the decoder weights")

    exposure = training_data @ enc_weights
    signature = dec_weights 



    return model, training_loss, signature.T, exposure.T



def test():
    pass