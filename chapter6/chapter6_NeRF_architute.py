import torch 
class NeuralRadianceField(torch.nn.Module):
    def __init__(self,n_harmonic_functions=60,n_hidden_neurons=256) -> None:
        super().__init__()
        self.harmonic_embedding = Ha