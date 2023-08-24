import torch
import numpy as np

class Categorical:
    def __init__(self, logits):
        self.probs = self.logits_to_probs(logits)
        

    def logits_to_probs(self, logits):
        e_x = torch.exp(logits - torch.max(logits, axis=-1, keepdims=True)[0])
        return e_x / torch.sum(e_x, axis=-1, keepdims=True)
    
    def sample(self):
        