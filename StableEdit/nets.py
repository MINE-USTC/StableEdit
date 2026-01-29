from typing import Tuple
import torch
import torch.nn as nn
import wandb
import swanlab
import logging
import math
log = logging.getLogger(__name__)

class RunningMeanStd(nn.Module):

    def __init__(self, size: int, label: str = "original", track_stats: bool = True, v_dim: int = None):
        super().__init__()

        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros((size)))
        self.register_buffer("var", torch.zeros((size)))
        self.register_buffer("std", torch.zeros((size)))
        

        self.label = label
        self.track_stats = track_stats
        self.v_dim = v_dim
        self.k_dim = size - v_dim if v_dim is not None else None
      
        if self.v_dim is not None:
            # assert size == self.k_dim + self.v_dim, f"size ({size}) != k_dim + v_dim ({self.k_dim} + {self.v_dim})"
            self.register_buffer("cov_key", torch.zeros((self.k_dim, self.k_dim)))
            self.register_buffer("cov_value", torch.zeros((self.v_dim, self.v_dim)))
        else:
            self.register_buffer("cov", torch.zeros((size, size)))
        self.step = 0
        self.mode = None

    def update(self, x: torch.FloatTensor, mode="inference") -> None:
        self.mode = mode
     
        n = self.n + x.shape[0]
        delta = x.mean(0) - self.mean
        self.mean += x.shape[0] * delta / n
      

        if self.label == "stable":
            
            self.var += x.shape[0] * x.var(0, unbiased=False) + self.n * x.shape[0] * delta.pow(2) / n
            self.std = (self.var / (n - 1 + torch.finfo(x.dtype).eps)).sqrt()
            

            if self.v_dim is not None:
                assert x.shape[1] == self.k_dim + self.v_dim, f"x.shape[1] ({x.shape[1]}) != k_dim + v_dim ({self.k_dim} + {self.v_dim})"
                _, v = x.split([self.k_dim, self.v_dim], dim=1)

                self.cov_value += v.shape[0] * torch.cov(v.T, correction=0) + self.n * v.shape[0] * torch.ger(delta[self.k_dim:], delta[self.k_dim:]) / n
                          
            else:
                self.cov += x.shape[0] * torch.cov(x.T, correction=0) + self.n * x.shape[0] * torch.ger(delta, delta) / n

           
        elif self.label == "original":  

            self.var += x.shape[0] * x.var(0, unbiased=False) + self.n * x.shape[0] * delta.pow(2) / n
            self.std = (self.var / (n - 1 + torch.finfo(x.dtype).eps)).sqrt()

            
        else:
            raise ValueError(f"Unknown RunningMeanStd mode: {self.label}")
        
        self.n = n

     
        if self.track_stats and self.mode == "inference":
            self.step += 1


            swanlab.log({
                f"n": float(self.n),
                f"mean_abs": self.mean.abs().mean().item(),
                f"var_mean": self.var.mean().item(),
                f"std_mean": self.std.mean().item(),
                f"std_max": self.std.max().item(),
                f"std_min": self.std.min().item(),
                f"delta_norm": delta.norm().item()
            }, step=self.step)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.label == "stable":
            if self.mode == "train":
                return (x - self.mean) / (self.std + torch.finfo(x.dtype).eps)
            else:
               
                if self.v_dim is not None and self.n > self.v_dim + 1:
   

                    self.cov_value_ = self.cov_value / (self.n - self.v_dim - 1 + torch.finfo(x.dtype).eps)

                    eigvals_v, eigvecs_v = torch.linalg.eigh(self.cov_value_)

                    cov_value_inv_sqrt = eigvecs_v @ torch.diag(eigvals_v.clamp(min=1e-6).rsqrt()) @ eigvecs_v.T
                    x_k, x_v = x.split([self.k_dim, self.v_dim], dim=1)

                    x_k = (x_k - self.mean[:self.k_dim])/(self.std[:self.k_dim] + torch.finfo(x.dtype).eps)
                    x_v = (x_v - self.mean[self.k_dim:]) @ cov_value_inv_sqrt
                 
                    x = torch.cat([x_k, x_v], dim=1)
                    return x
                  
                else:
                    
                    return (x - self.mean) / (self.std + torch.finfo(x.dtype).eps)

        else:
            return (x - self.mean) / (self.std + torch.finfo(x.dtype).eps)
        



class RLEditBlock(nn.Module):

    def __init__(self, size: int, rank: int, n_modules: int):
        super().__init__()

        self.A = nn.Parameter(torch.randn(size, rank))
        self.B = nn.Parameter(torch.zeros(rank, size))
        self.bias = nn.Parameter(torch.zeros(size))
        
        self.scale = nn.Embedding(n_modules, size)
        self.shift = nn.Embedding(n_modules, size)
        
        self.scale.weight.data.fill_(1)
        self.shift.weight.data.fill_(0)


    def forward(
        self,
        y: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> torch.FloatTensor:

        x = y @ self.A @ self.B + self.bias
        x = x.clamp(0) 

        x = self.scale(module_idx) * x + self.shift(module_idx)
        
        x = x + y

        return x


class RLEditNet(nn.Module):

    def __init__(
        self,
        key_size: int,
        value_size: int,
        rank: int,
        n_blocks: int,
        n_modules: int,
        lr: float
    ):
        super().__init__()
        self.layer_idx = key_size
        self.value_size = value_size
        self.key_size=key_size

        self.normalizer = RunningMeanStd(key_size + value_size, label='original', v_dim=value_size)
        self.blocks = nn.ModuleList([
            RLEditBlock(key_size + value_size, rank, n_modules)
            for _ in range(n_blocks)
        ])

        self.lr = nn.Embedding(n_modules, 1)
        self.lamda = nn.Embedding(n_modules, 1)
        
        self.lr.weight.data.fill_(lr)
        self.lamda.weight.data.fill_(0)
        # log.info('This turn without Lifelong Nomalization.')



    def forward(
        self,
        keys: torch.FloatTensor,
        values_grad: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> Tuple[torch.FloatTensor]:

        hidden_states = torch.cat((keys, values_grad), -1)
        hidden_states = self.normalizer(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, module_idx)
        return hidden_states.split([self.key_size, self.value_size], -1)


