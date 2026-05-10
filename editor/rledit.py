from typing import Dict
from omegaconf import DictConfig

import math
import os

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from nets import RLEditNet

from editor.base import BaseEditor
from util import get_module, get_shape, TracerDict

import numpy as np
from nets import RLEditNet

from itertools import islice
import logging
log = logging.getLogger(__name__)


from tqdm import tqdm
import swanlab

from util import (
    get_module,
    get_shape,
    empty_cache,
    cross_entropy,
    kl_div,
)
from datetime import datetime

def pad_tensor(tensor, target_length, dim=0, padding_value=0):

    tensor_length = tensor.size(dim)
    if tensor_length >= target_length:
        return tensor.narrow(dim, 0, target_length)
    else:
        padding = target_length - tensor_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = padding
        pad_tensor = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        mask = torch.cat([torch.ones(tensor_length, dtype=torch.float32, device=tensor.device),
                          torch.zeros(padding, dtype=torch.float32, device=tensor.device)], dim=0)
        return torch.cat([tensor, pad_tensor], dim=dim)


class RLEDIT(BaseEditor):

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        super().__init__(
            config,
            model
        )
        self.net = nn.ModuleDict({
            str(k): RLEditNet(
                *k,
                config.editor.rank,
                config.editor.n_blocks,
                v,
                config.editor.lr
            )
            for k, v in self.shape_counter.items()
        }).to(config.editor_device)

        self.opt = torch.optim.Adam(
            self.net.parameters(),
            config.editor.meta_lr
        )
        if config.editor.load_checkpoint:
            raise NotImplementedError("Loading checkpoint for RLEDIT is not implemented yet.")

        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def reset_hypernet(self):

        self.net = nn.ModuleDict({
            str(k): RLEditNet(
                *k,
                self.config.editor.rank,
                self.config.editor.n_blocks,
                v,
                self.config.editor.lr
            )
            for k, v in self.shape_counter.items()
        }).to(self.config.editor_device)
        
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            self.config.editor.meta_lr
        )


    def train(self, loader: DataLoader, save=False):
        """
        The training method for RLEdit.
        Model the sequential editing as a Markov Devision Process, and use the Paradigm of Reinforce Learning to solve the question.
        """

        sequence_tuples = []
        max_steps = self.config.num_seq
        time_decay = self.config.editor.time_decay

        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Train", ncols=100, total=max_steps)):

            sequence_tuples.append(tuples)
            self.cache(tuples["edit_tuples"],mode='inference')
            param_shifts = self.predict_param_shifts(mode='train')
            self.model.zero_grad()

            l2_reg_loss = 0
            for _, param_shift in param_shifts.items():
                l2_reg_loss += torch.sum(param_shift ** 2) 
            l2_reg_loss *= self.config.editor.reg_coef

            gen_losses_show = []
            self.edit_model(param_shifts, False)

            for back_idx, step_tuple in enumerate(reversed(sequence_tuples)):
                loss_e = 0.0
                for t in step_tuple["equiv_tuples"]:
                    logits = self.model(**t)["logits"]
                    loss_e = loss_e + cross_entropy(logits, t["labels"])
                gen_losses_show.append(loss_e.item())
                (loss_e * pow(time_decay, back_idx)).backward()  
                if back_idx + 1 >= self.config.editor.back_depth:
                    break

            self.edit_model(param_shifts, True)

            loc_losses_show = []
            for back_idx, step_tuple in enumerate(reversed(sequence_tuples)):
                loss_loc = 0.0
                for t in step_tuple["unrel_tuples"]:
                    with torch.no_grad():
                        refer_logits = self.model(**t)["logits"]  
                    self.edit_model(param_shifts, False)
                    logits = self.model(**t)["logits"]         
                    loss = kl_div(
                        refer_logits,
                        logits,
                        t["labels"]
                    )
                    loss_loc = loss_loc + (self.config.editor.loc_coef * loss)
                    self.edit_model(param_shifts, True)
                loc_losses_show.append(loss_loc.item())
                (loss_loc * pow(time_decay, back_idx)).backward()
                if back_idx + 1 >= self.config.editor.back_depth:
                    break
        
            self.edit_model(param_shifts, False)
            self.update_hypernet(param_shifts, False)


            swanlab.log({
                "gen_loss": np.mean(gen_losses_show),
                "loc_loss": np.mean(loc_losses_show)
            })

        self.opt.step()
        self.opt.zero_grad()

        if self.config.editor.save_checkpoint:
            hypernet_dir = f"saved_hypernets/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}"
            os.makedirs(hypernet_dir, exist_ok=True)
            torch.save({
                "net": self.net.state_dict(),
                "opt": self.opt.state_dict(),
                "config": dict(self.config),
            }, f"{hypernet_dir}/hypernet.pth")
            print(f"Hypernet saved to {hypernet_dir}")


    def predict_param_shifts(self, mode=None) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):

            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            keys = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}_{self.time}/{module_idx}_{idx}_keys.pth")
                for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size))
            ])
            values_grad = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}_{self.time}/{module_idx}_{idx}_values_grad.pth")
                for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size))
            ])
            value_diffs = torch.empty((0, net.value_size), device = self.config.editor_device)
            for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                keys_once = pad_tensor(keys[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
                with torch.no_grad():
                    (pesudo_keys, pesudo_values_grad) = net(
                        keys_once,
                        values_grad_once,
                        layer_idx,
                    )
                    coeffs = - net.lr(layer_idx) * (keys_once * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diffs = torch.cat((value_diffs, coeffs * pesudo_values_grad))
            

            with torch.no_grad():
                mat = keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device=self.config.editor_device)
            value_diffs = value_diffs[:keys.shape[0], :]
            param_shift = torch.linalg.solve(mat, keys.T @ value_diffs)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)


            if mode != "train":
                prev = self.prev_param_shift.get(module_name, None)
                if prev is not None and prev.shape == param_shift.shape:
                    # Frobenius inner product: <A,B>_F = trace(A^T B)
                    fro_inner = torch.sum(prev * param_shift).item()

                    denom = (prev.norm() * param_shift.norm()).item()
                    cos_sim = (fro_inner / denom) if denom > 0 else 0.0
                    swanlab.log({
                        f"{module_name}/fro_inner": fro_inner,
                        f"{module_name}/cos_sim": cos_sim,
                    })
                self.prev_param_shift[module_name] = param_shift.detach().clone()


            swanlab.log({
                f"{module_name}/values_grad/mean": values_grad.mean().item(),
                f"{module_name}/values_grad/std": values_grad.std().item(),

                f"{module_name}/coeffs/mean": coeffs.mean().item(),
                f"{module_name}/coeffs/std": coeffs.std().item(),
                f"{module_name}/coeffs/min": coeffs.min().item(),
                f"{module_name}/coeffs/max": coeffs.max().item(),

                f"{module_name}/param_shift/norm": param_shift.norm().item(),
                f"{module_name}/param_shift/max": param_shift.abs().max().item()
            })
            
        return param_shifts
        
        
    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor], update: bool):
        
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            keys = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}_{self.time}/{module_idx}_{idx}_keys.pth")
                for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size))
            ])
            values_grad = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}_{self.time}/{module_idx}_{idx}_values_grad.pth")
                for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size))
            ])
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32).to(self.config.editor_device)
            param_shift = param_shifts[module_name].to(self.config.editor_device)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            with torch.no_grad():
                mat = torch.linalg.solve(keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.editor_device), module_grad)
                lamda_grad = - net.lamda(layer_idx).exp() * (mat * param_shift).sum()
            value_diffs_grad = keys @ mat
            (lamda_grad * net.lamda(layer_idx)).backward()
            for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                keys_once = pad_tensor(keys[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
                (pesudo_keys, pesudo_values_grad) = net(
                    keys_once,
                    values_grad_once,
                    layer_idx,
                )
                coeffs = - net.lr(layer_idx) * (keys_once * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diff = coeffs * pesudo_values_grad
                value_diff = value_diff[:keys.shape[0] - start_idx, :]

                (value_diffs_grad[start_idx:end_idx] * value_diff).sum().backward()
            
        clip_grad_norm_(
            self.net.parameters(),
            self.config.editor.max_grad_norm
        )

        if update == True:
            self.opt.step()
            self.opt.zero_grad()



    def run(self, train_loader: DataLoader, valid_loader: DataLoader, fact_ds_loader=None):
        """
        Use RLEdit to complete sequential editing task.
        """
 
        if not self.config.editor.load_checkpoint:
            for _ in tqdm(range(self.config.editor.n_epochs), desc = "epoch"):
                self.train(train_loader)
                self.reset_model()

                empty_cache(self.config.editor.cache_dir, self.config, self.time)
        

        self.prev_param_shift = {} 
        self.sequential_valid(valid_loader)

        empty_cache(self.config.editor.cache_dir, self.config, self.time)
        self.reset_hypernet()
        self.reset_model()