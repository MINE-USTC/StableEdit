from typing import Dict, List
from omegaconf import DictConfig

from collections import Counter
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import islice

from tqdm import tqdm

import swanlab

from transformers import AutoTokenizer, LlamaTokenizerFast

import json
from datetime import datetime
import copy

from glue_eval.glue_eval import GLUEEval
from model import make_model
from util import (
    get_module,
    get_shape,
    empty_cache,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios,
    dict_to_
)

import logging
log = logging.getLogger(__name__)


class BaseEditor:

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        
        self.config = config
        self.model = model
        
        shape_counter = Counter()
        
        self.name2idx = {}
        for module_name in config.model.edit_modules:
            module = get_module(model, module_name)
            
            shape = get_shape(module.base_layer) if hasattr(module, 'base_layer') else get_shape(module)
            
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1
        
        self.shape_counter = shape_counter

        self.tuples_list = []

        self.tok = AutoTokenizer.from_pretrained(config.model.name_or_path)

        if self.tok.pad_token is None:

            print(config.model.name_or_path, "does not have a pad token, setting it to eos_token")
            self.tok.pad_token = self.tok.eos_token


    def reset_hypernet_normalizers(self):
        """
        self.net[str(shape)] -> RLEditNet(normalizer, blocks, lr, lamda)
        """
        if not hasattr(self, "net"):
            return

        def _reset_norm(norm):
            
            for attr in ("n", "mean", "var", "std", "cov", "cov_key", "cov_value"):
                if hasattr(norm, attr):
                    buf = getattr(norm, attr)
                    if torch.is_tensor(buf):
                        buf.zero_()
        
        for key, sub in self.net.items():
            if hasattr(sub, "normalizer"):
                _reset_norm(sub.normalizer)

        for m in self.net.modules():
            if m is not self.net and hasattr(m, "normalizer"):
                _reset_norm(m.normalizer)



    def decode_pred_and_label(self, t, logits):
        tok = self.tok
        labels = t["labels"]            # [B, L]
        pred_ids = logits.argmax(-1)    # [B, L]

        out = []
        B = labels.size(0)
        for b in range(B):
            mask = labels[b] != -100
            true_ids = labels[b][mask]         
            pred_ids_b = pred_ids[b][mask]      

            all_match = bool(torch.equal(pred_ids_b, true_ids))

            true_text = tok.decode(true_ids.tolist(),
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
            pred_text = tok.decode(pred_ids_b.tolist(),
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
            out.append((pred_text, true_text, all_match))
        return out
    

    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool,
        is_projected: bool = False
        ):
        for idx, (module_name, param_shift) in enumerate(param_shifts.items()):

            module = get_module(self.model, module_name)
           
            if self.config.for_lora == True:
                if isinstance(module.base_layer, nn.Linear):
                    param_shift = param_shift.T
            else:
                if isinstance(module, nn.Linear):
                    param_shift = param_shift.T

            if is_reverse:
                param_shift = - param_shift

            module.weight.data += param_shift.to(module.weight.data.dtype)

    def reset_model(self):
            del self.model
            
            torch.cuda.empty_cache()
            self.model = make_model(self.config.model, for_lora=self.config.for_lora).to(self.config.model_device)
    
   
    def cache(self, tuples: List[Dict[str, torch.LongTensor]], mode=None):

        for idx, t in enumerate(tuples):

            with TracerDict(
                self.model,
                self.config,
                t
            ) as tr:
                logits = self.model(**t)["logits"] 
                cross_entropy(logits, t["labels"]).backward() 
          
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                 
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1), mode=mode)
                dir_path = f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}_{self.time}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path,exist_ok=True)
                torch.save(keys, f"{dir_path}/{module_idx}_{idx}_keys.pth")
                torch.save(values_grad, f"{dir_path}/{module_idx}_{idx}_values_grad.pth")


    def train(self, loader: DataLoader, save=False):
        
        max_steps = self.config.num_seq

        train_max_steps = min(len(loader), max_steps)
        limited_loader = list(islice(loader, train_max_steps))

        for _, tuples in enumerate(tqdm(limited_loader, desc = "Train", ncols = 100, total=max_steps)):

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.model.zero_grad()

            gen_losses = []

            self.edit_model(param_shifts, False)
            

            for t in tuples["equiv_tuples"]:
                logits = self.model(**t)["logits"]
                loss = cross_entropy(logits, t["labels"])
                loss.backward()
                gen_losses += [loss.item()]
            self.edit_model(param_shifts, True)

            loc_losses = []
            for t in tuples["unrel_tuples"]:


                with torch.no_grad():
                    refer_logits = self.model(**t)["logits"]

                self.edit_model(param_shifts, False)
                logits = self.model(**t)["logits"]

                loss = kl_div(
                    refer_logits,
                    logits,
                    t["labels"]
                )
                (self.config.editor.loc_coef * loss).backward()
                self.edit_model(param_shifts, True)
                loc_losses += [loss.item()]
            
           
            self.update_hypernet(param_shifts, update=True)

            swanlab.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })
        
        if self.config.editor.save_checkpoint:
            hypernet_dir = f"saved_hypernets/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}"
            os.makedirs(hypernet_dir, exist_ok=True)
    
            torch.save(
                {
                    "config": dict(self.config),
                    "net": self.net.state_dict(),
                    "opt": self.opt.state_dict(),
                },
                f"{hypernet_dir}/hypernet.pth",
            )
            print(f"Hypernet saved to {hypernet_dir}")


    def sequential_valid(self, loader: DataLoader, train_loader: DataLoader = None, is_projected: bool = False, beta=1.0, half=100):


        max_steps = self.config.num_seq
        preheat_mode = getattr(self.config.editor, "preheat_mode", "start")  # start|middle|end|random|every_n|indices|none
       
        if train_loader is not None:
            if self.config.dataset.name in ["zsre", "fever", "ultraeditbench"]:
                train_max_steps = self.config.num_seq_zsre
            else:
                self.config.dataset.name in ["wikibigedit"]:
                train_max_steps = self.config.num_seq_wikibigedit

            
            train_max_steps = min(len(train_loader), train_max_steps)
           
            trainloader = list(islice(train_loader, train_max_steps))
  
            preheat_indices = set()
            if preheat_mode == "start":
                preheat_indices.add(0)
            elif preheat_mode == "middle":
                center = max_steps // 2
                preheat_indices.add(center)
            elif preheat_mode == "end":
                preheat_indices.add(max_steps - 1)
            elif preheat_mode == "q1":  
                preheat_indices.add(max_steps // 4)
            elif preheat_mode == "q3": 
                preheat_indices.add(3 * max_steps // 4)
            else:
                preheat_indices = set()
        else:
            preheat_indices = set()

        limited_loader = islice(loader, max_steps)
        log.info(f"Using {len(loader)} samples for sequential editing, max_steps={max_steps}.")
        step = 0
        if self.config.downstream_eval_steps > 0:
            log.info(f"Downstream evaluation every {self.config.downstream_eval_steps} steps.")
            glue_save_location = f'results/glue_eval/{self.config.editor.name}_{self.config.model.name}_{self.config.dataset.name}_{self.config.num_seq}_{self.time}/'
            os.makedirs(glue_save_location, exist_ok=True)
            
        for j, tuples in enumerate(tqdm(limited_loader, desc="Editing Time", ncols=100, total=max_steps)):
            
            if self.config.downstream_eval_steps > 0 and step == 0:
                glue_results = {'edit_num': -1}

                out_file = glue_save_location + "base.json"
                
                glue_eval = GLUEEval(self.model, self.tok, number_of_tests = 100)
                glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

                # store the individual overall result file
                output_filename = out_file.replace('.json', '_glue.json')
                with open(output_filename, "w") as f:
                    json.dump(glue_results, f, indent=4)
            
            if j in preheat_indices:
                log.info(f"Preheating mu and sigma at step {j} using {len(trainloader)} training samples.")
                for _, train_tuples in enumerate(tqdm(trainloader, desc="Searching for better start point for mu and sigma ", ncols=100, total=train_max_steps)):
                    self.cache(train_tuples["edit_tuples"], mode='train')
                    param_shifts = self.predict_param_shifts(mode="train")
                    self.edit_model(param_shifts, is_reverse=False, is_projected=False)
                
            self.cache(tuples["edit_tuples"], mode="inference") 
            param_shifts = self.predict_param_shifts()
            self.edit_model(param_shifts, is_reverse=False)
           
            self.tuples_list.append(tuples)
            if self.config.editor.name != "stableedit":
                self.opt.zero_grad()
        
            step = j + 1

            if self.config.downstream_eval_steps > 0 and step % self.config.downstream_eval_steps == 0:
                glue_results = {
                        'edit_num': step * self.config.dataset.n_edits,
                        }
                out_file = glue_save_location + "case_{}.json".format(step * self.config.dataset.n_edits) # stores the last case ID of the batch
    
                glue_eval = GLUEEval(self.model, self.tok, number_of_tests = 100)
                glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                        
                # store the individual overall result file
                output_filename = out_file.replace('.json', '_glue.json')
                with open(output_filename, "w") as f:
                    json.dump(glue_results, f, indent=4)

            
        self.model.eval()
        edit_succs, gen_succs, loc_succs = [], [], []
       
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in tqdm(self.tuples_list, desc=f"Eval time of {k}", ncols=100, total=len(self.tuples_list)):
                for t in tuple[k]:
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                        
                    s += succ_ratios(logits, t["labels"])
                    
        edit_succs = np.array(edit_succs)
        gen_succs = np.array(gen_succs)
        loc_succs = np.array(loc_succs)
        

        if self.config.model_cache==True:
            if not os.path.exists(self.config.model_cache_dir):
                os.makedirs(self.config.model_cache_dir)
            self.model.save_pretrained(self.config.model_cache_dir)
            self.tok.save_pretrained(self.config.model_cache_dir)
        
        if self.config.dataset.name=="wikibigedit":
            person_succs=[]
            mhop_succs=[]
            if self.config.dataset.eval_mhop==True:
                for k, s in zip(
                    ["person_tuples", "mhop_tuples"],
                    [person_succs, mhop_succs]
                ):
                    for tuple in tqdm(self.tuples_list, desc=f"Eval time of {k} ", ncols=100,total=len(self.tuples_list)):
                        for t in tuple[k]:
                            # import ipdb;ipdb.set_trace()
                            with torch.no_grad():
                                logits = self.model(**t)["logits"]
                            s += succ_ratios(logits, t["labels"])
            else:
                for k, s in zip(
                    ["person_tuples"],
                    [person_succs]
                ):
                    for tuple in tqdm(self.tuples_list, desc=f"Eval time of {k}", ncols=100,total=len(self.tuples_list)):
                        for t in tuple[k]:
                            # import ipdb;ipdb.set_trace()
                            with torch.no_grad():
                                logits = self.model(**t)["logits"]
                            s += succ_ratios(logits, t["labels"])

        final_results = {
            "ES": np.mean(edit_succs),
            "GS": np.mean(gen_succs),
            "LS": np.mean(loc_succs),
        }   
        print(final_results)   
       
        if self.config.dataset.name=="wikibigedit":
            final_results["person_score"]=np.mean(person_succs)
            if self.config.dataset.eval_mhop==True:
                final_results["mhop_score"]=np.mean(mhop_succs)

        log.info("final_results_sequential:")
        log.info(final_results)

        swanlab.log(final_results)

        self.model.train()

    def run(self, train_loader: DataLoader, valid_loader: DataLoader, fact_ds_loader=None):

        if not self.config.editor.load_checkpoint:
            for _ in range(self.config.editor.n_epochs):
                self.train(train_loader)
                self.reset_model()
                empty_cache(self.config.editor.cache_dir, self.config, self.time)
                
        self.sequential_valid(valid_loader)
        empty_cache(self.config.editor.cache_dir, self.config, self.time)
        self.reset_hypernet()
        self.reset_model()





