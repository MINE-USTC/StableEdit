

from omegaconf import DictConfig
import torch
import transformers
from util import get_module
import logging
log = logging.getLogger(__name__)
from hessian_dominant_subspace import prepare_lora_model


def make_model(config: DictConfig, for_lora: bool = False):
    
    if config.class_name == "AutoModelForFEVER":
    
        model = AutoModelForFEVER(config.name_or_path)
        model.load_state_dict(torch.load(config.weight_path))
    
    else:
        model_class = getattr(transformers, config.class_name)
      
        model = model_class.from_pretrained(config.name_or_path, torch_dtype= torch.bfloat16 if config.half==True else torch.float32)
        if "qwen" in config.name.lower():
            model.config._attn_implementation = "eager"

 
    if for_lora:
        model = prepare_lora_model(
            model,
            edit_modules=config.edit_modules,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )


    for param in model.parameters():
     
        param.requires_grad = False
        
    for module_name in config.edit_modules:
       
        
        log.info(f"Enabling gradient for module: {module_name}")
        # print(model)
        module = get_module(model, module_name)
        if for_lora:
            module.base_layer.weight.requires_grad = True
            log.info(f"{module.base_layer}")
        else:
            module.weight.requires_grad = True
  
    return model