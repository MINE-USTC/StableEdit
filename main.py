import hydra
from omegaconf import DictConfig, OmegaConf
import importlib
from data.base import make_loader
from data.knowns import KnownsDataset
from model import make_model

import swanlab
import random
import numpy as np
import torch
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
 
    log = logging.getLogger(__name__)
    log.info(config)

    swanlab.init(
        project = f"{config.dataset.name}_{config.model.name}",
        experiment_name = f"{config.editor.name}_{str(config.dataset.n_edits)}",
        config = OmegaConf.to_container(config, resolve = True),
        mode="offline"          
    )
    set_seed(42)

    config.dataset.num_seq=config.num_seq
   
    data_module = importlib.import_module(f"data.{config.dataset.name}")
    data_class = getattr(data_module, f"{config.dataset.name.upper()}Dataset")
   

    train_loader, valid_loader = make_loader(config, data_class)

  
    model = make_model(config.model).to(config.model_device)
        
    editor_module = importlib.import_module(f"editor.{config.editor.name}")
    editor_class = getattr(editor_module, config.editor.name.upper())
    editor = editor_class(config, model)

    editor.run(train_loader, valid_loader)


if __name__ == "__main__":
    main()