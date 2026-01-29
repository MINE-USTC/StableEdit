from typing import Union, Tuple, List, Dict, Optional
from omegaconf import DictConfig
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from transformers.pytorch_utils import Conv1D
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def empty_cache(path: str, config, time):

    if time is None:
        dir_path = f"{config.editor.cache_dir}/{config.model.name}_{config.dataset.name}_{config.editor.name}_{config.dataset.n_edits}_{config.num_seq}"
    else:
        dir_path = f"{config.editor.cache_dir}/{config.model.name}_{config.dataset.name}_{config.editor.name}_{config.dataset.n_edits}_{config.num_seq}_{time}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    try:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error while clearing cache: {e}")


def get_module(module: nn.Module, module_name: str) -> nn.Module:

    for name in module_name.split("."):

        module = getattr(module, name)
    return module


def get_shape(module: Union[nn.Linear, Conv1D]) -> Tuple[int]:
    
    shape = tuple(module.weight.shape)

    return shape[::-1] if isinstance(module, nn.Linear) else shape
    
    
def cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
):
    if len(logits.shape) == 2:
        return F.binary_cross_entropy_with_logits(logits, labels)

    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        
        return F.cross_entropy(logits, labels)


def log(x: torch.FloatTensor) -> torch.FloatTensor:
    return (x + torch.finfo(x.dtype).eps).log()


def kl_div(
    refer_logits: torch.FloatTensor,
    logits: torch.FloatTensor,
    labels: torch.LongTensor
) -> torch.Tensor:
    
    if len(logits.shape) == 2:
        refer_probs = F.sigmoid(refer_logits)
        probs = F.sigmoid(logits)
        return (refer_probs * (log(refer_probs) - log(probs))) + ((1 - refer_probs) * (log(1 - refer_probs) - log(1 - probs)))
    
    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        refer_logits = refer_logits[ans_indice]
        logits = logits[ans_indice]
        refer_log_probs = refer_logits.log_softmax(-1)
        log_probs = logits.log_softmax(-1)
        
        return F.kl_div(
            log_probs,
            refer_log_probs,
            reduction = "batchmean",
            log_target = True
        )


def succ_ratios(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    old_labels: torch.LongTensor=None
) -> List[float]:
    
    if old_labels is None:
    
        if len(logits.shape) == 2:
            return ((logits > 0) == labels).squeeze(-1).to("cpu").numpy().tolist()
        
        if len(logits.shape) == 3:
   
            n_corr = (logits.argmax(-1) == labels).sum(-1)
            n_tokens = (labels != -100).sum(-1)
            return (n_corr / n_tokens).to("cpu").numpy().tolist()
    
    else:

        if len(logits.shape) == 2:

            if old_labels.shape[1] > labels.shape[1]:
                old_labels = old_labels[:, :labels.shape[1]]
            label_probs = logits[torch.arange(logits.size(0)), labels]
            old_label_probs = logits[torch.arange(logits.size(0)), old_labels]
            success = (label_probs > old_label_probs).to(torch.float32)

        if len(logits.shape) == 3:
            # import ipdb;ipdb.set_trace()
            batch_size, seq_len, _ = logits.shape

            if old_labels.shape[1] > labels.shape[1]:
                old_labels = old_labels[:, :labels.shape[1]]

            if labels.shape[1] > old_labels.shape[1]:
                move = labels.shape[1] - old_labels.shape[1]
                labels = labels[:, :old_labels.shape[1]]
                seq_len -= move

            valid_mask = (labels != -100) & (old_labels != -100)
            label_probs = logits[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), labels]
            old_label_probs = logits[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), old_labels]
            success = ((label_probs > old_label_probs) & valid_mask).to(torch.float32)

        n_corr = success.sum(-1)
        n_tokens = (labels != -100).sum(-1)

        return (n_corr / n_tokens).to("cpu").numpy().tolist()



class Tracer:

    def __init__(
        self,
        module: nn.Module,
        cache_mask: torch.LongTensor
    ):

        cache_indices = torch.where(cache_mask)

        
        def forward_hook(
            module: nn.Module,
            inputs: Tuple[torch.FloatTensor], # 
            outputs: Tuple[torch.FloatTensor]
        ):
            self.keys = inputs[0][cache_indices].detach()
          
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            self.values = out[cache_indices].detach()



        def backward_hook(
            module: nn.Module,
            inputs_grad: Tuple[torch.FloatTensor],
            outputs_grad: Tuple[torch.FloatTensor]
        ):
            self.values_grad = outputs_grad[0][cache_indices].detach()

        self.handles = [
            module.register_forward_hook(forward_hook),
            module.register_full_backward_hook(backward_hook)
        ]




class TracerDict(dict):
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        tuples: Dict[str, torch.LongTensor],
        fact = False    
    ):
        
        if any("encoder" in m for m in config.model.edit_modules) and any("decoder" in m for m in config.model.edit_modules):
            
            for module_name in config.model.edit_modules:
                if "encoder" in module_name:
                    cache_mask = tuples["attention_mask"]
                else:
                    cache_mask = tuples["decoder_attention_mask"]
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)

        else:

            if config.editor.token == "ans":
                cache_mask = tuples["labels"] != -100
            else:
                cache_mask = tuples["attention_mask"]

            if fact == True:
                last_layer = model.config.num_hidden_layers - 1
                if config.model.name == "EleutherAI_gpt-j-6B":
                    module_name = f"transformer.h.{last_layer}"
                else:
                    module_name = f"model.layers.{last_layer}"
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)

            else:
                for module_name in config.model.edit_modules:
                    module = get_module(model, module_name)
                    self[module_name] = Tracer(module, cache_mask)


            
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        for v in self.values():
            for h in v.handles:
                h.remove()



class FixedSubsetSampler(Sampler):
    """Represents a fixed sequence of data set indices.
    Subsets can be created by specifying a subset of output indexes.
    """

    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def subset(self, new_subset):
        return FixedSubsetSampler(self.dereference(new_subset))

    def dereference(self, indices):
        """
        Translate output sample indices (small numbers indexing the sample)
        to input sample indices (larger number indexing the original full set)
        """
        return [self.samples[i] for i in indices]


class FixedRandomSubsetSampler(FixedSubsetSampler):
    """Samples a fixed number of samples from the dataset, deterministically.
    Arguments:
        data_source,
        sample_size,
        seed (optional)
    """

    def __init__(self, data_source, start=None, end=None, seed=1):
        rng = random.Random(seed)
        shuffled = list(range(len(data_source)))
        rng.shuffle(shuffled)
        self.data_source = data_source
        super(FixedRandomSubsetSampler, self).__init__(shuffled[start:end])

    def class_subset(self, class_filter):
        """
        Returns only the subset matching the given rule.
        """
        if isinstance(class_filter, int):

            def rule(d):
                return d[1] == class_filter

        else:
            rule = class_filter
        return self.subset(
            [i for i, j in enumerate(self.samples) if rule(self.data_source[j])]
        )

def make_loader(
    dataset, sample_size=None, batch_size=1, sampler=None, random_sample=None, **kwargs
):
    """Utility for creating a dataloader on fixed sample subset."""
    import typing

    if isinstance(dataset, typing.Callable):
        # To support deferred dataset loading, support passing a factory
        # that creates the dataset when called.
        dataset = dataset()
    if isinstance(dataset, torch.Tensor):
        # The dataset can be a simple tensor.
        dataset = torch.utils.data.TensorDataset(dataset)
    if sample_size is not None:
        assert sampler is None, "sampler cannot be specified with sample_size"
        if sample_size > len(dataset):
            print(
                "Warning: sample size %d > dataset size %d"
                % (sample_size, len(dataset))
            )
            sample_size = len(dataset)
        if random_sample is None:
            sampler = FixedSubsetSampler(list(range(sample_size)))
        else:
            sampler = FixedRandomSubsetSampler(
                dataset, seed=random_sample, end=sample_size
            )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, **kwargs
    )


class TokenizedDataset(Dataset):
    """
    Converts a dataset of text samples into a dataset of token sequences,
    as converted by a supplied tokenizer. The tokens come along with position
    ids and attention masks, they can be supplied directly to the model.
    """

    def __init__(self, text_dataset, tokenizer=None, maxlen=None, field="text", for_hessian=False):
        self.text_dataset = text_dataset
        self.field = field
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.for_hessian = for_hessian
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        text = self.text_dataset[i]
        if self.field is not None:
            text = text[self.field]

        token_list = self.tokenizer.encode(
            text, truncation=True, max_length=self.maxlen
        )

    
        position_ids = list(range(len(token_list)))
        attention_mask = [1] * len(token_list)
        return dict(
            input_ids=torch.tensor(token_list),
            position_ids=torch.tensor(position_ids),
            attention_mask=torch.tensor(attention_mask),
            labels=torch.tensor(token_list) if self.for_hessian else None,
        )

def length_collation(token_size):
    """
    Sorts a batch of sequences and breaks it up into subbatches
    of same-sized sequences, padding as needed.  Each batch
    has no more than token_size total tokens (or a single
    sequence, if the sequence happens to be larger).
    """

    def collate_fn(items):

    
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        
        return batches

    return collate_fn


def make_padded_batch(items):
    """
    Pads sequences in a batch, so they are all the same length as the longest.
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}
    return {
        k: pad_sequence([d[k] for d in items if len(d["input_ids"])], batch_first=True)
        for k, v in items[0].items()
    }

def dict_to_(data, device):
    """
    Moves a dictionary of tensors to the specified device.
    """
    for k in data:
        data[k] = data[k].to(device)
    return data

def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """


    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]

    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

   
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1) # [batch_size, vocab_size]


            tk = torch.topk(softmax_out, top_k, dim=1).indices
            
           
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
                      
            new_toks = torch.gather(tk, 1, new_tok_indices)

           
            if cur_context.stop == input_ids.size(1):
               
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)


    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]

    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt




def perplexity(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0] 
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()

