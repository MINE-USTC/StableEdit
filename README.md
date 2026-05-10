# StableEdit

Official code for the ICML 2026 paper **More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing**.

## Overview

Sequential model editing must continuously inject new facts while preserving previously edited knowledge and general model behavior. In long editing horizons, existing methods often suffer from catastrophic forgetting and model collapse.

This project studies the mechanism behind long-horizon stability and introduces **StableEdit**, a method built on two main observations:

- Lifelong editors that remain stable over many editing steps share a common normalization mechanism, which we call **Lifelong Normalization (LN)**.
- Early edits can have a **positive cumulative effect**, improving the stability of later edits instead of hurting them.

Based on the theoretical analysis in the paper, StableEdit strengthens lifelong stability through:

- an explicit warm-up stage for running statistics
- full whitening of editing features
- ridge-regularized updates with bounded norms and improved orthogonality

## Highlights

- A theoretical account of why LN stabilizes lifelong sequential editing.
- A practical editor, `stableedit`, with minimal overhead over existing pipelines.
- Support for multiple editors: `stableedit`, `rledit`, `malmen`, and `mend`.
- Support for multiple datasets: `zsre`, `fever`, `wikibigedit`, and `ultraeditbench`.

## Repository Structure

```text
.
├── config/              # Hydra configs for datasets, models, and editors
├── data/                # Dataset loaders and raw data directory
├── editor/              # Editing algorithms
├── glue_eval/           # Downstream evaluation scripts and subsets
├── main.py              # Main entry point
├── model.py             # Model loading utilities
├── nets.py              # Core network components
├── run.sh               # Example launch script
├── requirements.txt
└── README.md
```

## Environment Setup

We recommend Python 3.10 and PyTorch 2.3.0.

```bash
conda create -n stableedit python=3.10
conda activate stableedit
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Installed Python dependencies include:

- `hydra-core`
- `transformers`
- `datasets`
- `scikit-learn`
- `nltk`
- `sentencepiece`
- `protobuf`

## Data Preparation

Raw data should be placed under `data/raw/` with the following layout:

```text
data/raw/
├── known_1000.json
├── fever/
│   ├── fever_train.json
│   ├── fever_eval_20k.json
│   └── fever_eval_100k.json
├── ultraeditbench/
│   ├── UltraEditBench_train_20k.json
│   └── UltraEditBench_2M.json
├── wikibigedit/
│   ├── wikibigedit_train_17k.json
│   ├── wikibigedit_eval_17k.json
│   └── wikibigedit.json
└── zsre/
    ├── zsre_train.json
    ├── zsre_eval_20k.json
    └── zsre_eval_100k.json
```

Because several JSON files are large, this repository tracks them with **Git LFS**:

```bash
git lfs install
git lfs track "data/raw/**/*.json"
git add .gitattributes
```

If your data already exists in another local folder, you can copy it into the repository with commands like:

```bash
mkdir -p data/raw/zsre data/raw/fever data/raw/wikibigedit data/raw/ultraeditbench
cp /path/to/raw/known_1000.json data/raw/
cp /path/to/raw/zsre/*.json data/raw/zsre/
cp /path/to/raw/fever/*.json data/raw/fever/
cp /path/to/raw/wikibigedit/*.json data/raw/wikibigedit/
cp /path/to/raw/ultraeditbench/*.json data/raw/ultraeditbench/
```

## Quick Start

The simplest way to launch an experiment is:

```bash
sh run.sh
```

The current example in `run.sh` launches StableEdit on `zsre` with `mistral-7b`. An equivalent direct command is:

```bash
python main.py dataset=zsre model=mistral-7b editor=stableedit num_seq=200 \
    num_seq_zsre=20 \
    editor.cache_dir=cache \
    editor.lr=1e-6 \
    editor.alpha=10 \
    editor.RunningMeanStd_mode=stable \
    editor.preheat_mode=start \
    editor.batch_size=512 \
    dataset.batch_size=10 \
    dataset.n_edits=100 \
    model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj, model.layers.31.mlp.down_proj]"
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management.

- `config/config.yaml`: global defaults
- `config/dataset/*.yaml`: dataset-specific paths and sequence lengths
- `config/model/*.yaml`: base model names and editable modules
- `config/editor/*.yaml`: editor-specific hyperparameters

Common command-line arguments:

- `dataset`: dataset name, such as `zsre`, `fever`, `wikibigedit`, or `ultraeditbench`
- `model`: model preset, such as `mistral-7b`, `llama-3-instruct`, `gpt-j`, or `qwen2.5-7b`
- `editor`: editing method, such as `stableedit`, `rledit`, `malmen`, or `mend`
- `num_seq`: total number of sequential editing steps during evaluation
- `num_seq_zsre`: number of warm-up training steps used for `zsre`, `fever`, and `ultraeditbench`
- `num_seq_wikibigedit`: number of warm-up training steps used for `wikibigedit`
- `dataset.n_edits`: number of edit instances per sequential step
- `dataset.batch_size`: mini-batch size for preparing edit tuples
- `editor.batch_size`: batch size used inside the editor when solving parameter updates
- `editor.lr`: editing step size
- `editor.alpha`: ridge regularization strength in StableEdit
- `editor.RunningMeanStd_mode`: running-statistics mode for lifelong normalization
- `editor.preheat_mode`: warm-up position for statistics initialization, one of `start`, `q1`, `middle`, `q3`, `end`, or `none`
- `editor.cache_dir`: directory used to store intermediate cached keys and value gradients
- `model.edit_modules`: target modules to be edited
- `downstream_eval_steps`: optional interval for GLUE-style downstream evaluation

## Notes on `run.sh`

`run.sh` currently includes:

- sample commands for `llama-3-instruct` and `mistral-7b`
- commented examples for `zsre`, `fever`, `wikibigedit`, and `ultraeditbench`
- `SBATCH` headers for cluster usage

If you are not using Slurm, you can ignore the `SBATCH` lines and run the Python commands directly in your shell.

## Acknowledgements

This repository builds on the broader lifelong model editing literature and experimental setups. We thank [UltraEdit](https://github.com/zjunlp/EasyEdit) for inspiring parts of the evaluation and data preparation workflow.

The Markdown syntax for a hidden link is:

```md
[UltraEdit](https://github.com/zjunlp/EasyEdit)
```

## Contact

For questions about the paper or code, please contact:

- Zhi Zheng: `zhengzhi97@ustc.edu.cn`
- Tong Xu: `tongxu@ustc.edu.cn`
- Enhong Chen: `cheneh@ustc.edu.cn`
