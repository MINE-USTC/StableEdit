## Dataset Preparation
We use the same raw datasets as ULTRAEDIT.
## 🚀 Setup

Create the environment and install dependencies:

```bash
conda create -n stableedit python=3.10
conda activate stableedit
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
## 🧪 Run

Run the main experiment with:

```bash
sh run.sh
```

The `run.sh` script includes a sample command like:

```python
python main.py dataset=zsre model=mistral-7b editor=stableedit num_seq=200 \
    num_seq_zsre=20 \
    editor.cache_dir=cache \
    editor.lr=1e-6 \
    editor.alpha=10 \
    editor.RunningMeanStd_mode=stable\
    editor.preheat_mode=start\
    editor.batch_size=512\
    dataset.batch_size=10 \
    dataset.n_edits=100 \
    model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj, model.layers.31.mlp.down_proj]" \
```
