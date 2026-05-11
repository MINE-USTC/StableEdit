#!/bin/bash

#SBATCH --partition=debug 

#SBATCH --nodes=1

#SBATCH --gres=gpu:1 

# unset CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=3


######################################  llama-3-instruct  ###################################################
# python main.py dataset=zsre model=llama-3-instruct editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


# python main.py dataset=fever model=llama-3-instruct editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512 \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.24.mlp.down_proj, model.layers.25.mlp.down_proj, model.layers.26.mlp.down_proj]" \


# python main.py dataset=wikibigedit model=llama-3-instruct editor=stableedit num_seq=170 \
#     num_seq_wikibigedit=20 \
#     editor.cache_dir=cache \
#     downstream_eval_steps=0 \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


# python main.py dataset=ultraeditbench model=llama-3-instruct editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \

# python main.py dataset=wikibigedit model=llama-3-instruct editor=stableedit num_seq=5000 \
#     num_seq_wikibigedit=170 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     dataset.eval_mhop=False \
#     dataset.valid_path=./data/raw/wikibigedit/wikibigedit.json\
#     model.edit_modules="[model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \

# python main.py dataset=ultraeditbench model=llama-3-instruct editor=stableedit num_seq=20000 \
#     num_seq_zsre=200 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


######################################  mistral-7b  ###################################################

python main.py dataset=zsre model=mistral-7b editor=stableedit num_seq=200 \
    num_seq_zsre=20 \
    downstream_eval_steps=0 \
    editor.cache_dir=cache \
    editor.lr=1e-6 \
    editor.alpha=10 \
    editor.RunningMeanStd_mode=stable\
    editor.preheat_mode=start\
    editor.batch_size=512\
    dataset.batch_size=10 \
    dataset.n_edits=100 \
    model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj, model.layers.31.mlp.down_proj]" \


# python main.py dataset=fever model=mistral-7b editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.lr=1e-6 \
#     editor.alpha=10 \
#     editor.RunningMeanStd_mode=stable\
#     editor.preheat_mode=start\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


# python main.py dataset=wikibigedit model=mistral-7b editor=stableedit num_seq=170 \
#     num_seq_wikibigedit=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.lr=1e-6 \
#     editor.alpha=10 \
#     editor.RunningMeanStd_mode=stable\
#     editor.preheat_mode=start\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


# python main.py dataset=ultraeditbench model=mistral-7b editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.lr=1e-6 \
#     editor.alpha=10 \
#     editor.RunningMeanStd_mode=stable\
#     editor.preheat_mode=start\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \

# python main.py dataset=wikibigedit model=mistral-7b editor=stableedit num_seq=5000 \
#     num_seq_wikibigedit=170 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.lr=1e-6 \
#     editor.alpha=10 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     dataset.eval_mhop=False \
#     dataset.valid_path=./data/raw/wikibigedit/wikibigedit.json\
#     model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


# python main.py dataset=ultraeditbench model=mistral-7b editor=stableedit num_seq=20000 \
#     num_seq_zsre=200 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.lr=1e-6 \
#     editor.alpha=10 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.28.mlp.down_proj, model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


######################################  gpt-j  ###################################################
# python main.py dataset=zsre model=gpt-j editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.7.mlp.fc_out, transformer.h.8.mlp.fc_out, transformer.h.9.mlp.fc_out, transformer.h.10.mlp.fc_out, transformer.h.11.mlp.fc_out, transformer.h.12.mlp.fc_out, transformer.h.13.mlp.fc_out, transformer.h.14.mlp.fc_out, transformer.h.15.mlp.fc_out, transformer.h.18.mlp.fc_out, transformer.h.19.mlp.fc_out, transformer.h.20.mlp.fc_out, transformer.h.21.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out, transformer.h.25.mlp.fc_out]" \


# python main.py dataset=fever model=gpt-j editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.8.mlp.fc_out, transformer.h.9.mlp.fc_out, transformer.h.10.mlp.fc_out, transformer.h.11.mlp.fc_out, transformer.h.12.mlp.fc_out, transformer.h.13.mlp.fc_out, transformer.h.14.mlp.fc_out, transformer.h.18.mlp.fc_out, transformer.h.19.mlp.fc_out, transformer.h.20.mlp.fc_out, transformer.h.21.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out, transformer.h.25.mlp.fc_out, transformer.h.26.mlp.fc_out]" \


# python main.py dataset=wikibigedit model=gpt-j editor=stableedit num_seq=170 \
#     num_seq_wikibigedit=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.8.mlp.fc_out, transformer.h.9.mlp.fc_out, transformer.h.10.mlp.fc_out, transformer.h.11.mlp.fc_out, transformer.h.12.mlp.fc_out, transformer.h.13.mlp.fc_out, transformer.h.14.mlp.fc_out, transformer.h.15.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out]" \


# python main.py dataset=ultraeditbench model=gpt-j editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.8.mlp.fc_out, transformer.h.9.mlp.fc_out, transformer.h.10.mlp.fc_out, transformer.h.11.mlp.fc_out, transformer.h.12.mlp.fc_out, transformer.h.13.mlp.fc_out, transformer.h.14.mlp.fc_out, transformer.h.15.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out]" \


# python main.py dataset=wikibigedit model=gpt-j editor=stableedit num_seq=5000 \
#     num_seq_wikibigedit=170 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     dataset.eval_mhop=False \
#     dataset.valid_path=./data/raw/wikibigedit/wikibigedit.json\
#     model.edit_modules="[transformer.h.8.mlp.fc_out, transformer.h.9.mlp.fc_out, transformer.h.10.mlp.fc_out, transformer.h.11.mlp.fc_out, transformer.h.12.mlp.fc_out, transformer.h.13.mlp.fc_out, transformer.h.14.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out]" \



# python main.py dataset=ultraeditbench model=gpt-j editor=stableedit num_seq=20000 \
#     num_seq_zsre=200 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.8.mlp.fc_out, transformer.h.9.mlp.fc_out, transformer.h.10.mlp.fc_out, transformer.h.11.mlp.fc_out, transformer.h.12.mlp.fc_out, transformer.h.13.mlp.fc_out, transformer.h.14.mlp.fc_out, transformer.h.15.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out]" \


# python main.py dataset=zsre model=qwen2.5-7b editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable \
#     editor.batch_size=512 \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.8.mlp.down_proj, model.layers.9.mlp.down_proj, model.layers.10.mlp.down_proj, model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.13.mlp.down_proj, model.layers.14.mlp.down_proj, model.layers.18.mlp.down_proj, model.layers.19.mlp.down_proj, model.layers.20.mlp.down_proj, model.layers.21.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.24.mlp.down_proj, model.layers.25.mlp.down_proj]" \


# python main.py dataset=fever model=qwen2.5-7b editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.9.mlp.down_proj, model.layers.10.mlp.down_proj, model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.13.mlp.down_proj, model.layers.14.mlp.down_proj, model.layers.15.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.24.mlp.down_proj, model.layers.25.mlp.down_proj, model.layers.26.mlp.down_proj]" \


# python main.py dataset=wikibigedit model=qwen2.5-7b editor=stableedit num_seq=170 \
#     num_seq_wikibigedit=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stale\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.8.mlp.down_proj, model.layers.9.mlp.down_proj, model.layers.10.mlp.down_proj, model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.13.mlp.down_proj, model.layers.14.mlp.down_proj, model.layers.18.mlp.down_proj, model.layers.19.mlp.down_proj, model.layers.20.mlp.down_proj, model.layers.21.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.24.mlp.down_proj]" \


# python main.py dataset=ultraeditbench model=qwen2.5-7b editor=stableedit num_seq=200 \
#     num_seq_zsre=20 \
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=2e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.8.mlp.down_proj, model.layers.9.mlp.down_proj, model.layers.10.mlp.down_proj, model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.13.mlp.down_proj, model.layers.14.mlp.down_proj, model.layers.18.mlp.down_proj, model.layers.19.mlp.down_proj, model.layers.20.mlp.down_proj, model.layers.21.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.24.mlp.down_proj]" \


# python main.py dataset=wikibigedit model=qwen2.5-7b editor=stableedit num_seq=5000 \
#     num_seq_wikibigedit=170\
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     dataset.eval_mhop=False \
#     dataset.valid_path=./data/raw/wikibigedit/wikibigedit.json\
#     model.edit_modules="[model.layers.8.mlp.down_proj, model.layers.9.mlp.down_proj, model.layers.10.mlp.down_proj, model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.13.mlp.down_proj, model.layers.14.mlp.down_proj, model.layers.18.mlp.down_proj, model.layers.19.mlp.down_proj, model.layers.20.mlp.down_proj, model.layers.21.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.24.mlp.down_proj]" \


# python main.py dataset=ultraeditbench model=qwen2.5-7b editor=stableedit num_seq=20000 \
#     num_seq_zsre=200\
#     downstream_eval_steps=0 \
#     editor.cache_dir=cache \
#     editor.alpha=10 \
#     editor.lr=1e-6 \
#     editor.RunningMeanStd_mode=stable\
#     editor.batch_size=512\
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.8.mlp.down_proj, model.layers.9.mlp.down_proj, model.layers.10.mlp.down_proj, model.layers.11.mlp.down_proj, model.layers.12.mlp.down_proj, model.layers.13.mlp.down_proj, model.layers.14.mlp.down_proj, model.layers.18.mlp.down_proj, model.layers.19.mlp.down_proj, model.layers.20.mlp.down_proj, model.layers.21.mlp.down_proj, model.layers.22.mlp.down_proj, model.layers.23.mlp.down_proj, model.layers.24.mlp.down_proj]" \

