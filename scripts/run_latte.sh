# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m wikitext2 > res/bf16

export CUDA_VISIBLE_DEVICES=0 
python latte_quant.py
