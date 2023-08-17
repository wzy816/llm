# llm

A minimal implementation of decoder-only transformer in pytorch.

## Features

- model architecture sim. to llama with RMS norm and original RoPe (not rotate half);
- train on single GPU 80GB with automatic mixed precision (amp) and gradient checkpoint(;
- no any kind of parallelism for simplicity.

## Prepare Env

```bash
conda create -n llm python=3.10
conda activate llm
cd llm
pip install -r requirements.txt
```

## Train on Toy Reverse Dataset

Train a model to reserve a string. See [notebook](toyreverse.ipynb)

![loss](images/llm_toyreverse_loss.png)
![result](images/llm_toyreverse_result.png)

## Train on Scifi Novel Dataset

Download and unzip [scifi](https://huggingface.co/datasets/wzy816/scifi) to data/scifi.

```bash
# tokenize
python3 -m llm.tokenizer --data_dir=/mnt/llm/data/scifi --model_prefix=/mnt/llm/tokenizer/scifi_8000 --vocab_size=8000

# prepare train dataset
python3 -m llm.dataset --data_dir=/mnt/llm/data/scifi --tokenizer_model_file=/mnt/llm/tokenizer/scifi_8000.model --context_size=1024

# train
python3 -m llm.train --project=llm_scifi --data_dir=/mnt/llm/data/scifi --tokenizer_model_file=/mnt/llm/tokenizer/scifi_8000.model --output_dir=/mnt/llm_scifi

# train from checkpoints
python3 -m llm.train --project=llm_scifi --data_dir=/mnt/llm/data/scifi --tokenizer_model_file=/mnt/llm/tokenizer/scifi_8000.model --checkpoint_dir=/mnt/llm_scifi/step={}_loss={} --output_dir=/mnt/llm_scifi --init_step={}

# inference
python3 -m llm.inference --checkpoint_dir=/mnt/llm_scifi/step={}_loss={} --tokenizer_model_file=/mnt/llm/tokenizer/scifi_8000.model --prompt='在这道光和声音里，天和地分开了。' --max_new_tokens=16

```

## Reference

- [facebook llama](https://github.com/facebookresearch/llama/blob/main/llama/model.py)
- [huggingface llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [mingpt](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)
- [llama2.c](https://github.com/karpathy/llama2.c/blob/master/model.py)
- [Chinchilla paper](https://arxiv.org/pdf/2203.15556.pdf)
