import os
import random
from pathlib import Path

import click
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, context_size, data_dir, sample_every):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.npz_files = list(Path(data_dir).rglob(
            f'*.{context_size}.npz'))
        assert len(self.npz_files) > 0
        self.sample_every = sample_every

    def __iter__(self):
        while True:
            random.shuffle(self.npz_files)
            for fn in tqdm(self.npz_files):
                npzfile = np.load(fn)
                for k in npzfile.files:
                    tokens = npzfile[k]

                    if len(tokens) < self.context_size+1:
                        continue
                    else:
                        diff = len(tokens) - (self.context_size+1)
                        for i in random.sample(range(diff+1), (diff+1)//self.sample_every):
                            x = torch.from_numpy(tokens[i:i+self.context_size])
                            y = torch.from_numpy(
                                tokens[i+1:i+self.context_size+1])
                            yield x.long().cuda(), y.long().cuda()


def tokenize_txt_to_npz(data_dir, tokenizer, context_size, merge_level):
    for f in list(Path(data_dir).rglob(f'*.{context_size}.npz')):
        os.remove(f)
        
    total_tokens = 0

    files = list(Path(data_dir).rglob('*.txt'))
    for fn in tqdm(files):
        
        # merge adjacent lines to force array length >= ctx+1
        if merge_level == 'line':
            out = {}
            last_i = None
            with open(fn, 'r') as f:
                tokens = []
                last_i = None
                for i, line in enumerate(f.readlines()):
                    tokens += tokenizer.encode(line)
                    if len(tokens) >= context_size+1:
                        out[str(i)] = tokens
                        last_i = i
                        tokens = []
            if len(tokens) > 0 and last_i is not None:
                out[str(last_i)] += tokens

            for k in out:
                total_tokens += len(out[k])
                
            if out:
                outfile = Path(fn).with_suffix(f'.{context_size}.npz')
                np.savez(outfile, **out)
                
        # no merge, skip line if array length < ctx+1
        elif merge_level == 'none':
            out = {}
            with open(fn, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    tokens = tokenizer.encode(line)
                    if len(tokens) >= context_size+1:
                        out[str(i)] = tokens
            for k in out:
                total_tokens += len(out[k])
            if out:
                outfile = Path(fn).with_suffix(f'.{context_size}.npz')
                np.savez(outfile, **out)
                
        # merge lines in entire files, skip file if array length < ctx+1
        elif merge_level == 'file':
            tokens = []
            with open(fn, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    tokens += tokenizer.encode(line)
                    
            if len(tokens) >= context_size+1:
                total_tokens += len(tokens)
                outfile = Path(fn).with_suffix(f'.{context_size}.npz')
                np.savez(outfile, tokens)
                
    print(f'total tokens: {total_tokens/1e6:.2f}M')


@click.command()
@click.option('data_dir', '--data_dir', required=True)
@click.option('tokenizer_model_file', '--tokenizer_model_file', required=True)
@click.option('context_size', '--context_size', required=True, type=int)
@click.option('merge_level', '--merge_level', required=False, type=click.Choice(['none', 'line', 'file']), default='line')
def main(data_dir, tokenizer_model_file, context_size, merge_level):
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
    tokenize_txt_to_npz(data_dir, tokenizer, context_size, merge_level)


if __name__ == '__main__':
    main()
