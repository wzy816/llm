import os
import re
from pathlib import Path

import click
import sentencepiece as spm


@click.command()
@click.option('data_dir', '--data_dir', required=True)
@click.option('model_prefix', '--model_prefix', required=True)
@click.option('model_type', '--model_type', required=False, default='bpe')
@click.option('vocab_size', '--vocab_size', required=False, default=8000)
@click.option('max_sentence_length', '--max_sentence_length', required=False, default=4192)
@click.option('input_sentence_size', '--input_sentence_size', required=False, default=0)
@click.option('shuffle_input_sentence', '--shuffle_input_sentence', required=False, default=False)
def main(data_dir, model_prefix, model_type, vocab_size, max_sentence_length, input_sentence_size, shuffle_input_sentence):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    def iter(directory):
        filenames = list(Path(directory).rglob('*.txt'))
        for fn in filenames:
            with open(os.path.join(directory, fn), 'r') as f:
                for line in f.readlines():
                    line = line.strip()

                    if len(line.encode()) > max_sentence_length:
                        parts = re.split("([.?!。？！])", line)
                        merged = []
                        for i in range(0, len(parts), 2):
                            p = parts[i]
                            if i+1 < len(parts):
                                p = p+parts[i+1]
                            if len(p.encode()) <= max_sentence_length:
                                merged.append(p)
                        if len(merged) > 0:
                            l = ''
                            for m in merged:
                                if len((l+m).encode()) > max_sentence_length:
                                    yield l.encode()
                                    l = ''
                                l = l + m
                            yield l.encode()

                    else:
                        yield line.encode()

    spm.SentencePieceTrainer.train(sentence_iterator=iter(data_dir),
                                   model_prefix=model_prefix,
                                   model_type=model_type,
                                   vocab_size=vocab_size,
                                   pad_id=0,
                                   unk_id=1,
                                   bos_id=2,
                                   eos_id=3,
                                   max_sentence_length=max_sentence_length,
                                   input_sentence_size=input_sentence_size,
                                   shuffle_input_sentence=shuffle_input_sentence,
                                   user_defined_symbols=[])


if __name__ == '__main__':
    main()
