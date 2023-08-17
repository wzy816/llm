import json

import click
import sentencepiece as spm
import torch
import torch.nn.functional as F

from llm.model import LLM
from llm.train import Config


def sample_top_p(probs, top_p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate(model, tokenizer, prompt, context_size, max_new_tokens, temperature, top_p):
    prompt_tokens = tokenizer.encode(prompt)
    tokens = torch.full((1, len(prompt_tokens)+max_new_tokens),
                        tokenizer.pad_id()).long().cuda()
    tokens[0, :len(prompt_tokens)] = torch.LongTensor(prompt_tokens).cuda()

    for i in range(max_new_tokens):
        end = i+len(prompt_tokens)
        start = max(0, end - context_size)
        x = tokens[:, start:end]
        logits = model.forward(x)
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
        tokens[0, end] = next_token

    return tokenizer.decode(tokens[0, len(prompt_tokens):].tolist())


@click.command()
@click.option('checkpoint_dir', '--checkpoint_dir', required=True)
@click.option('tokenizer_model_file', '--tokenizer_model_file', required=True)
@click.option('prompt', '--prompt', required=True)
@click.option('max_new_tokens', '--max_new_tokens', required=True, type=int)
@click.option('temperature', '--temperature', required=False, default=1)
@click.option('top_p', '--top_p', required=False, default=5)
def main(checkpoint_dir, tokenizer_model_file, prompt, max_new_tokens, temperature, top_p):
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
    with open(checkpoint_dir + '/config.json', 'r') as f:
        config_json = json.loads(json.load(f))
        config = Config(**config_json)

    llm = LLM(tokenizer.vocab_size(), tokenizer.pad_id(), config)
    state = torch.load(checkpoint_dir + '/weights.pt', map_location='cuda:0')
    llm.load_state_dict(state, strict=False)

    print(generate(llm, tokenizer, prompt, config.context_size,
          max_new_tokens, temperature, top_p))


if __name__ == '__main__':
    main()
