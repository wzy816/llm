import dataclasses
import datetime
import json
import math
import os
import random
from dataclasses import dataclass

import click
import sentencepiece as spm
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

from llm.dataset import Dataset
from llm.model import LLM


@dataclass
class Config:
    # 117M model
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    head_dim: int = dim // num_heads
    max_seq_len: int = 1024

    # dataset
    context_size: int = max_seq_len
    sample_every: int = 256

    # train
    batch_size: int = 32
    micro_batch_size: int = 100
    min_save_step: int = 20
    max_save_loss: float = 2.0
    min_train_lr: float = 1e-12
    max_train_step: int = 2000

    # optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    
    # ReduceLROnPlateau
    patience: int = 50
    factor: float = 0.1
        
    # placeholder
    total_params: str = ''
    vocab_size: int = 0


class Trainer():
    def __init__(self, project, tokenizer, checkpoint_dir, data_dir, output_dir, config, init_step):
        self.project = project
        self.name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.config = config
        
        self.tokenizer = tokenizer
        self.config.vocab_size = self.tokenizer.vocab_size()
        self.padding_idx = self.tokenizer.pad_id()

        self.llm = LLM(self.config.vocab_size, self.padding_idx, self.config)
        if checkpoint_dir is not None:
            state = torch.load(checkpoint_dir+'/weights.pt',
                               map_location='cuda:0')
            self.llm.load_state_dict(state, strict=False)
        else:
            self.llm.init_weights()
        print(self.llm)

        self.config.total_params = self.llm.count_parameters()        
        print(f'model parameter {self.config.total_params}')

        dataset = Dataset(tokenizer,
                               self.config.context_size,
                               data_dir,
                               self.config.sample_every)
        self.loader = DataLoader(dataset,
                                 batch_size=self.config.batch_size)
        self.optimizer = AdamW(self.llm.parameters(),
                               lr=self.config.lr,
                               weight_decay=self.config.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                       patience=self.config.patience,
                                                                       factor=self.config.factor,
                                                                       eps=self.config.min_train_lr)
        self.output_dir = output_dir
        self.init_step = init_step

    def compute_loss(self, logits, y):
        return F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1), ignore_index=-1)

    def train(self, use_wandb=True):
        if use_wandb:            
            wandb.init(project=self.project,
                       name=self.name,
                       config=self.config)

        self.llm.train()
        wandb.watch(self.llm, log='all')

        scaler = torch.cuda.amp.GradScaler()
        step = self.init_step
        loss = math.inf
        token_cnt = 0
        last_save_loss = None
        last_save_step = 0
        it = iter(self.loader)

        while True:
            lr = self.optimizer.param_groups[0]['lr']

            prev_loss = loss
            loss = 0
            token_per_step = 0
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for _ in range(self.config.micro_batch_size):
                    x, y = next(it)
                    token_per_step += int(torch.count_nonzero(x))
                    logits = self.llm.forward(x)
                    l = self.compute_loss(logits, y)
                    if not math.isnan(l):
                        micro_loss = l / self.config.micro_batch_size
                        loss += micro_loss
                        scaler.scale(micro_loss).backward()
            token_cnt += token_per_step

            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.llm.parameters(), max_norm=1)
            scaler.step(self.optimizer)
            scale = scaler.get_scale()
            scaler.update()
            if scale <= scaler.get_scale():
                self.lr_scheduler.step(loss)

            self.optimizer.zero_grad(set_to_none=True)

            if use_wandb:
                wandb.log({'loss': loss,
                           'token_cnt': token_cnt,
                           'learning_rate': lr,
                           'token_per_step': token_per_step,
                           'loss_ratio': loss / prev_loss}, step=step)

            if math.isnan(loss):
                break
            if last_save_loss is None or loss < last_save_loss:
                if step - last_save_step >= self.config.min_save_step:
                    if loss <= self.config.max_save_loss:
                        self.save(step, loss)
                        last_save_step = step
                        last_save_loss = loss

            step += 1
            if step > self.config.max_train_step:
                break

        if use_wandb:
            wandb.finish()

    def save(self, step, loss):
        directory = os.path.join(
            self.output_dir, self.name, f'step={step}_loss={loss:.3f}')
        os.makedirs(directory, exist_ok=True)

        ck_path = directory + '/weights.pt'
        torch.save(self.llm.state_dict(), ck_path)

        cf_path = directory + '/config.json'
        config_json = json.dumps(dataclasses.asdict(self.config))
        with open(cf_path, "w") as o:
            json.dump(config_json, o)


@click.command()
@click.option('project', '--project', required=True)
@click.option('data_dir', '--data_dir', required=True)
@click.option('tokenizer_model_file', '--tokenizer_model_file', required=True)
@click.option('output_dir', '--output_dir', required=True)
@click.option('config_file', '--config_file', required=False)
@click.option('checkpoint_dir', '--checkpoint_dir', required=False, default=None)
@click.option('init_step', '--init_step', required=False, default=0)
def main(project, data_dir, tokenizer_model_file, output_dir, config_file, checkpoint_dir, init_step):
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    
    config = Config()
    if config_file is not None:
        with open(config_file, 'r') as f:
            try:
                d = yaml.safe_load(f)
                config = Config(**d)
            except yaml.YAMLError as e:
                print(e)
    
    trainer = Trainer(project, tokenizer, checkpoint_dir,
                      data_dir, output_dir, config, init_step)
    trainer.train()


if __name__ == '__main__':
    main()
