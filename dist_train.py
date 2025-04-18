import torch
import torch.nn.functional as F
from data.tiny_shakespear import CharDataset
from torch.utils.data import DataLoader
from Transformer.llm import GPT

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import hydra
from omegaconf import DictConfig
import logging

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        train_data: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])
        # compile should be after DDP, refer to https://pytorch.org/docs/main/notes/ddp.html
        self.mode = torch.compile(self.model)
        self.model.train()

        # Creates a GradScaler for mixed precision training.
        self.scaler = torch.GradScaler()

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        # using mixed precision
        with torch.autocast(device_type='cuda'):
            output = self.model(source, False, 0)
            loss = self.loss_fn(output.flatten(start_dim=0, end_dim=-2), targets.flatten())
        # loss.backward()
        # self.optimizer.step()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    ddp_setup()
    dataset = CharDataset(cfg.data)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    model = GPT(cfg.model.vocabulary_size,
                cfg.model.embedding_size,
                cfg.model.training_seq_len,
                cfg.model.nlayer,
                cfg.model.nhead,
                cfg.model.ndim,
                cfg.model.ndim_feedforward,
                cfg.model.drop_out)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr)
    trainer = Trainer(data_loader, model, optimizer, cfg.args.save_every, cfg.args.snapshot_path)
    log.info(f"start training...")
    trainer.train(cfg.args.total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    main()