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
from hydra.core.config_store import ConfigStore

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
        # self.model = torch.compile(model)
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
        self.model.train()

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source, False, 0)
        loss = self.loss_fn(output.flatten(start_dim=0, end_dim=-2), targets.flatten())
        loss.backward()
        self.optimizer.step()

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

cs = ConfigStore.instance()

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    ddp_setup()
    dataset = CharDataset(cfg.Data)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.Data.batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    model = GPT(cfg.Model.vocabulary_size,
                cfg.Model.embedding_size,
                cfg.Model.training_seq_len,
                cfg.Model.nlayer,
                cfg.Model.nhead,
                cfg.Model.ndim,
                cfg.Model.ndim_feedforward,
                cfg.Model.drop_out)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.Model.lr)
    trainer = Trainer(model, data_loader, optimizer, cfg.Args.save_every, cfg.Args.snapshot_path)
    trainer.train(cfg.Args.total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    main()