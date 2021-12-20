from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from dataset_generator.rotation_comparator import RotationComparator
from rotation_dataset import RotationDataset
from train_lightning import RotationNetPl


class AccLoss:
    """Accumulate all the losses and average at end of epoch"""
    def __init__(self):
        self.total_loss = 0
        self.num_steps = 0

    def acc(self, loss):
        self.total_loss += loss
        self.num_steps += 1

    def compute(self):
        return self.total_loss / self.num_steps


class RotationNetPlInference(RotationNetPl):
    def __init__(self, generate_viz=False):
        super().__init__()
        self.rot_comp = RotationComparator("./test_visuals","./dataset_generator/coordinate.urdf")
        self.total_loss_test = 0.0
        self.total_step_test = 0
        self.generate_viz = generate_viz

        self.acc_train_deg = AccLoss()
        self.acc_test_deg = AccLoss()
        self.acc_val_deg = AccLoss()

    # Just get metrics on val and test
    # def training_step(self, batch, batch_idx):
    #     loss, angle = self._step(batch)
    #     self.acc_train_deg.acc(angle)
    #     return {"loss": loss}
    #
    # def on_train_epoch_end(self) -> None:
    #     # Log the final error in degrees. For readability
    #     mean_angle = self.acc_train_deg.compute()
    #     self.log(f"Inference/train_angle", mean_angle, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        loss, angle = self._step(batch)
        self.acc_val_deg.acc(angle)
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        # Log the final error in degrees. For readability
        mean_angle = self.acc_val_deg.compute()
        self.log(f"Inference/val_angle", mean_angle, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, angle = self._step(batch)
        self.acc_test_deg.acc(angle)

        if self.generate_viz:
            # Run inference again and generate visualizations of the outputs
            inputs = batch["input"]
            bsize = inputs.shape[0]
            labels = batch["target"].reshape(bsize, 3, 3)  # Shape: [B, 3, 3]
            preds = self.model(inputs)  # Shape: [B, 6]
            for idx, (gt_, pred_) in enumerate(zip(labels.detach().cpu(), preds.detach().cpu())):
                self.rot_comp.compare_rotations(gt_.numpy(), pred_.numpy(), batch_idx*bsize+idx)

        return {"loss": loss}

    def on_test_epoch_end(self) -> None:
        # Log the final error in degrees. For readability
        mean_angle = self.acc_test_deg.compute()
        self.log(f"Inference/test_angle", mean_angle, on_step=False, on_epoch=True)


def main():
    pl.seed_everything(42, workers=True)  # set seed for reproducibility
    GENERATE_VIZ = False
    wandb_project = "rotenv"
    WANDB_RUN_ID = "16t1xy13"
    CKPT_PATH = f"./logs/{wandb_project}/{WANDB_RUN_ID}/checkpoints/best.ckpt"  # Make sure the keep the logs structure

    dir_root = Path("./logs")
    dir_root.mkdir(exist_ok=True)
    model = RotationNetPlInference.load_from_checkpoint(CKPT_PATH, generate_viz=GENERATE_VIZ)
    wb_logger = pl_loggers.WandbLogger(name=None, id=WANDB_RUN_ID, entity="rotteam", project=wandb_project,
                                       save_dir=str("./logs"))

    data_root_dir ="./dataset/"
    #if not data_root_dir.is_dir():
        #raise ValueError(f"Dir does not exist: {data_root_dir}")

    test_dataset = RotationDataset(data_root_dir+"test" + '/', True)
    val_dataset = RotationDataset(data_root_dir+"val" + '/', True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1, drop_last=False)

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        logger=wb_logger
    )

    # Validation
    _ = trainer.validate(model, val_dataloaders=val_loader)

    # Testing
    _ = trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
