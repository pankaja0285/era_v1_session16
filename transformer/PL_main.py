import torch
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import Callback
import torchmetrics
from .config import get_weights_file_path
#from config import get_weights_file_path
import pytorch_lightning as pl


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, config, verbose: bool = False):
        super().__init__()
        self.config = config
        self.verbose = verbose

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        # # save the model at the end of every epoch
        # model_filename = get_weights_file_path(self.config, f"{trainer.current_epoch}")
        # trainer.save_checkpoint(model_filename)
        
        # **** Save only the last 2 epochs as I am running out of space
        current_epoch = trainer.current_epoch
        max_epochs = self.config['num_epochs']
        epochs_to_save = self.config['epochs_to_save']
        print(f"Difference of max epochs and current epoch, to check for saving: {max_epochs - current_epoch}")
        if abs(max_epochs - current_epoch) <= epochs_to_save:
            # model_filename = get_weights_file_path(self.config, f"{trainer.current_epoch}")
            # instead do this
            model_folder = self.config['model_folder']
            model_basename = self.config['model_basename']
            model_filename_only = f"{model_basename}{current_epoch}.pt"
            # str(Path('.') / model_folder / model_filename)
            # modified path to below instead...
            model_filename = f"{os.getcwd()}/{model_folder}/{model_filename_only}"
            trainer.save_checkpoint(model_filename)
        else:
            print(f"Skip saving current epoch {current_epoch} - weights...")

class PrintAccuracyAndLoss(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        trainer.model.log("train_epoch_loss", train_loss)
        print(f"Epoch {trainer.current_epoch}: train_loss={train_loss:.4f}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        assert len(trainer.model.predicted_list) > 0, "Validation: predicted list is empty"
        assert len(trainer.model.expected_list) > 0, "Validation: expected list is empty"

        # torchmetrics.CharErrorRate, torchmetrics.WordErrorRate and torchmetrics.BLEUScore are all deprecated
        # instead use from torchmetrics.text as shown below...
        # compute the character error rate
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(trainer.model.predicted_list, trainer.model.expected_list)

        # compute word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(trainer.model.predicted_list, trainer.model.expected_list)

        # compute the BLEU metric
        metric = torchmetrics.text.BLEUScore(n_gram=2)
        bleu = metric(trainer.model.predicted_list, trainer.model.expected_list)

        trainer.model.log("validation_epoch_wer", wer)
        trainer.model.log("validation_epoch_cer", cer)
        trainer.model.log("validation_epoch_bleu", bleu)
        trainer.model.predicted_list = []
        trainer.model.expected_list = []
        assert len(trainer.model.predicted_list) == 0, "Validation: predicted list is not reset"
        assert len(trainer.model.expected_list) == 0, "Validation: expected list is not reset"
        return

def train_transformer(model, datamodule, config, ckpt_path=None, epochs=2):
    trainer = Trainer(
        enable_checkpointing=True,
        max_epochs=epochs,
        accelerator="auto",
        #accelerator=None,
        devices=1 if torch.cuda.is_available() else None,
        #logger=CSVLogger(save_dir="logs/"),
        logger=TensorBoardLogger(save_dir=config["rundir"], name=config["experiment_name"], default_hp_metric=False),
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=10),
                   #RichProgressBar(refresh_rate=10, leave=True),
                   PeriodicCheckpoint(config, verbose=True),
                   PrintAccuracyAndLoss()],
        num_sanity_val_steps=0,
        precision=16
    )
    
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt_path)
    trainer.test(model, datamodule.test_dataloader())
    return trainer