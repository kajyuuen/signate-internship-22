from src.nn.lightning_data_module.crowdfunding_data_module import CrowdfundingDataModule
from src.nn.lightning_module.bert_finetuner import BERTFinetuner
from src.utils import data_setup

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra
import numpy as np
import torch
import random

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@hydra.main(config_name="config/nn/config.yaml")
def main(cfg):
    fix_seed(cfg.seed)
    model_name = cfg.model.name

    train_df, test_df = data_setup(cfg.is_test, ["get_sentence"])
    data_module = CrowdfundingDataModule(
        model_name, train_df, test_df
    )
    model = BERTFinetuner(model_name,
                          num_labels=2,
                          learning_rate=cfg.model.learning_rate,
                          output_name=cfg.logger_name)

    logger_name = cfg.logger_name if not cfg.is_test else "test/" + cfg.logger_name
    tb_logger = pl_loggers.TensorBoardLogger(hydra.utils.to_absolute_path("lightning_logs"),
                                             name=logger_name,
                                             default_hp_metric=False)

    early_stop_callback = EarlyStopping(
        monitor='valid_f1',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(gpus=1,
                         max_epochs=cfg.trainer.max_epochs,
                         logger=tb_logger,
                         callbacks=[early_stop_callback])
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())

if __name__ == "__main__":
    main()

