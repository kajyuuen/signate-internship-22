import torch
import pytorch_lightning as pl
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig
)
from transformers import AdamW

import numpy as np

class BERTFinetuner(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        output_name: str,
        learning_rate: float               
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        
        self.valid_f1 = pl.metrics.classification.F1()
        self.best_valid_f1 = 0

        self.learning_rate = learning_rate
        self.output_name = output_name

        self.save_hyperparameters("learning_rate", "model_name")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], labels=batch["target"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], labels=batch["target"])
        val_loss, logits = outputs["loss"], outputs["logits"]

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, axis=1)
        labels = batch["target"]

        self.valid_f1(preds, labels)
        self.log('valid_f1', self.valid_f1)
        self.log("valid_loss", val_loss)

    def validation_epoch_end(self, outs):
        total_valid_f1 = self.valid_f1.compute()
        self.best_valid_f1 = max(self.best_valid_f1, total_valid_f1)
        self.logger.log_hyperparams(params=self.hparams, metrics={"best_valid_f1": self.best_valid_f1})

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        logits = outputs["logits"]
        ps = torch.softmax(logits, dim=1)
        probs = [p for _, p in ps.tolist()]
        preds = torch.argmax(ps, axis=1).tolist()

        return [batch["ids"], probs, preds]

    def test_epoch_end(self, output_results):
        rows = {}
        prob_rows = {}
        for ids, probs, preds in output_results:
            prob_rows.update({i: prob for i, prob in zip(ids.tolist(), probs)})
            rows.update({i: pred for i, pred in zip(ids.tolist(), preds)})

        # output
        # Probability
        text = "\n".join([ "{}, {}".format(k, v) for k, v in prob_rows.items()])
        with open("bert_prob_{}.csv".format(self.output_name), mode='w') as f:
            f.write(text)

        # Predict
        text = "\n".join([ "{}, {}".format(k, v) for k, v in rows.items()])
        with open("bert_result_{}.csv".format(self.output_name), mode='w') as f:
            f.write(text)
        
    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        return optimizer