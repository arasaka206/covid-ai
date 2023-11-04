from torch import nn
from torchvision import models
from torchmetrics.classification import MulticlassAUROC
from pytorch_lightning import LightningModule
from typing import Tuple
import torch
import torch.nn.functional as F

class CovidMmodel(LightningModule):
    def __init__(self,
                 input_shape: Tuple[int] = (3, 224, 224),
                 num_classes: int = 3,
                 learning_rate: float = 5e-4,
                 dropout: float = 0.25,
                 label_smoothing: float = 0.1):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.feature_extractor = models.resnet18(pretrained = True)
        n_sizes = self.get_feature_extractor_output(input_shape)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_sizes, num_classes)
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing)
        self.aucroc = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None)

    def get_feature_extractor_output(self, input_shape):
        tmp_input = torch.autograd.Variable(torch.rand(1, *input_shape))
        output_feature = self.feature_extractor(tmp_input)
        n_size = output_feature.data.view(1, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    def calculate_accuracy(self, y_hat, y):
        _, preds = y_hat.max(dim=1)
        acc = (preds == y).sum().item() / len(y)
        return acc
    def training_step(self, batch, batch_idx):
        image, label = batch
        prediction = self.forward(image)
        loss = self.cross_entropy(prediction, label)
        loss = F.cross_entropy(prediction, label)
        acc = self.calculate_accuracy(prediction, label)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        return loss,acc

    def validation_step(self, batch, batch_idx):
        image, label = batch
        prediction = self.forward(image)
        loss = self.cross_entropy(prediction, label)
        loss = F.cross_entropy(prediction, label)
        acc = self.calculate_accuracy(prediction, label)
        return {"loss": loss, "prediction": prediction, "label": label}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        prediction = torch.cat([x['prediction'] for x in outputs], dim=0)
        label = torch.cat([x['label'] for x in outputs], dim=0)
        aucroc = self.aucroc(prediction, label)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/aucroc/covid", aucroc[0], on_epoch=True, prog_bar=True)
        self.log("val/aucroc/non-covid", aucroc[1], on_epoch=True, prog_bar=True)
        self.log("val/aucroc/normal", aucroc[2], on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        prediction = self.forward(image)
        loss = self.cross_entropy(prediction, label)
        return {"loss": loss, "prediction": prediction, "label": label}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        prediction = torch.cat([x['prediction'] for x in outputs], dim=0)
        label = torch.cat([x['label'] for x in outputs], dim=0)
        aucroc = self.aucroc(prediction, label)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("test/aucroc/covid", aucroc[0], on_epoch=True, prog_bar=True)
        self.log("test/aucroc/non-covid", aucroc[1], on_epoch=True, prog_bar=True)
        self.log("test/aucroc/normal", aucroc[2], on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
