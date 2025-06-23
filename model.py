import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassificationModel(L.LightningModule):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # Use the class balancing weights linked from the datamodule
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        return loss

    def validation_step(self, batch, batch_idx):
        ...

