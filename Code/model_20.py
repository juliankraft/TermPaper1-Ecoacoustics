import torch
from lightning import LightningModule

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_max_pool: int, **kwargs):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', **kwargs)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', **kwargs)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=n_max_pool, stride=n_max_pool)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out) + x
        return self.maxpool(out)


class ResNet(LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            n_max_pool: int,
            n_res_blocks: int,
            num_classes: int,
            learning_rate: float = 0.001,
            **kwargs):
        super().__init__()

        self.learning_rate = learning_rate

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', **kwargs)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()
        self.res_blocks = torch.nn.Sequential(
            *[ResBlock(
                in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, n_max_pool=n_max_pool, **kwargs) for _ in range(n_res_blocks)]
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.convout = torch.nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1, **kwargs)

        self.softmax = torch.nn.Softmax(dim=1)

        self.cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # x: (N, C=1, H, W)
        x = x.unsqueeze(1)

        # Run input through a first convolutional layer.
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        # Run input through the residual blocks.
        out = self.res_blocks(out)

        # Run input through the average pooling layer. The output is a tensor of shape (N, C, 1, 1).
        out = self.avgpool(out)

        # Run input through the output convolutional layer. This is the same as a fully connected layer but it works with 4D tensors.
        out = self.convout(out)

        # Flatten the output tensor to have shape (N, C).
        out  = out.flatten(1)

        out = self.softmax(out)

        return out

    def accuracy(self, y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        labels = torch.argmax(y, dim=1)
        return torch.sum(labels == labels_hat).item() / (len(y) * 1.0)

    def cross_entropy(self, y_hat, y):
        return self.cross_entropy_loss_fn(y_hat, y.float())

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch

        y_hat = self(x)

        loss = self.cross_entropy(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log_dict({'train_loss': loss, 'train_acc': self.accuracy(y_hat, y)})

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch

        y_hat = self(x)

        loss = self.cross_entropy(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log_dict({'val_loss': loss, 'val_acc': self.accuracy(y_hat, y)})

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0)