import torch
from lightning import LightningModule

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_max_pool: int, **kwargs):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', **kwargs)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', **kwargs)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=n_max_pool, stride=n_max_pool)

        self.residual_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same', **kwargs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out) + self.residual_conv(x)
        return self.maxpool(out)


class ResNet(LightningModule):
    def __init__(
            self,
            in_channels: int,
            base_channels: int,
            kernel_size: int,
            n_max_pool: int,
            n_res_blocks: int,
            num_classes: int,
            learning_rate: float = 0.001,
            class_weights: list[float] | None = None,
            **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.class_weights = class_weights

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=kernel_size, padding='same', **kwargs)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=base_channels)
        self.relu = torch.nn.ReLU()

        # Create a sequence of residual blocks with increasing depth (channel size).
        resnet_layers = []
        current_out_channels = base_channels
        current_in_channels = current_out_channels
        for depth in range(n_res_blocks):
            current_out_channels *= 2
            n_max_pool = n_max_pool if depth % 2 == 0 else 1
            resnet_layers.append(ResBlock(
                in_channels=current_in_channels, out_channels=current_out_channels, kernel_size=kernel_size, n_max_pool=n_max_pool, **kwargs)
            )

            current_in_channels = current_out_channels

        self.res_blocks = torch.nn.Sequential(
            *resnet_layers
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.convout = torch.nn.Conv2d(in_channels=current_out_channels, out_channels=num_classes, kernel_size=1, **kwargs)

        self.softmax = torch.nn.Softmax(dim=1)

        if class_weights is None:
            class_weights = None
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

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
        x, y, _ = batch

        y_hat = self(x)

        loss = self.cross_entropy(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log_dict({'train_loss': loss, 'train_acc': self.accuracy(y_hat, y)}, logger=True, on_step=True, on_epoch=True)

        # self.log('train_loss', loss, on_epoch=True, on_epoch=True)
        # self.log('train_acc', self.accuracy(y_hat, y), on_epoch=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, _ = batch

        y_hat = self(x)

        loss = self.cross_entropy(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log_dict({'val_loss': loss, 'val_acc': self.accuracy(y_hat, y)})

        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, _ = batch

        y_hat = self(x)

        loss = self.cross_entropy(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log_dict({'test_loss': loss, 'test_acc': self.accuracy(y_hat, y)})

        return loss

    def predict_step(self, batch, batch_idx):
        x, _, idx = batch

        y_hat = self(x).detach().cpu().numpy()

        del x

        return {'y_hat': y_hat, 'idx': idx}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
