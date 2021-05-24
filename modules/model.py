import torch
import numpy as np
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from utils import total_variation, to_device, downsample, mse_loss, detection_metric, segmentation_metric

#############################
#        Parameters         #
#############################
CE_WEIGHTS = [0.4, 0.7, 0.9]
TV_WEIGHT = 0


class LocationAware1X1Conv2d(torch.nn.Conv2d):
    """
    Location-Dependent convolutional layer in accordance to
    (Azizi, N., Farazi, H., & Behnke, S. (2018, October).
    Location dependency in video prediction. In International Conference on Artificial Neural Networks (pp. 630-638). Springer, Cham.)
    """

    def __init__(self, w, h, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, bias=bias)
        self.locationBias = torch.nn.Parameter(torch.rand((1, 3, h, w), requires_grad=True))

    def forward(self, inputs, w, h):
        # Upsample location bias to match image size
        if self.locationBias[0].shape != (3, h, w):
            upsampled_bias = torch.nn.functional.interpolate(self.locationBias, size=((h, w)), mode='nearest')
            self.locationBias = torch.nn.Parameter(upsampled_bias, requires_grad=True)

        # Perform convolution
        convRes = super().forward(inputs)

        # Add location bias
        return convRes + self.locationBias


class Res18Encoder(torch.nn.Module):
    def __init__(self):
        """
        Initialize ResNet-18 encoder used in the model.
        """
        super().__init__()

        # Load pretrained ResNet18
        self.resnet18 = models.resnet18(pretrained=True)

        # Get ResNet components
        resnet_children = list(self.resnet18.children())

        # Create encoder hierarchy
        self.conv1 = resnet_children[0]
        self.bn1 = resnet_children[1]
        self.relu = resnet_children[2]
        self.maxpool = resnet_children[3]
        self.layer1 = resnet_children[4]
        self.layer2 = resnet_children[5]
        self.layer3 = resnet_children[6]
        self.layer4 = resnet_children[7]

    def freeze(self):
        """
        Freeze encoder weights
        """
        for child in self.resnet18.children():
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze encoder weights
        """
        for child in self.resnet18.children():
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Definition of the forward function of the model.

        :param x: Input Image
        :return: Four outputs of the model at different stages
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


class Decoder(torch.nn.Module):
    def __init__(self, w: int, h: int):
        """
        Initialize the decoder used in model.

        Defined layers are in order of usage.

        :param w: The width of the output image 
        :param h: The height of the output image
        """
        super().__init__()

        # ReLu
        self.relu = torch.nn.functional.relu

        # Transposed convolution 1
        self.convTrans1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)

        # Convolution 1
        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1)

        # Batch Norm 1
        self.bn1 = torch.nn.BatchNorm2d(512)

        # Transposed convolution 1
        self.convTrans2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)

        # Convolution 2
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=1, stride=1)

        # Batch Norm 2
        self.bn2 = torch.nn.BatchNorm2d(512)

        # Transposed convolution 3
        self.convTrans3 = torch.nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0)

        # Convolution 3
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1)

        # Batch Norm 3
        self.bn3 = torch.nn.BatchNorm2d(256)

        self.convD = LocationAware1X1Conv2d(w, h, 256, 3)
        self.convS = LocationAware1X1Conv2d(w, h, 256, 3)

        # share learnable bias between both heads by remove the locationBias from the segmentation head
        # and override it with the locationBias from the detection head
        del self.convS.locationBias
        self.convS.locationBias = self.convD.locationBias

    def forward(self, w: int, h: int, x4: torch.tensor, x3: torch.tensor, x2: torch.tensor, x1: torch.tensor) -> torch.tensor:
        """
        Definition of the forward function of the model (called when a parameter is passed
        as through ex: model(x))
        :param w: Final output of the encoder
        :param h: Output from layer 3 in the encoder
        :param x4: Final output of the encoder
        :param x3: Output from layer 3 in the encoder
        :param x2: Output from layer 2 in the encoder
        :param x1: Output from layer 1 in the encoder
        :return: Either detection or segmentation result
        """
        x = self.relu(x4)
        x = self.convTrans1(x)
        x = torch.cat((x, self.conv1(x3)), 1)

        x = self.bn1(self.relu(x))
        x = self.convTrans2(x)
        x = torch.cat((x, self.conv2(x2)), 1)

        x = self.bn2(self.relu(x))
        x = self.convTrans3(x)
        x = torch.cat((x, self.conv3(x1)), 1)
        x = self.bn3(self.relu(x))

        xs = self.convS(x, w, h)
        xd = self.convD(x, w, h)
        return xs, xd


class Model(torch.nn.Module):
    def __init__(self, device, w: int, h: int):
        """
        Initialize the decoder used in model.

        :param w: Width of input image 
        :param h: Height of input image
        """
        super().__init__()

        # CPU or GPU
        self.device = device

        # Model encoder
        self.encoder = Res18Encoder()

        # Model decoder
        self.decoder = Decoder(int(w/4), int(h/4))

        # Cross-Entropy loss
        self.weights = torch.tensor([0.4, 0.7, 0.9])
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.weights)

    def freeze_encoder(self):
        """
        Freeze encoder weights
        """
        self.encoder.freeze()

    def unfreeze_encoder(self):
        """
        Unfreeze encoder weights
        """
        self.encoder.unfreeze()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Definition of the forward function of the model (called when a parameter is passed
        as through ex: model(x))

        :param x: Input image
        :return: Detection and Segmentation result
        """
        # Ouput resolution
        w = int(x.shape[3]/4)
        h = int(x.shape[2]/4)

        # Encoder feature maps
        x1, x2, x3, x4 = self.encoder(x)

        # Decoder output
        xs, xd = self.decoder(w, h, x4, x3, x2, x1)

        return xs, xd

    def training_step_detection(self, batch: tuple) -> float:
        """
        One training step for the detection head of the network

        :param batch: Tuple of training images and target output
        :return: Mean Squared Error training loss
        """
        self.train()

        # Unpack batch
        images, targets = batch

        # Downsample targets to 0.25 * dimension
        downsampled_target = downsample(targets)
        downsampled_target = to_device(downsampled_target, self.device)

        # Run forward pass
        _, output = self.forward(images)

        # Return MSE loss
        return mse_loss(output, downsampled_target)

    def validation_detection(self, dataloader):
        total_f1, total_accuracy, total_recall, total_precision, total_fdr = detection_metric(self, dataloader)
        return total_f1, total_accuracy, total_recall, total_precision, total_fdr

    def training_step_segmentation(self, batch):
        """
        One training step for the segmentation head of the network

        :param batch: Tuple of training images and target output
        :return: Cross Entropy training loss
        """
        self.train()
        # Unpack batch
        images, targets = batch

        downsampled_target = downsample(targets)
        downsampled_target = torch.squeeze(downsampled_target, dim=1)  # (batch_size,1,H,W) -> (batch_size,H,W)
        downsampled_target = downsampled_target.type(torch.LongTensor)  # convert the target from float to int
        downsampled_target = to_device(downsampled_target, self.device)

        # Run forward pass
        output, _ = self.forward(images)

        # Return Cross-Entropy loss + Total Variation loss

        ce_loss = torch.nn.CrossEntropyLoss(weight=to_device(torch.tensor(CE_WEIGHTS), self.device))
        tvar_loss = total_variation(output, TV_WEIGHT, [0, 1])
        return ce_loss(output, downsampled_target) + tvar_loss

    def validation_segmentation(self, dataloader):
        accuracy, iou = segmentation_metric(self, dataloader)
        return accuracy, iou


class EfficientNetB0Encoder(torch.nn.Module):
    """
    Initialize EfficientNet-B0 encoder used in the model.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained EfficientNetB0
        self.efficientNetB0 = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv1 = torch.nn.Conv2d(320, 512, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(512, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

        self.conv2 = torch.nn.Conv2d(112, 256, kernel_size=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(256, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

        self.conv3 = torch.nn.Conv2d(40, 128, kernel_size=1, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(128, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

        self.conv4 = torch.nn.Conv2d(24, 64, kernel_size=1, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

    def freeze(self):
        """
        Freeze encoder weights
        """
        for child in self.efficientNetB0.children():
            for param in child.parameters():
                param.requires_grad = False

        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.conv3.parameters():
            param.requires_grad = False

        for param in self.conv4.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze encoder weights
        """
        for child in self.efficientNetB0.children():
            for param in child.parameters():
                param.requires_grad = True

        for param in self.conv1.parameters():
            param.requires_grad = True

        for param in self.conv2.parameters():
            param.requires_grad = True

        for param in self.conv3.parameters():
            param.requires_grad = True

        for param in self.conv4.parameters():
            param.requires_grad = True

    def forward(self, x: torch.tensor) -> torch.tensor:
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self.efficientNetB0._swish(self.efficientNetB0._bn0(self.efficientNetB0._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self.efficientNetB0._blocks):
            drop_connect_rate = self.efficientNetB0._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.efficientNetB0._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == 10:
                #x3 = self.efficientNetB0._swish(self.bn2(self.conv2(x)))
                x3 = x
            if idx == 4:
                #x2 = self.efficientNetB0._swish(self.bn3(self.conv3(x)))
                x2 = x
            if idx == 2:
                #x1 = self.efficientNetB0._swish(self.bn4(self.conv4(x)))
                x1 = x

        # Head
        #x4 = self.efficientNetB0._swish(self.bn1(self.conv1(x)))
        x4 = x

        return x1, x2, x3, x4


class EfficientNetB0Decoder(torch.nn.Module):
    def __init__(self, w: int, h: int):
        """
        Initialize the decoder used in model.

        Transpose convolutional layers are used for up-sampling,
        and location-dependent convolutional layers are used in the
        output heads with the learnable bias shared between them.

        Defined layers are in order of usage.

        :param w: The max width of the output image 
        :param h: The max height of the output image
        """
        super().__init__()

        # ReLu
        self.relu = torch.nn.functional.relu

        # Transposed convolution 1
        self.convTrans1 = torch.nn.ConvTranspose2d(320, 256, kernel_size=2, stride=2, padding=0)

        # Convolution 1
        self.conv1 = torch.nn.Conv2d(112, 256, kernel_size=1, stride=1)

        # Batch Norm 1
        self.bn1 = torch.nn.BatchNorm2d(512)

        # Transposed convolution 1
        self.convTrans2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)

        # Convolution 2
        self.conv2 = torch.nn.Conv2d(40, 256, kernel_size=1, stride=1)

        # Batch Norm 2
        self.bn2 = torch.nn.BatchNorm2d(512)

        # Transposed convolution 3
        self.convTrans3 = torch.nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0)

        # Convolution 3
        #self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(24, 128, kernel_size=1, stride=1)

        # Batch Norm 3
        self.bn3 = torch.nn.BatchNorm2d(256)

        self.convD = LocationAware1X1Conv2d(w, h, 256, 3)
        self.convS = LocationAware1X1Conv2d(w, h, 256, 3)

        # share learnable bias between both heads by remove the locationBias from the segmentation head
        # and override it with the locationBias from the detection head
        del self.convS.locationBias
        self.convS.locationBias = self.convD.locationBias

    def forward(self, w: int, h: int, x4: torch.tensor, x3: torch.tensor, x2: torch.tensor, x1: torch.tensor) -> torch.tensor:
        """
        Definition of the forward function of the model (called when a parameter is passed
        as through ex: model(x))
        :param w: Final output of the encoder
        :param h: Output from layer 3 in the encoder
        :param x4: Final output of the encoder
        :param x3: Output from layer 3 in the encoder
        :param x2: Output from layer 2 in the encoder
        :param x1: Output from layer 1 in the encoder
        :return: Either detection or segmentation result
        """
        x = self.relu(x4)
        x = self.convTrans1(x)
        x = torch.cat((x, self.conv1(x3)), 1)

        x = self.bn1(self.relu(x))
        x = self.convTrans2(x)
        x = torch.cat((x, self.conv2(x2)), 1)

        x = self.bn2(self.relu(x))
        x = self.convTrans3(x)
        x = torch.cat((x, self.conv3(x1)), 1)
        x = self.bn3(self.relu(x))

        xs = self.convS(x, w, h)
        xd = self.convD(x, w, h)

        return xs, xd


class EfficientNetB0Model(torch.nn.Module):
    def __init__(self, device, w: int, h: int):
        """
        Initialize the decoder used in model.
        Transpose-convolutional layers are used for up-sampling the representations
        :param w: Width of input image 
        :param h: Height of input image
        """
        super().__init__()

        # CPU or GPU
        self.device = device

        # Model encoder
        self.encoder = EfficientNetB0Encoder()

        # Model decoder
        self.decoder = EfficientNetB0Decoder(int(w/4), int(h/4))

    def freeze_encoder(self):
        """
        Freeze encoder weights
        """
        self.encoder.freeze()

    def unfreeze_encoder(self):
        """
        Unfreeze encoder weights
        """
        self.encoder.unfreeze()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Definition of the forward function of the model (called when a parameter is passed
        as through ex: model(x))

        :param x: Four Inputs from the Decoder model
        :return: Detection and Segmentation result
        """
        # Ouput resolution
        w = int(x.shape[3]/4)
        h = int(x.shape[2]/4)

        # Encoder feature maps
        x1, x2, x3, x4 = self.encoder(x)

        # Decoder output
        xs, xd = self.decoder(w, h, x4, x3, x2, x1)

        return xs, xd

    def training_step_detection(self, batch):
        # Unpack batch
        images, targets = batch

        # Downsample targets to 0.25 * dimension
        downsampled_target = downsample(targets)
        downsampled_target = to_device(downsampled_target, self.device)

        # Run forward pass
        _, output = self.forward(images)

        # Return MSE
        return mse_loss(output, downsampled_target)

    def validation_detection(self, dataloader):
        total_f1, total_accuracy, total_recall, total_precision, total_fdr = detection_metric(self, dataloader)
        return total_f1, total_accuracy, total_recall, total_precision, total_fdr

    def training_step_segmentation(self, batch):
        # Unpack batch
        images, targets = batch

        downsampled_target = downsample(targets)
        downsampled_target = torch.squeeze(downsampled_target, dim=1)  # (batch_size,1,H,W) -> (batch_size,H,W)
        downsampled_target = downsampled_target.type(torch.LongTensor)  # convert the target from float to int
        downsampled_target = to_device(downsampled_target, self.device)

        # Run forward pass
        output, _ = self.forward(images)

        # Return Cross-Entropy loss + Total Variation loss
        ce_loss = torch.nn.CrossEntropyLoss(weight=to_device(torch.tensor(CE_WEIGHTS), self.device))
        tvar_loss = total_variation(output, TV_WEIGHT, [0, 1])
        return ce_loss(output, downsampled_target) + tvar_loss

    def validation_segmentation(self, dataloader):
        accuracy, iou = segmentation_metric(self, dataloader)
        return accuracy, iou
