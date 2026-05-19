import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bn_size,
        drop_rate,
        norm_type="batch",
    ):
        super(_DenseLayer, self).__init__()

        inter_channels = bn_size * growth_rate

        self.norm1 = self._get_norm(norm_type, num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features,
            inter_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.norm2 = self._get_norm(norm_type, inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.drop_rate = drop_rate

    def _get_norm(self, norm_type, num_channels):
        if norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            return nn.GroupNorm(num_channels, num_channels, affine=True)
        elif norm_type == "group":
            return nn.GroupNorm(4, num_channels, affine=True)
        elif norm_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = torch.cat([x, out], dim=1)
        return out


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        num_input_features,
        growth_rate,
        bn_size,
        drop_rate,
        norm_type="batch",
    ):
        super(_DenseBlock, self).__init__()

        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_type=norm_type,
            )
            layers.append(layer)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        norm_type="batch",
    ):
        super(_Transition, self).__init__()

        self.norm = self._get_norm(norm_type, num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def _get_norm(self, norm_type, num_channels):
        if norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            return nn.GroupNorm(num_channels, num_channels, affine=True)
        elif norm_type == "group":
            return nn.GroupNorm(4, num_channels, affine=True)
        elif norm_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out


class DenseNet121(nn.Module):
    """
    CIFAR-style DenseNet-121.

    Interface aligned with your ConvNet:
    - self.classifier is the final classification layer.
    - forward(x, return_features=True) returns (logit, out).
    - forward(x, return_features=False) returns logit.
    """

    def __init__(
        self,
        num_classes,
        channel=3,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0.0,
        norm_type="batch",
        im_size=(32, 32),
    ):
        super(DenseNet121, self).__init__()

        self.norm_type = norm_type

        # CIFAR-style stem: 3x3 conv, no 7x7 conv and no maxpool.
        self.features = nn.Sequential(
            nn.Conv2d(
                channel,
                num_init_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self._get_norm(norm_type, num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features

        # Dense blocks and transition layers.
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_type=norm_type,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                num_output_features = num_features // 2
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_output_features,
                    norm_type=norm_type,
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_output_features

        self.features.add_module("norm_final", self._get_norm(norm_type, num_features))
        self.features.add_module("relu_final", nn.ReLU(inplace=True))

        self.classifier = nn.Linear(num_features, num_classes)

        self._initialize_weights()

    def _get_norm(self, norm_type, num_channels):
        if norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            return nn.GroupNorm(num_channels, num_channels, affine=True)
        elif norm_type == "group":
            return nn.GroupNorm(4, num_channels, affine=True)
        elif norm_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x, return_features=False):
        out = self.features(x)

        # Global average pooling.
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)

        logit = self.classifier(out)

        if return_features:
            return logit, out
        else:
            return logit

    def get_feature(self, x, return_prob=False, return_logit=False):
        features = []

        out = x
        for name, module in self.features.named_children():
            out = module(out)
            if "denseblock" in name or "transition" in name:
                features.append(out)

        feat = F.adaptive_avg_pool2d(out, 1)
        feat = torch.flatten(feat, 1)
        logit = self.classifier(feat)

        if return_prob:
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            return features, logit
        else:
            return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)