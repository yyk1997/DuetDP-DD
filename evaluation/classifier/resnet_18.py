import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type="batch"):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = self._get_norm(norm_type, planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = self._get_norm(norm_type, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                self._get_norm(norm_type, planes * self.expansion),
            )

    def _get_norm(self, norm_type, num_channels):
        if norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            # InstanceNorm-style GroupNorm, consistent with your ConvNet setting.
            return nn.GroupNorm(num_channels, num_channels, affine=True)
        elif norm_type == "group":
            return nn.GroupNorm(4, num_channels, affine=True)
        elif norm_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet18(nn.Module):
    """
    CIFAR-style ResNet-18.

    The interface is aligned with your ConvNet:
    - self.classifier is the final classification layer.
    - forward(x, return_features=True) returns (logit, out).
    - forward(x, return_features=False) returns logit.
    """

    def __init__(
        self,
        num_classes,
        channel=3,
        norm_type="batch",
        base_width=64,
        im_size=(32, 32),
    ):
        super(ResNet18, self).__init__()

        self.in_planes = base_width
        self.norm_type = norm_type

        # CIFAR-style stem: 3x3 conv, no maxpool.
        self.conv1 = nn.Conv2d(
            channel, base_width, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = self._get_norm(norm_type, base_width)

        self.layer1 = self._make_layer(BasicBlock, base_width,     num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_width * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_width * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_width * 8, num_blocks=2, stride=2)

        # For CIFAR-style ResNet-18, the final feature dimension is 512 if base_width=64.
        feat_dim = base_width * 8 * BasicBlock.expansion
        self.classifier = nn.Linear(feat_dim, num_classes)

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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=s,
                    norm_type=self.norm_type,
                )
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

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

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        features.append(out)

        out = self.layer1(out)
        features.append(out)

        out = self.layer2(out)
        features.append(out)

        out = self.layer3(out)
        features.append(out)

        out = self.layer4(out)
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