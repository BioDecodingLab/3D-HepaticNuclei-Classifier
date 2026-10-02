"""Original direct-model architectures, retained without structural changes."""

import torch
from torch import nn


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SmallResNet3D(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, base_channels=32, dropout=0.2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, blocks=2, stride=2
        )
        self.layer4 = self._make_layer(
            base_channels * 4, base_channels * 8, blocks=2, stride=2
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, num_classes)

        self._init_weights()

    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [BasicBlock3D(in_planes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_resnet3d_classifier(num_classes, dropout=0.2, base_channels=32):
    return SmallResNet3D(
        num_classes=num_classes,
        in_channels=1,
        base_channels=base_channels,
        dropout=dropout,
    )


class DINOBackboneWrapper(nn.Module):
    """
    Wraps your DINO model and extracts a feature tensor.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)

        # common possibilities
        if torch.is_tensor(out):
            feats = out
        elif isinstance(out, dict):
            for key in [
                "x_norm_clstoken",
                "x_cls",
                "cls_token",
                "features",
                "embeddings",
                "x",
            ]:
                if key in out and torch.is_tensor(out[key]):
                    feats = out[key]
                    break
            else:
                raise TypeError(
                    f"Could not find tensor features in DINO dict output keys: {list(out.keys())}"
                )
        elif isinstance(out, (list, tuple)):
            tensor_candidates = [z for z in out if torch.is_tensor(z)]
            if len(tensor_candidates) == 0:
                raise TypeError("DINO returned list/tuple without tensor outputs.")
            feats = tensor_candidates[0]
        else:
            raise TypeError(f"Unsupported DINO output type: {type(out)}")

        if feats.ndim > 2:
            feats = feats.flatten(start_dim=1)

        return feats


class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.feature_extractor = DINOBackboneWrapper(backbone)

        if hasattr(backbone, "embed_dim"):
            feat_dim = backbone.embed_dim
        elif hasattr(backbone, "num_features"):
            feat_dim = backbone.num_features
        else:
            raise AttributeError(
                "Could not infer feature dimension from backbone. "
                "Expected attribute 'embed_dim' or 'num_features'."
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        return logits
