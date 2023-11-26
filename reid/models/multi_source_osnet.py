from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .meta_modules import *
from .backbone.osnet import osnet_ibn_x1_0, osnet_x1_0


class OSNetBase(MetaModule):
    def __init__(self, name, net_config=None):
        super(OSNetBase, self).__init__()
        self.net_config = net_config

        model = {'osnet': osnet_x1_0(),
                 'osnet-ibn': osnet_ibn_x1_0()}

        osnet = model[name]
        self.base = nn.Sequential(
            osnet.conv, osnet.bn, osnet.relu, osnet.maxpool,
            osnet.conv2, osnet.conv3, osnet.conv4, osnet.conv5
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_features = osnet.fc[0].in_features

        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.meta_convert(self)
        self.meta_sequential_convert(self)

    def meta_convert(self, m):
        for name, sub_module in m.named_children():
            if isinstance(sub_module, nn.Conv2d):
                new_module = MetaConv2d(sub_module.in_channels, sub_module.out_channels,
                                        sub_module.kernel_size, sub_module.stride, sub_module.padding,
                                        bias=sub_module.bias is not None, groups=sub_module.groups)

                state_dict = sub_module.state_dict()
                new_module.load_state_dict(state_dict)
                setattr(m, name, new_module)
                continue
            if isinstance(sub_module, nn.BatchNorm2d):
                new_module = MetaBatchNorm2d(sub_module.num_features, affine=sub_module.affine)
                state_dict = sub_module.state_dict()
                new_module.load_state_dict(state_dict)
                setattr(m, name, new_module)
                continue
            if isinstance(sub_module, nn.BatchNorm1d):
                new_module = MetaBatchNorm1d(sub_module.num_features, affine=sub_module.affine)
                state_dict = sub_module.state_dict()
                new_module.load_state_dict(state_dict)
                setattr(m, name, new_module)
                continue
            if isinstance(sub_module, nn.InstanceNorm2d):
                new_module = MetaInstanceNorm2d(sub_module.num_features, affine=sub_module.affine)
                state_dict = sub_module.state_dict()
                new_module.load_state_dict(state_dict)
                setattr(m, name, new_module)
                continue
            if isinstance(sub_module, nn.Linear):
                new_module = MetaLinear(sub_module.in_features, sub_module.out_features,
                                        bias=sub_module.bias is not None)
                state_dict = sub_module.state_dict()
                new_module.load_state_dict(state_dict)
                setattr(m, name, new_module)
                continue
            self.meta_convert(sub_module)

    def meta_sequential_convert(self, m):
        for name, module in m.named_children():
            self.meta_sequential_convert(module)
            if isinstance(module, nn.Sequential):
                new_module = MetaSequential(*module._modules.values())
                setattr(m, name, new_module)

    def forward(self, x, params=None, training=False):
        x = self.base(x, params=self.get_subdict(params, 'base'))
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        bn_x = self.feat_bn(x, params=self.get_subdict(params, 'feat_bn'))
        if not self.training:
            if training:
                return bn_x
            bn_x = F.normalize(bn_x)
            return bn_x
        return x, bn_x


class Classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(num_features, num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        logits = self.classifier(x)
        return logits


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        x = self.projector(x)
        return x
