from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

# from mxnet.ndarray.contrib import

# __all__ = ['blresnet_model']
__all__ = ['get_blmodel']

def blm_make_layer(block, inplanes, planes, blocks, stride=1, last_relu=True):
    downsample = nn.HybridSequential(prefix='')
    if stride != 1:
        downsample.add(nn.AvgPool2D(3, strides=2, padding=1))
    if inplanes != planes:
        downsample.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=inplanes))
        # ?
        downsample.add(BatchNorm(in_channels=planes))

    # downsample = []
    # if stride != 1:
    #     downsample.append(nn.AvgPool2D(3, strides=2, padding=1))
    # if inplanes != planes:
    #     downsample.append(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=inplanes))
    #     ?
    #     downsample.append(BatchNorm(in_channels=planes))
    # downsample = None if downsample == [] else downsample

    # layers = []
    layers = nn.HybridSequential(prefix='')
    if blocks == 1:
        layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
        # layers.add(block(inplanes, planes, stride=stride))
        # layers.append(block(inplanes, planes, stride=stride, downsample=downsample))
    else:
        layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
        # layers.add(block(inplanes, planes, stride=stride))
        # layers.append(block(inplanes, planes, stride=stride, downsample=downsample))
        for i in range(1, blocks):
            layers.add(block(planes, planes, last_relu=last_relu if i == blocks - 1 else True))
            # layers.append(block(planes, planes, last_relu=last_relu if i == blocks - 1 else True))
    return layers


def blr_make_layer(block, inplanes, planes, blocks, stride=1):
    # self.downsample = nn.HybridSequential(prefix='')
    downsample = nn.HybridSequential(prefix='')
    if stride != 1:
        downsample.add(nn.AvgPool2D(3, strides=2, padding=1))
    if inplanes != planes:
        downsample.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=inplanes))
        downsample.add(BatchNorm(in_channels=planes))

    layers = nn.HybridSequential(prefix='')
    layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
    for i in range(1, blocks):
        layers.add(block(planes, planes))
    return layers

"""
# TORCH
# bottleneck
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out
# bl模块
class bLModule(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, alpha, beta, stride):
        super(bLModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1, 2, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha, max(1, blocks // beta - 1))
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.fusion = self._make_layer(block, out_channels, out_channels, 1, stride=stride)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, stride=stride, downsample=downsample))
        else:
            layers.append(block(inplanes, planes, stride, downsample))
            for i in range(1, blocks):
                layers.append(block(planes, planes,
                                    last_relu=last_relu if i == blocks - 1 else True))

        return nn.Sequential(*layers)

    def forward(self, x):
        big = self.big(x)
        little = self.little(x)
        little = self.little_e(little)
        big = torch.nn.functional.interpolate(big, little.shape[2:])
        out = self.relu(big + little)
        out = self.fusion(out)
        return out

class bLResNet(nn.Module):
    def __init__(self, block, layers, alpha, beta, num_classes=1000):
        num_channels = [64, 128, 256, 512]
        self.inplanes = 64
        super(bLResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])

        # bL模块
        self.layer1 = bLModule(block, num_channels[0], num_channels[0] *
                               block.expansion, layers[0], alpha, beta, stride=2)
        self.layer2 = bLModule(block, num_channels[0] * block.expansion,
                               num_channels[1] * block.expansion, layers[1], alpha, beta, stride=2)
        self.layer3 = bLModule(block, num_channels[1] * block.expansion,
                               num_channels[2] * block.expansion, layers[2], alpha, beta, stride=1)
        self.layer4 = self._make_layer(
            block, num_channels[2] * block.expansion, num_channels[3] * block.expansion, layers[3], stride=2)


        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3] * block.expansion, num_classes)

        # 未完待续
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BottleneckV1):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        bx = self.b_conv0(x)
        bx = self.bn_b0(bx)
        lx = self.l_conv0(x)
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)
        x = self.relu(bx + lx)
        x = self.bl_init(x)
        x = self.bn_bl_init(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
# TORCH
"""


# MXNET
# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    # expansion = 4
    # def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):

    # add
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')

        self.body.add(nn.Conv2D(planes//self.expansion, kernel_size=1, strides=stride, in_channels=inplanes))
        self.body.add(BatchNorm(in_channels=planes // self.expansion))# ->
        self.body.add(nn.Activation('relu'))

        self.body.add(_conv3x3(planes//self.expansion, 1, planes//self.expansion))
        self.body.add(BatchNorm(in_channels=planes // self.expansion))
        self.body.add(nn.Activation('relu'))

        self.body.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=planes//self.expansion))
        self.body.add(BatchNorm(in_channels=planes))
        self.body.add(nn.Activation('relu'))

        # if downsample:
        #     self.downsample = downsample

        # self.downsample = nn.HybridSequential(prefix='')
        # if stride != 1:
        #     self.downsample.add(nn.AvgPool2D(3, strides=2, padding=1))
        # if inplanes != planes:
        #     self.downsample.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=inplanes))
        #     # ?
        #     self.downsample.add(BatchNorm(in_channels=planes))
        self.downsample = downsample

        self.last_relu = last_relu

        #     self.downsample = nn.HybridSequential(prefix='')
        #     for i in downsample:
        #         self.downsample.add(i)

        # self.last_relu = last_relu

        # if downsample:
        #     self.downsample = nn.HybridSequential(prefix='')
        #     self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
        #                                   use_bias=False, in_channels=in_channels))
        #     self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        # else:
        #     self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        out = self.body(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.last_relu:
            out = F.Activation(out + residual, act_type='relu')
            return out
        else:
            return out + residual

class BLModule(HybridBlock):
    def __init__(self, block, int_channels, out_channels, blocks, alpha, beta, stride, hw):
        super(BLModule, self).__init__()

        self.relu = nn.Activation('relu')
        # self.big = nn.HybridSequential(prefix='')
        # self.big.add(self._make_layer(block, int_channels, out_channels, blocks - 1, 2, last_relu=False))
        self.big = blm_make_layer(block, int_channels, out_channels, blocks - 1, 2, last_relu=False)

        self.little = nn.HybridSequential(prefix='')
        self.little.add(blm_make_layer(block, int_channels, out_channels // alpha, max(1, blocks // beta - 1)))
        # self.little = self._make_layer(block, int_channels, out_channels // alpha, max(1, blocks // beta - 1))

        self.little_e = nn.HybridSequential(prefix='')
        self.little_e.add(nn.Conv2D(out_channels, kernel_size=1, in_channels=out_channels // alpha))
        self.little_e.add(BatchNorm(in_channels=out_channels))

        # self._up_kwargs = {'height': height, 'width': width}
        # self.fusion = nn.HybridSequential(prefix='')
        # self.fusion.add(nn.Activation('relu'))
        # self.fusion.add(self._make_layer(block, out_channels, out_channels, 1, stride=stride))
        self.fusion = blm_make_layer(block, out_channels, out_channels, 1, stride=stride)
        self.hw = hw


    def hybrid_forward(self, F, x):
        big = self.big(x)
        little = self.little(x)
        little = self.little_e(little)
        # little = self.little_e(little)
        # nn.functional.py #2427
        # big = torch.nn.functional.interpolate(big, little.shape[2:])

        # oshape = little.infer_shape(data=(1, 3, 56, 56))[2:]
        # big = F.contrib.BilinearResize2D(data=big, like=oshape)

        # cls_id = little.slice_axis(axis=-1, begin=2, end=4)
        # print(cls_id)
        # big = F.contrib.BilinearResize2D(data=big, like=cls_id)
        # bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        # a = F.slice_axis(self.little, axis=0, begin=2, end=4)
        # print(a)

        big = F.contrib.BilinearResize2D(data=big, height=self.hw, width=self.hw)
        # big = F.contrib.BilinearResize2D(data=big, like=a)
        out = self.relu(big + little)
        #out = self.relu(little)

        # a = F.slice_axis(self.little, axis=0, begin=2, end=4)

        # like=little.shape[2:]
        # big = F.contrib.BilinearResize2D(data=big, **self._up_kwargs)
        # big = F.contrib.BilinearResize2D(data=big, like=little.slice_axis(little, axis=0, begin=2, end=4))

        # big = F.contrib.BilinearResize2D(data=big, like=little.shape[2:])
        #

        out = self.fusion(out)
        # out = self.relu(big + little)
        # out = self.fusion(out)
        return out

class BLResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, alpha=2, beta=4, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    # def __init__(self, block, layers, channels, alpha=2, beta=4, classes=1000, norm_layer=BatchNorm, **kwargs):
        super(BLResNetV1, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, in_channels=3))
                # momentom 0.9/0.1
                self.features.add(norm_layer(in_channels=channels[0]))
                self.features.add(nn.Activation('relu'))

            self.relu = nn.Activation('relu')

            # bl part1 : input 112
            # 3*3 ,64, s2
            #
            self.b_conv0 = nn.Conv2D(channels[0], 3, 2, 1, use_bias=False, in_channels=channels[0])
            self.bn_b0 = norm_layer(in_channels=channels[0])
            # 3*3 ,32; 3*3 ,32, s2; 1*1, 64
            self.l_conv0 = nn.Conv2D(channels[0] // alpha, 3, 1, 1, use_bias=False, in_channels=channels[0])
            self.bn_l0 = norm_layer(in_channels=channels[0] // alpha)
            self.l_conv1 = nn.Conv2D(channels[0] // alpha, 3, 2, 1, use_bias=False, in_channels=channels[0] // alpha)
            self.bn_l1 = norm_layer(in_channels=channels[0] // alpha)
            self.l_conv2 = nn.Conv2D(channels[0], 1, 1, use_bias=False, in_channels=channels[0] // alpha)
            self.bn_l2 = norm_layer(in_channels=channels[0])

            # input 56
            self.bl_init = nn.Conv2D(channels[0], 1, 1, use_bias=False, in_channels=channels[0])
            self.bn_bl_init = norm_layer(in_channels=channels[0])

            # bl part2 : 待修改
            self.layers = nn.HybridSequential(prefix='')
            # input 56
            self.layers.add(BLModule(block, channels[0], channels[0] * block.expansion, layers[0], alpha, beta, stride=2, hw = 56))
            # input 28
            self.layers.add(BLModule(block, channels[0] * block.expansion, channels[1] * block.expansion, layers[1], alpha, beta, stride=2, hw = 28))
            # input 14
            self.layers.add(BLModule(block, channels[1] * block.expansion, channels[2] * block.expansion, layers[2], alpha, beta, stride=1, hw =14))

            # _make_layer 实现未必正确
            # input 14
            self.layers.add(blr_make_layer(block, channels[2] * block.expansion, channels[3] * block.expansion, layers[3], stride=2))


            # 未更改
            # def _make_layer(self, block, inplanes, planes, blocks, stride=1):
            #     downsample = []
            #     if stride != 1:
            #         downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
            #     if inplanes != planes:
            #         downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            #         downsample.append(nn.BatchNorm2d(planes))
            #     downsample = None if downsample == [] else nn.Sequential(*downsample)
            #
            #     layers = []
            #     layers.append(block(inplanes, planes, stride, downsample))
            #     for i in range(1, blocks):
            #         layers.append(block(planes, planes))
            #     return nn.Sequential(*layers)

            # 待删除
            # for i, num_layer in enumerate(layers):
            #     stride = 1 if i == 0 else 2
            #     self.features.add(self._make_layer(block, num_layer, channels[i+1],
            #                                        stride, i+1, in_channels=channels[i],
            #                                        last_gamma=last_gamma, use_se=use_se,
            #                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs))


            # input 7*7
            self.layers.add(nn.GlobalAvgPool2D())
            self.layers.add(nn.Flatten())
            self.fc = nn.Dense(classes, in_units=channels[-1] * block.expansion)


    # def _make_layer(self, block, inplanes, planes, blocks, stride=1):
    #     downsample = []
    #     if stride != 1:
    #         downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
    #     if inplanes != planes:
    #         downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
    #         downsample.append(nn.BatchNorm2d(planes))
    #     downsample = None if downsample == [] else nn.Sequential(*downsample)
    #
    #     layers = []
    #     layers.append(block(inplanes, planes, stride, downsample))
    #     for i in range(1, blocks):
    #         layers.append(block(planes, planes))
    #
    #     return nn.Sequential(*layers)


    # def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
    #                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
    #
    #     layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
    #
    #     with layer.name_scope():
    #         layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
    #                         last_gamma=last_gamma, use_se=use_se, prefix='',
    #                         norm_layer=norm_layer, norm_kwargs=norm_kwargs))
    #         for _ in range(layers-1):
    #             layer.add(block(channels, 1, False, in_channels=channels,
    #                             last_gamma=last_gamma, use_se=use_se, prefix='',
    #                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))
    #     return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)

        # part1
        bx = self.b_conv0(x)
        bx = self.bn_b0(bx)

        lx = self.l_conv0(x)
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)

        x = self.relu(bx + lx)
        x = self.bl_init(x)
        x = self.bn_bl_init(x)
        x = self.relu(x)

        # part2
        x = self.layers(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.gappool(x)

        # x = x.view(x.size(0), -1) #
        # x = nn.Flatten(data=x)
        # flatten = Flatten(data=data, name='flat')  # now this is 2D
        # .get_output_shape(flatten, data=(2, 3, 4, 5))
        # {'flat_output': (2L, 60L)}

        # squeeze
        x = self.fc(x)
        # print(x.get_internals().list_outputs())
        # x = self.output(x)

        return x



# Nets ResNetV1
class ResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x

# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),

               # 50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 128, 256, 512]),

               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

# resnet_net_versions = [ResNetV1]
# resnet_block_versions = [{'bottle_neck': BottleneckV1}]

# resnet_net_versions = [ResNetV1, ResNetV2]
# resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
#                          {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]

# Constructor
def blget_resnet(version, num_layers, pretrained=False, ctx=cpu(),
               root='~/.mxnet/models', use_se=False, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    # 'bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    # resnet_class = resnet_net_versions[version-1]
    # block_class = resnet_block_versions[version-1][block_type]

    blresnet_class = BLResNetV1 #
    block_class = BottleneckV1
    alpha = 2
    beta = 4
    # block, layers, channels, classes=1000, thumbnail=False,
    #                  last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs
    net = blresnet_class(block_class, layers, channels, alpha, beta, **kwargs)
    """
    if pretrained:
        from .model_store import get_model_file
        if not use_se:
            net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx)
        else:
            net.load_parameters(get_model_file('se_resnet%d_v%d'%(num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    """
    return net

def blresnet50_v1(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return blget_resnet(1, 50, use_se=False, **kwargs)


_models = {'blresnet50_v1': blresnet50_v1}

def get_blmodel(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


