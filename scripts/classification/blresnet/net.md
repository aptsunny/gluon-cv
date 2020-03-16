```markdown
# pytorch
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)

```

```markdown
# mxnet
BLResNetV1(
  (features): HybridSequential(
    (0): Conv2D(3 -> 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
    (2): Activation(relu)
  )
  (relu): Activation(relu)
  (b_conv0): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn_b0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
  (l_conv0): Conv2D(64 -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn_l0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
  (l_conv1): Conv2D(32 -> 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn_l1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
  (l_conv2): Conv2D(32 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn_l2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
  (bl_init): Conv2D(64 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn_bl_init): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
  (layers): HybridSequential(
    (0): BLModule(
      (relu): Activation(relu)
      (big): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(64 -> 64, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (2): Activation(relu)
            (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (5): Activation(relu)
            (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
            (1): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
          )
        )
        (1): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(256 -> 64, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (2): Activation(relu)
            (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (5): Activation(relu)
            (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (8): Activation(relu)
          )
        )
      )
      (little): HybridSequential(
        (0): HybridSequential(
          (0): BottleneckV1(
            (body): HybridSequential(
              (0): Conv2D(64 -> 32, kernel_size=(1, 1), stride=(1, 1))
              (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
              (2): Activation(relu)
              (3): Conv2D(32 -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
              (5): Activation(relu)
              (6): Conv2D(32 -> 128, kernel_size=(1, 1), stride=(1, 1))
              (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
              (8): Activation(relu)
            )
            (downsample): HybridSequential(
              (0): Conv2D(64 -> 128, kernel_size=(1, 1), stride=(1, 1))
              (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            )
          )
        )
      )
      (little_e): HybridSequential(
        (0): Conv2D(128 -> 256, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
      )
      (fusion): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(256 -> 64, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (2): Activation(relu)
            (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (5): Activation(relu)
            (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
          )
        )
      )
    )
    (1): BLModule(
      (relu): Activation(relu)
      (big): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(256 -> 128, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
            (1): Conv2D(256 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          )
        )
        (1): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
        )
        (2): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
        )
      )
      (little): HybridSequential(
        (0): HybridSequential(
          (0): BottleneckV1(
            (body): HybridSequential(
              (0): Conv2D(256 -> 64, kernel_size=(1, 1), stride=(1, 1))
              (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
              (2): Activation(relu)
              (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
              (5): Activation(relu)
              (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
              (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
              (8): Activation(relu)
            )
            (downsample): HybridSequential(
            
            )
          )
        )
      )
      (little_e): HybridSequential(
        (0): Conv2D(256 -> 512, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
      )
      (fusion): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
          )
        )
      )
    )
    (2): BLModule(
      (relu): Activation(relu)
      (big): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 256, kernel_size=(1, 1), stride=(2, 2))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
            (1): Conv2D(512 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
          )
        )
        (1): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
        (2): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
        (3): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
        (4): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
      )
      (little): HybridSequential(
        (0): HybridSequential(
          (0): BottleneckV1(
            (body): HybridSequential(
              (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(1, 1))
              (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
              (2): Activation(relu)
              (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
              (5): Activation(relu)
              (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
              (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
              (8): Activation(relu)
            )
            (downsample): HybridSequential(
            
            )
          )
        )
      )
      (little_e): HybridSequential(
        (0): Conv2D(512 -> 1024, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
      )
      (fusion): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
          
          )
        )
      )
    )
    (3): HybridSequential(
      (0): BottleneckV1(
        (body): HybridSequential(
          (0): Conv2D(1024 -> 512, kernel_size=(1, 1), stride=(2, 2))
          (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (2): Activation(relu)
          (3): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (5): Activation(relu)
          (6): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
          (8): Activation(relu)
        )
        (downsample): HybridSequential(
          (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
          (1): Conv2D(1024 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
        )
      )
      (1): BottleneckV1(
        (body): HybridSequential(
          (0): Conv2D(2048 -> 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (2): Activation(relu)
          (3): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (5): Activation(relu)
          (6): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
          (8): Activation(relu)
        )
      )
      (2): BottleneckV1(
        (body): HybridSequential(
          (0): Conv2D(2048 -> 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (2): Activation(relu)
          (3): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (5): Activation(relu)
          (6): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
          (8): Activation(relu)
        )
      )
    )
    (4): GlobalAvgPool2D(size=(1, 1), stride=(1, 1), padding=(0, 0), ceil_mode=True, global_pool=True, pool_type=avg, layout=NCHW)
    (5): Flatten
  )
  (fc): Dense(2048 -> 1000, linear)
)

```

```markdown
# pytorch 

bLResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  
  (b_conv0): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn_b0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (l_conv0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn_l0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (l_conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn_l1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (l_conv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn_l2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bl_init): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn_bl_init): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  
  (layer1): bLModule(
    (relu): ReLU(inplace=True)
    (big): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): AvgPool2d(kernel_size=3, stride=2, padding=1)
          (1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (little): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (little_e): Sequential(
      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fusion): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
      )
    )
  )
  (layer2): bLModule(
    (relu): ReLU(inplace=True)
    (big): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): AvgPool2d(kernel_size=3, stride=2, padding=1)
          (1): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (little): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (little_e): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fusion): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
      )
    )
  )
  (layer3): bLModule(
    (relu): ReLU(inplace=True)
    (big): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): AvgPool2d(kernel_size=3, stride=2, padding=1)
          (1): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (little): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (little_e): Sequential(
      (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (fusion): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): AvgPool2d(kernel_size=3, stride=2, padding=1)
        (1): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (gappool): AdaptiveAvgPool2d(output_size=1)
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)

```

```markdown 3rd mxnet
BLResNetV1(
  (features): HybridSequential(
    (0): Conv2D(3 -> 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
    (2): Activation(relu)
    (3): Layer_0(
      (big_branch): HybridConcurrent(
        (0): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
      )
      (little_branch): HybridConcurrent(
        (0): Conv2D(64 -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
        (2): Activation(relu)
        (3): Conv2D(32 -> 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
        (5): Activation(relu)
        (6): Conv2D(32 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
      )
      (relu): Activation(relu)
      (fusion): HybridConcurrent(
        (0): Conv2D(64 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
        (2): Activation(relu)
      )
    )
    (4): BLModule(
      (big): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(64 -> 64, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (2): Activation(relu)
            (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (5): Activation(relu)
            (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
            (1): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
          )
        )
        (1): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(256 -> 64, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (2): Activation(relu)
            (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (5): Activation(relu)
            (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (8): Activation(relu)
          )
        )
      )
      (little): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(64 -> 32, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
            (2): Activation(relu)
            (3): Conv2D(32 -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
            (5): Activation(relu)
            (6): Conv2D(32 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): Conv2D(64 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
          )
        )
      )
      (little_e): HybridSequential(
        (0): Conv2D(128 -> 256, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
      )
      (relu): Activation(relu)
      (fusion): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(256 -> 64, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (2): Activation(relu)
            (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (5): Activation(relu)
            (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
          )
        )
      )
    )
    (5): BLModule(
      (big): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(256 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
            (1): Conv2D(256 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          )
        )
        (1): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
        )
        (2): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
        )
      )
      (little): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(256 -> 64, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (2): Activation(relu)
            (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
            (5): Activation(relu)
            (6): Conv2D(64 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
          
          )
        )
      )
      (little_e): HybridSequential(
        (0): Conv2D(256 -> 512, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
      )
      (relu): Activation(relu)
      (fusion): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
          )
        )
      )
    )
    (6): BLModule(
      (big): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
            (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
            (1): Conv2D(512 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
          )
        )
        (1): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
        (2): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
        (3): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
        (4): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
        )
      )
      (little): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(512 -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (2): Activation(relu)
            (3): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
            (5): Activation(relu)
            (6): Conv2D(128 -> 512, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
          
          )
        )
      )
      (little_e): HybridSequential(
        (0): Conv2D(512 -> 1024, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
      )
      (relu): Activation(relu)
      (fusion): HybridSequential(
        (0): BottleneckV1(
          (body): HybridSequential(
            (0): Conv2D(1024 -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (2): Activation(relu)
            (3): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
            (5): Activation(relu)
            (6): Conv2D(256 -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1024)
            (8): Activation(relu)
          )
          (downsample): HybridSequential(
          
          )
        )
      )
    )
    (7): HybridSequential(
      (0): BottleneckV1(
        (body): HybridSequential(
          (0): Conv2D(1024 -> 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (2): Activation(relu)
          (3): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (5): Activation(relu)
          (6): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
          (8): Activation(relu)
        )
        (downsample): HybridSequential(
          (0): AvgPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False, global_pool=False, pool_type=avg, layout=NCHW)
          (1): Conv2D(1024 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
        )
      )
      (1): BottleneckV1(
        (body): HybridSequential(
          (0): Conv2D(2048 -> 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (2): Activation(relu)
          (3): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (5): Activation(relu)
          (6): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
          (8): Activation(relu)
        )
      )
      (2): BottleneckV1(
        (body): HybridSequential(
          (0): Conv2D(2048 -> 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (2): Activation(relu)
          (3): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
          (5): Activation(relu)
          (6): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1))
          (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=2048)
          (8): Activation(relu)
        )
      )
    )
    (8): GlobalAvgPool2D(size=(1, 1), stride=(1, 1), padding=(0, 0), ceil_mode=True, global_pool=True, pool_type=avg, layout=NCHW)
    (9): Flatten
  )
  (fc): Dense(2048 -> 1000, linear)
)

```

```markdown
from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.nn import HybridConcurrent

__all__ = ['get_blmodel']

# reference: https://github.com/IBM/BigLittleNet

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

def blm_make_layer(block, inplanes, planes, blocks, stride=1, last_relu=True):
    downsample = nn.HybridSequential(prefix='')
    if stride != 1:
        downsample.add(nn.AvgPool2D(3, strides=2, padding=1))
    if inplanes != planes:
        downsample.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=inplanes))
        downsample.add(BatchNorm(in_channels=planes))
    layers = nn.HybridSequential(prefix='')
    if blocks == 1:
        layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
    else:
        layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
        for i in range(1, blocks):
            layers.add(block(planes, planes, last_relu=last_relu if i == blocks - 1 else True))
    return layers

def blr_make_layer(block, inplanes, planes, blocks, stride=1):
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

def make_b_layer(channel, norm_layer, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # bl part1 : input 112
        # 3*3 ,64, s2
        # self.b_conv0 = nn.Conv2D(channels[0], 3, 2, 1, use_bias=False, in_channels=channels[0])
        # self.bn_b0 = norm_layer(in_channels=channels[0])
        out.add(nn.Conv2D(channel, 3, 2, 1, use_bias=False, in_channels=channel))
        out.add(norm_layer(in_channels=channel))
    return out

def make_l_layer(channel, norm_layer, alpha,prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # 3*3 ,32; 3*3 ,32, s2; 1*1, 64
        # self.l_conv0 = nn.Conv2D(channels[0] // alpha, 3, 1, 1, use_bias=False, in_channels=channels[0])
        # self.bn_l0 = norm_layer(in_channels=channels[0] // alpha)
        # self.l_conv1 = nn.Conv2D(channels[0] // alpha, 3, 2, 1, use_bias=False, in_channels=channels[0] // alpha)
        # self.bn_l1 = norm_layer(in_channels=channels[0] // alpha)
        # self.l_conv2 = nn.Conv2D(channels[0], 1, 1, use_bias=False, in_channels=channels[0] // alpha)
        # self.bn_l2 = norm_layer(in_channels=channels[0])
        out.add(nn.Conv2D(channel // alpha, 3, 1, 1, use_bias=False, in_channels=channel))
        out.add(norm_layer(in_channels=channel // alpha))
        out.add(nn.Conv2D(channel // alpha, 3, 2, 1, use_bias=False, in_channels=channel // alpha))
        out.add(norm_layer(in_channels=channel // alpha))
        out.add(nn.Conv2D(channel, 1, 1, use_bias=False, in_channels=channel // alpha))
        out.add(norm_layer(in_channels=channel))
    return out

def make_bl_layer(channel, norm_layer, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        # input 56
        # self.relu = nn.Activation('relu')
        # self.bl_init = nn.Conv2D(channels[0], 1, 1, use_bias=False, in_channels=channels[0])
        # self.bn_bl_init = norm_layer(in_channels=channels[0])
        # self.relu = nn.Activation('relu')
        out.add(nn.Activation('relu'))
        out.add(nn.Conv2D(channel, 1, 1, use_bias=False, in_channels=channel))
        out.add(norm_layer(in_channels=channel))
        out.add(nn.Activation('relu'))
    return out



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

        self.downsample = downsample
        self.last_relu = last_relu

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
        self.hw = hw

        with self.name_scope():
            self.big = blm_make_layer(block, int_channels, out_channels, blocks - 1, 2, last_relu=False)
            self.little = blm_make_layer(block, int_channels, out_channels // alpha, max(1, blocks // beta - 1))

            self.little_e = nn.HybridSequential(prefix='')
            self.little_e.add(nn.Conv2D(out_channels, kernel_size=1, in_channels=out_channels // alpha))
            self.little_e.add(BatchNorm(in_channels=out_channels))

            self.relu = nn.Activation('relu')
            self.fusion = blm_make_layer(block, out_channels, out_channels, 1, stride=stride)

    def hybrid_forward(self, F, x):
        big = self.big(x)
        little = self.little(x)
        little = self.little_e(little)
        big = F.contrib.BilinearResize2D(data=big, height=self.hw, width=self.hw)
        out = self.relu(big + little)
        out = self.fusion(out)
        return out


class BLResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, alpha=2, beta=4, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BLResNetV1, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, in_channels=3))
                self.features.add(norm_layer(in_channels=channels[0]))
                self.features.add(nn.Activation('relu'))

                self.features.add(make_b_layer(channels[0], norm_layer, 'b_branch'))
                self.features.add(make_l_layer(channels[0], norm_layer, alpha, 'l_branch'))
                self.features.add(make_bl_layer(channels[0], norm_layer, 'make_bl_layer'))

                # input 56
                self.features.add(BLModule(block, channels[0], channels[0] * block.expansion, layers[0], alpha, beta, stride=2, hw = 56))
                # input 28
                self.features.add(BLModule(block, channels[0] * block.expansion, channels[1] * block.expansion, layers[1], alpha, beta, stride=2, hw = 28))
                # input 14
                self.features.add(BLModule(block, channels[1] * block.expansion, channels[2] * block.expansion, layers[2], alpha, beta, stride=1, hw =14))

                # input 14 1024(256*4) -> 2048(512*4)
                self.features.add(blr_make_layer(block, channels[2] * block.expansion, channels[3] * block.expansion, layers[3], stride=2))

                # input 7*7
                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Flatten())
                self.fc = nn.Dense(classes, in_units=channels[-1] * block.expansion)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        # bx = self.b_conv0(x)
        # bx = self.bn_b0(bx)
        #
        # lx = self.l_conv0(x)
        # lx = self.bn_l0(lx)
        # lx = self.relu(lx)
        # lx = self.l_conv1(lx)
        # lx = self.bn_l1(lx)
        # lx = self.relu(lx)
        # lx = self.l_conv2(lx)
        # lx = self.bn_l2(lx)
        #
        # x = self.relu(bx + lx)
        # x = self.bl_init(x)
        # x = self.bn_bl_init(x)
        # x = self.relu(x)
        # x = self.layers(x)
        x = self.fc(x)
        return x

# Specification
resnet_spec = {50: ('bottle_neck', [3, 4, 6, 3], [64, 128, 256, 512])}

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
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version

    blresnet_class = BLResNetV1
    block_class = BottleneckV1
    alpha = 2
    beta = 4
    net = blresnet_class(block_class, layers, channels, alpha, beta, **kwargs)
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



```