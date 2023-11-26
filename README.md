# [Multiple Domain Experts Collaborative Learning: Multi-Source Domain Generalization For Peron Re-Identification](https://arxiv.org/pdf/2105.12355.pdf)


## Requirements


### Prepare Pre-trained Models
```shell
mkdir logs && cd logs
# download the the pretrained model and save it to logs
```
The file tree should be
```
logs
└── pretrained
    ├── resnet50_ibn_a.pth.tar
    └── resnet101_ibn_a.pth.tar
```

The pretrained models for ibn should be found in [IBN-Net](https://github.com/XingangPan/IBN-Net)


### Training

```shell
# Note, this script is for  my slurm environment 
# and you should modify it to adapt you own training environment
./scripts/multi_source/train_to_market.sh resnet_ibn50a VI_Face_1080TI 8 ${description} ${port}
```


