## ↳ Stargazers
[![Stargazers repo roster for @zqx951102/DFSI](https://reporoster.com/stars/zqx951102/DFSI)](https://github.com/zqx951102/DFSI/stargazers)

## ↳ Forkers
[![Forkers repo roster for @zqx951102/DFSI](https://reporoster.com/forks/zqx951102/DFSI)](https://github.com/zqx951102/DFSI/network/members)


![Python >=3.5](https://img.shields.io/badge/Python->=3.5-yellow.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
<img src="doc/0.jpg" width="300" height="100" alt="图片名称"/>
</div>

This repository hosts the source code of our paper: [Dynamic Frequency Selection and Spatial Interaction Fusion for Robust Person Search](https://doi.org/10.1016/j.inffus.2025.103314). 


Challenges and Motivation:
<div align="center">
<img src="./doc/1.jpg" width="600" height="500"/>
</div>


The network structure:

<div align="center">
<img src="./doc/2.jpg" width="800" height="350"/>
</div>


****
## :fire: NEWS :fire:

- [2025.5.07] **📣Accept**
  
- [2025.4.29] **📣We received comments requiring minor revisions from the Journal of Information Fusion!**

- [2025.4.01] **📣We received comments requiring major revisions from the Journal of Information Fusion!**

- [2025.2.25] **📣We submitted our paper to Information Fusion!**
  
- [2025.2.23] **📣We released the code.**



## Installation

Run `pip install -r requirements.txt` in the root directory of the project.


## Quick Start

Let's say `$ROOT` is the root directory.

1. Download [CUHK-SYSU](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) and [PRW](https://drive.google.com/file/d/1Pz81MP8ePlNZMLm_P-AIkUERyOAXWOTV/view?usp=sharing) datasets, and unzip them to `$ROOT/data`

```
data
├── CUHK-SYSU
├── PRW
```

2. Following the link in the above table, download our pretrained model to anywhere you like, e.g., `$ROOT/exp_cuhk`

Performance profile:
<div align="center">
  
| Dataset   | Name          | ASTD                                                        |
| --------- | ------------- | ------------------------------------------------------------ |
| CUHK-SYSU | ckpt_epoch_12.pth  | [model](https://drive.google.com/file/d/17mDmKqheoOtlb7iRLqFK7yV1H-DWyEJO/view?usp=sharing)|
| PRW       | ckpt_epoch_13.pth  | [model](https://drive.google.com/file/d/17-rU8ep-bA1NN55hxHErfKPeW91eG0Zv/view?usp=sharing) |

</div>

Please see the Demo photo:

<div align="center">
<img src="./doc/query.jpg" width="600" height="450"/>
</div>


**Note**: At present, our script only supports single GPU training, but distributed training will be also supported in future. By default, the batch size and the learning rate during training are set to 3 and 0.003 respectively, which requires about 28GB of GPU memory. If your GPU cannot provide the required memory, try smaller batch size and learning rate (*performance may degrade*). Specifically, your setting should follow the [*Linear Scaling Rule*](https://arxiv.org/abs/1706.02677): When the minibatch size is multiplied by k, multiply the learning rate by k. For example:



## Training
```
CUHK:
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/cuhk_sysu_resnet.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/cuhk_sysu_convnext.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/cuhk_sysu_solider.yaml

PRW：
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/prw_resnet.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/prw_convnext.yaml
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/prw_solider.yaml


if out of memory, modify this：
./configs/cuhk_sysu_convnext.yaml    BATCH_SIZE: 3  #5  

Before running, you need to modify the addresses in these two files and link them to the directory where your data is located.
./configs/_path_cuhk_sysu.yaml
./configs/_path_prw.yaml
```

**Tip**: If the training process stops unexpectedly, you can resume from the specified checkpoint.

```
python train.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /path/to/your/checkpoint
```

**Note**: You need to modify the base_dir address in the file ./configs/_path_solider_weights.yaml.
like this：
<div align="center">
<img src="./doc/8.jpg" />
</div>

<div align="center">
  
| Name          | Address                                                       |
| ------------- | ------------------------------------------------------------ |
| swin_base.pth  |[model](https://drive.google.com/file/d/1uh7tO34tMf73MJfFqyFEGx42UBktTbZU/view?usp=drive_link)|
| swin_small.pth  |[model](https://drive.google.com/file/d/11uYzAkAv_8EvqpsKyK6W6ZM76UlnFbmo/view?usp=sharing)|
| swin_tiny.pth  |[model](https://drive.google.com/file/d/12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8/view?usp=drive_link)|

</div>


## Comparison with SOTA:

<div align="center">
<img src="./doc/7.jpg" width="640" height="590"/>
</div>


## Evaluation of different gallery size:

<div align="center">
<img src="./doc/4.jpg" width="700" height="360"/>
</div>
Remember that when you test other code, you still need to set it to 100！！

## Qualitative Results on CUHK-SYSU:
<div align="center">
<img src="./doc/5.jpg" width="700" height="580"/>
</div>


## Qualitative Results on PRW:
<div align="center">
<img src="./doc/6.jpg" width="700" height="420"/>
</div>

## Acknowledgment
Thanks to the authors of the following repos for their code, which was integral in this project:
- [SeqNet](https://github.com/serend1p1ty/SeqNet)
- [NAE](https://github.com/dichen-cd/NAE4PS)
- [GFN](https://github.com/LukeJaffe/GFN)
- [torchvision](https://github.com/pytorch/vision)

## Pull Request

Pull request is welcomed! Before submitting a PR, **DO NOT** forget to run `./dev/linter.sh` that provides syntax checking and code style optimation.


## Citation
If you find this code useful for your research, please cite our paper
```
@article{zhang2025dynamic,
  title={Dynamic frequency selection and spatial interaction fusion for robust person search},
  author={Zhang, Qixian and Miao, Duoqian and Zhang, Qi and Zhao, Cairong and Zhang, Hongyun and Sun, Ye and Wang, Ruizhi},
  journal={Information Fusion},
  volume={124},
  pages={103314},
  year={2025},
  publisher={Elsevier}
}
```
```
@article{zhang2024learning,
  title={Learning adaptive shift and task decoupling for discriminative one-step person search},
  author={Zhang, Qixian and Miao, Duoqian and Zhang, Qi and Wang, Changwei and Li, Yanping and Zhang, Hongyun and Zhao, Cairong},
  journal={Knowledge-Based Systems},
  volume={304},
  pages={112483},
  year={2024},
  publisher={Elsevier}
}
```
```
@article{zhang2024attentive,
  title={Attentive multi-granularity perception network for person search},
  author={Zhang, Qixian and Wu, Jun and Miao, Duoqian and Zhao, Cairong and Zhang, Qi},
  journal={Information Sciences},
  volume={681},
  pages={121191},
  year={2024},
  publisher={Elsevier}
}
```
```
@inproceedings{li2021sequential,
  title={Sequential End-to-end Network for Efficient Person Search},
  author={Li, Zhengjia and Miao, Duoqian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2011--2019},
  year={2021}
}
```

## Contact
If you have any question, please feel free to contact us. E-mail: [zhangqx@tongji.edu.cn](mailto:zhangqx@tongji.edu.cn) 
