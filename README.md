# Scene Text Recognition with Vision-language Circular Refinement

Implementation of VLCR for blind review.

A model for scene text recognition. The code is adapted from MGP-STR (https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR). Thanks. <br>


### Install requirements
- This work was tested with PyTorch 1.11.0, CUDA 11.7, python 3.9 and Ubuntu20.04. You can install the packages needed with the folloing commands, but the package "wand" and "ImageMagick" for data augmentation may require extra installation in your environment. I built the project from source code in mine and I suppose its version might lead to some fixable issue. <br>

```
pip3 install -r requirements.txt
```

### Dataset

Download lmdb dataset from [Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://github.com/FangShancheng/ABINet).

- Training datasets

    1. [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/) (MJ): 
        - Use `tools/create_lmdb_dataset.py` to convert images into LMDB dataset
        - [LMDB dataset BaiduNetdisk(passwd:n23k)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)
    2. [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) (ST):
        - Use `tools/crop_by_word_bb.py` to crop images from original [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) dataset, and convert images into LMDB dataset by `tools/create_lmdb_dataset.py`
        - [LMDB dataset BaiduNetdisk(passwd:n23k)](https://pan.baidu.com/s/1mgnTiyoR8f6Cm655rFI4HQ)

- Evaluation datasets
  LMDB datasets can be downloaded from [BaiduNetdisk(passwd:1dbv)](https://pan.baidu.com/s/1RUg3Akwp7n8kZYJ55rU5LQ), [GoogleDrive](https://drive.google.com/file/d/1dTI0ipu14Q1uuK4s4z32DqbqF3dJPdkk/view?usp=sharing).<br>
    1. ICDAR 2013 (IC13)
    2. ICDAR 2015 (IC15)
    3. IIIT5K Words (IIIT)
    4. Street View Text (SVT)
    5. Street View Text-Perspective (SVTP)
    6. CUTE80 (CUTE)

- The structure of data folder as below.
```
data
├── evaluation
│   ├── CUTE80
│   ├── IC13_857
│   ├── IC15_1811
│   ├── IIIT5k_3000
│   ├── SVT
│   └── SVTP
├── training
│   ├── MJ
│   │   ├── MJ_test
│   │   ├── MJ_train
│   │   └── MJ_valid
│   └── ST
```
At this time, training datasets and evaluation datasets are LMDB datasets <br>


### Models Weights

We propose trained models of two scales: tiny(./vlcr_model_weight/tiny.pth) and base(./vlcr_model_weight/base.pth).

Tiny model: https://pan.baidu.com/s/14S6yAOYWP6T8waILxVdViw?pwd=xi9n
Base model: https://pan.baidu.com/s/1b3Z3LYVKZO4z2waS2AfUwQ?pwd=rynw

Please download them and put them in ./vlcr/vlcr_model_weight/ if you want to use them.

The average accuracy in six benchmarks is perspectively 91.1% and 93.6%.


### Train

Please edit path-related and GPU-related settings in the sh files first.

VLCR-base

```
./vlcr_sh/train_base.sh
```

VLCR-tiny

```
./vlcr_sh/train_tiny.sh
```


### Test

Please edit path-related and GPU-related settings in the sh files first and make sure you have downloaded tiny.pth and base.pth.

VLCR-base
```
./vlcr_sh/eval/eval_base.sh
```

VLCR-tiny
```
./vlcr_sh/eval/eval_tiny.sh
```



## *License*

VLCR is released under the terms of the [Apache License, Version 2.0](LICENSE).

```
VLCR is an algorithm for scene text recognition and the code and models herein created by the authors can only be used for research purpose.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```