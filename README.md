# BEV-LaneDet: a Simple and Effective 3D Lane Detection Baseline 
## Introduction
BEV-LaneDet is an efficient and robust monocular 3D lane detection system. First, we introduce the Virtual Camera, which unifies the intrinsic/extrinsic parameters of cameras mounted on different vehicles to ensure the consistency of the spatial relationship between cameras. It can effectively promote the learning process due to the unified visual space. Secondly, we propose a simple but efficient 3D lane representation called Key-Points Representation. This module is more suitable for representing the complicated and diverse 3D lane structures. Finally, we present a lightweight and chip-friendly spatial transformation module called Spatial Transformation Pyramid to transform multi-scale front view features into BEV features.  Experimental results demonstrate that our
work outperforms the state-of-the-art approaches in terms of F-Score, being 10.6% higher on the OpenLane dataset and 5.9% higher on the Apollo 3D synthetic dataset, with a speed of 185 FPS. Our paper has been accepted by cvpr2023 [arxiv](https://arxiv.org/abs/2210.06006).


- [Get Started](#getstart)
- [Benchmark](#benchmark)
- [Visualization](#visualization)


## <span id="getstart">Get Started</span>

### Installation
- To run our code, make sure you are using a machine with at least one GPU.
- Setup the enviroment 
```
pip install -r requirements.txt
```
### Training and evaluation on OpenLane
- Please refer to [OpenLane](https://github.com/OpenPerceptionX/OpenLane) for downloading OpenLane Dataset. For example: download OpenLane dataset to /dataset/openlane

- How to train:
    1. Please modify the configuration in the /tools/openlane_config.py
    2. Execute the following code:
```
cd tools
python3 train_openlane.py
```
- How to evaluation:
    1. Please modify the configuration in the /tools/val_openlane.py
    2. Execute the following code:
```
cd tools
python val_openlane.py
```

### Training and evaluation on Apollo 3D Lane Synthetic
- Please refer to [Apollo 3D Lane Synthetic](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset) for downloading Apollo 3D Lane Synthetic Dataset. For example: download OpenLane dataset to /dataset/apollo

- How to train:
    1. Please modify the configuration in the /tools/apollo_config.py
    2. Execute the following code:
```
cd tools
python3 train_apollo.py
```
- How to evaluation:
    1. Please modify the configuration in the /tools/val_apollo.py
    2. Execute the following code:
```
cd tools
python val_apollo.py
```

## <span id="benchmark">Benchmark</span>

### Results of different models on OpenLane dataset

| Method | F-Score | X error  near | X error far | Z error near | Z error far|GFLOPs | TensorRT | PyTorch  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
| Gen-LaneNet | 29.7 |0.309|0.877|0.16|0.75| 34 | – | 54FPS  |
| PersFormer  | 47.8 |0.322|0.778|0.213|0.681| 143 | – | 21FPS |
| Ours  | 58.4 | 0.309 |0.659|0.244|0.631| 53| 185FPS | 102FPS| 

### Results of different models on Apollo 3D Lane Synthetic (Balanced Scence)
| Method | F-Score | X error  near | X error far | Z error near | Z error far|
| ---- | ---- | ---- | ---- | ---- | ---- |
| 3D-LaneNet | 86.4 |0.068|0.477|0.015|0.202|
| Gen-LaneNet | 88.1 |0.061|0.486|0.012|0.214|
| CLGO | 91.9 |0.061|0.361|0.029|0.25|
| PersFormer  | 92.9 |0.054|0.356|0.01|0.234|
| Ours  | 98.7 | 0.016 |0.242|0.02|0.216|


### Virtual Camera

CPU implementation is here: [Virutal Camera on CPU](./csrc/README.md)

|  Hardware   | Single-thread  | Multi-thread |
|  ----  | ----  | ----|
| Apple M1  | 1.5ms | 0.5ms |
| Intel Xeon Platinum 8163 @ 2.5 GHz  |5.5ms  | 1.2ms|
| Nv V100| - | TODO |


## <span id="visualization">Visualization</span>
### OpenLane
Full-length (10 mins) video of OpenLane is here: [Video](./virtualization/ol.mp4) or you can find in https://www.youtube.com/watch?v=Mqh0N2cOctM

![OpenLane](./visualization/ol.gif)

### Apollo 3D Lane Synthetic
You can watch video of Apollo 3D Lane Synthetic in https://www.youtube.com/watch?v=WC36c4wO_QM
