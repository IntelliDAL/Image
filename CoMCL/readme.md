# Image

| Paper | File |
| A Clinical-oriented Multiview Contrastive Learning Method for Disease Diagnosis in Low-quality Medical Images | CoMCL&Downstream Task |
| | |

## **Methodology**

[![model_structure]](https://github.com/IntelliDAL/Image/blob/main/images/model3.jpg)

## Requirements

- Python 3
- CUDA 11
- yaml
- PIL
- tqdm
- PyTorch=1.7.1
- torchvision

## Dataset

All the used dataset are publicly-accessible:

[EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)

[EyeQ](https://github.com/HzFu/EyeQ_enhancement)

[IDRiD](https://idrid.grand-challenge.org/)

[ChesXpert](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## Train

**Stage1**: Construct of four views' samples

V1:  Lesion patch

V2: Low quality lesion patch

V3: Healthy patch

V4: Low quality healthy patch  

Firstly, based on whether EyePACS images have lesion labels and EyeQ image quality labels, the images are divided into lesion images, healthy images, low-quality lesion images, and low-quality healthy images, respectively.

Using pre trained lesion detectors to detect lesion images, randomly cropping healthy images, and obtaining patch stacks from four different perspectives.



**Stage2**: Train

```
$ python train.py
```



## Evaluate

Fine-tuning the model in downstream tasks and conducting testing.

The dataset should be stored in the following file format:

```
eyepacs
|-- train
|   |-- 0
|   |   |-- 1.png
|   |   |-- 2.png
|   |   `-- 3.png
|   |-- 1
|   |   |-- 4.png
|   |   |-- 5.png
|   |   `-- 6.png
|   |-- 2
|   |   |-- 7.png
|   |   |-- 8.png
|   |   `-- 9.png
|   |-- 3
|   |   |-- 10.png
|   |   |-- 11.png
|   |   `-- 12.png
|   `-- 4
|       |-- 13.png
|       |-- 14.png
|       `-- 15.png
|-- val
|   |-- ...
|-- test
|   |-- ...
```

Code is avalible at the Downstream Task folder.

Execute the following commands:

```
$ python main.py -config='eyepacs.yaml'
```



## Acknowledgment

Thanks for the [Lesion_CL](https://github.com/YijinHuang/Lesion-based-Contrastive-Learning) for the lesion detection network and the implementation of models, [MoCo](https://github.com/facebookresearch/moco) for the contrastive loss.