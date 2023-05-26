# Image

|  Paper   | File  |
|  Lesion-aware Contrastive Learning for Diabetic Retinopathy Diagnosis  | Contrstive Learning&Downstream Task  |
|   |  |

##  **Methodology**

![model_structure](./images/model3.jpg)



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

[IDRiD](https://idrid.grand-challenge.org/)

[Messidor](https://www.adcis.net/en/third-party/messidor/)



## Train

Stage1: Construct of positive patch set and negative patch set



Stage2: Train the teacher model

```
$ python train.py
```

Stage3: Train the student model

```
$ python student_train.py
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

Execute the following commands:

```
$ python student_train.py -config='eyepacs.yaml'
```



## Acknowledgment

Thanks for the [Lesion_CL](https://github.com/YijinHuang/Lesion-based-Contrastive-Learning) for the lesion detection network and the implementation of models, [MoCo](https://github.com/facebookresearch/moco) for the contrastive loss. 
