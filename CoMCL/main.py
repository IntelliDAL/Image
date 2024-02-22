import os
import random
import shutil

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from torchvision import models
from config import *
from resnet import BasicBlock
from student_train import student_train
from train import evaluate, train
from utils import generate_dataset, generate_model, show_config


def main():
    # print configuration
    '''
    show_config({
        'BASIC CONFIG': BASIC_CONFIG,
        'DATA CONFIG': DATA_CONFIG,
        'TRAIN CONFIG': TRAIN_CONFIG
    })
    '''

    # reproducibility
    seed = BASIC_CONFIG['random_seed']
    set_random_seed(seed)

    # create folder
    save_path = BASIC_CONFIG['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # build model
    device = BASIC_CONFIG['device']
    
    # t_model = generate_model(
    #     BASIC_CONFIG['t_network'],
    #     NET_CONFIG,
    #     device,
    #     BASIC_CONFIG['pretrained'],
    #     BASIC_CONFIG['t_model_path']
    # ) 

    network = BASIC_CONFIG['network']
    
    model = generate_model(
        network,
        NET_CONFIG,
        device,
        BASIC_CONFIG['pretrained'],
        BASIC_CONFIG['checkpoint']
    )

    # create dataset
    if not BASIC_CONFIG['data_index']:
        train_dataset,val_dataset = generate_dataset(
            DATA_CONFIG,
            BASIC_CONFIG['data_path']
        )
    else:
        train_dataset,val_dataset = generate_dataset(
            DATA_CONFIG,
            BASIC_CONFIG['data_path'],
            BASIC_CONFIG['data_index']
        )

    checkpoint = BASIC_CONFIG['checkpoint']
    # create logger
    record_path = BASIC_CONFIG['record_path']
    if os.path.exists(record_path):
        shutil.rmtree(record_path)
    logger = SummaryWriter(BASIC_CONFIG['record_path'])

    # create estimator and then train
    train(
        model=model,
        train_config=TRAIN_CONFIG,
        data_config=DATA_CONFIG,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        save_path=save_path,
        device=device,
        checkpoint=checkpoint,
        logger=logger
    )
    # student_train(
    #     t_model=t_model,
    #     model=model,
    #     train_config=TRAIN_CONFIG,
    #     data_config=DATA_CONFIG,
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     save_path=save_path,
    #     device=device,
    #     cl_lamda=BASIC_CONFIG['cl_lamda'],
    #     kd_lamda=BASIC_CONFIG['kd_lamda'],
    #     checkpoint=checkpoint,
    #     logger=logger
    # )

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
