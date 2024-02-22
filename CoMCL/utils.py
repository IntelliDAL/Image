import os
import pickle
from tabnanny import check

#import apex    用不到多个显卡，安装apex有问题暂时注释掉
import torch
from torchvision import datasets, transforms

from modules import ContrastiveModel,MoCo
from data import generate_dataset_from_pickle, generate_dataset_from_folder,DatasetFromDict, data_transforms


def generate_dataset(data_config, data_path, data_index=None):
    transform = data_transforms(data_config,data_path,data_index)
    # datasets = generate_dataset_from_pickle(data_path, data_index, data_config, transform)
    datasets = generate_dataset_from_folder(data_path, transform, transform)

    return datasets


def generate_model(network, net_config, device, pretrained=True, checkpoint=None):
    if network not in net_config.keys():
        raise NotImplementedError('Not implemented network.')

    model = MoCo(
        net_config[network],
        pretrained,
        head='mlp',
        dim_in=2048,
        feat_dim=128
    ) .to(device)

    if checkpoint:
        #在train.py中保存的是model，所以checkpoint直接load
        # model = torch.load(checkpoint)
        # 如果是只有resnet，这里strict设为False
        #后面同意保存的参数
        pretrained_model = torch.load(checkpoint)
        pretrained_model = pretrained_model['model']
        model.load_state_dict(pretrained_model, strict=True)
        model.cuda()
          
        print('Load weights form {}'.format(checkpoint))

    if device == 'cuda' and torch.cuda.device_count() > 1:
        pass    #没有安装apex，也没有多的显卡暂时注释掉
        #model = apex.parallel.convert_syncbn_model(model)
        #model = torch.nn.DataParallel(model)

    return model

def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


def show_config(configs):
    for name, config in configs.items():
        print('====={}====='.format(name))
        print_config(config)
        print('=' * (len(name) + 10))
        print()


def print_config(config, indentation=''):
    for key, value in config.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indentation, key))
            print_config(value, indentation + '    ')
        else:
            print('{}{}: {}'.format(indentation, key, value))


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
