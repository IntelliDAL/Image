import os,re
from unittest.mock import patch
import pickle
import random
from unicodedata import name
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from loader import pil_loader


def auto_statistics(data_path, data_index, batch_size, num_workers, input_size):
    print('Calculating mean and std of training set for data normalization.')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    if data_index:
        train_set = pickle.load(open(data_index, 'rb'))['train']
        train_dataset = DatasetFromDict(train_set, transform=transform)
    else:
        train_path = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(train_path, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    return channel_mean.tolist(), channel_std.tolist()


def generate_dataset_from_pickle(data_path, pkl, data_config, transform):
    data = pickle.load(open(pkl, 'rb'))

    train_dataset = LesionPatchGenerator(data_path, data['train']['healthy'], data['train']['unhealthy'], data_config['patch_size'], transform)
    val_dataset = LesionPatchGenerator(data_path, None, data['val'], data_config['patch_size'], transform)

    return train_dataset,val_dataset

# 自己写了个函数
def generate_dataset_from_folder(data_path, train_transform, test_transform):
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')

    train_dataset = CustomizedImageFolder(train_path, train_transform, loader=pil_loader)
    val_dataset = CustomizedImageFolder(val_path, test_transform, loader=pil_loader)

    return train_dataset, val_dataset

def data_transforms(data_config,data_path,data_index):
    data_aug = data_config['data_augmentation']
    patch_size = data_config['patch_size']
    input_size = data_config['input_size']
    mean, std = data_config['mean'], data_config['std']

    if mean == 'auto' or std == 'auto':
        mean, std = auto_statistics(
            data_path,
            data_index,
            30,  # batchsize
            8,   #num_worker
            data_config['input_size']
        )
        data_config['mean'] = mean
        data_config['std'] = std

    print("The mean value of this dataset is ",mean)
    print("The std value of this dataset is ",std)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=data_aug['brightness'],
            contrast=data_aug['contrast'],
            saturation=data_aug['saturation'],
            hue=data_aug['hue']
        ),
        transforms.RandomResizedCrop(
            size=(patch_size, patch_size),
            scale=data_aug['scale'],
            ratio=data_aug['ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['degrees'],
            translate=data_aug['translate']
        ),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform


class LesionPatchGenerator(Dataset):
    def __init__(self, data_path, healthy_imgs, lesion_imgs, patch_size, transform=None):
        super(LesionPatchGenerator, self).__init__()
        self.healthy_imgs = []
        self.lesion_imgs = [] 
        if healthy_imgs:
            for name, lesions in healthy_imgs.items():        #name for img_name ex: 1_left.jpeg
                
                if len(name) > 16: 
                    name = re.split('/|\\\\',name)[-3:]  #匹配路径字符
                    name = '/'.join(name)
                else:
                    print('Something wrong with lesion.pkl!')
                path = os.path.join(data_path, name)   
                self.healthy_imgs.append((path, lesions))

        for name, lesions in lesion_imgs.items():        #name for img_name ex: 1_left.jpeg
            
            if len(name) > 16: 
                name = re.split('/|\\\\',name)[-3:]  #匹配路径字符
                name = '/'.join(name)
            else:
                print('Something wrong with lesion.pkl!')
            path = os.path.join(data_path, name)   
            self.lesion_imgs.append((path, lesions))

        self.patch_size = patch_size
        self.transform = transform
        self.m = len(healthy_imgs) if healthy_imgs else 0
        self.n = len(lesion_imgs)
        

    def __len__(self):
        return len(self.lesion_imgs)

    def __getitem__(self, index):    #可迭代对象
        
        img_path, lesions = self.lesion_imgs[index]
        if self.m == 0:
            img = self.pil_loader(img_path)

            bbox = random.choice(lesions)
            patch_1 = self.generate_patch(img, bbox)
            patch_2 = self.generate_patch(img, bbox)

            if self.transform is not None:
                patch_1 = self.transform(patch_1)
                patch_2 = self.transform(patch_2)
            # print(patch_1.size())
            return patch_1, patch_2
        else:
            healthy_loader = []
            
            h_l_rate = self.m//self.n  #healthy//lesion rate
            for i in range(h_l_rate):
                healthy_loader.append((self.healthy_imgs[index+i*self.n]))

            img = self.pil_loader(img_path)

            # print("img_path:",img_path,"lesions:",lesions)
            bbox = random.choice(lesions)
            patch_1 = self.generate_patch(img, bbox)
            patch_2 = self.generate_patch(img, bbox)

            if self.transform is not None:
                patch_1 = self.transform(patch_1)
                patch_2 = self.transform(patch_2)
            # print(patch_1.size())

            healthy_patch = torch.Size([])
            for img_path,patch in healthy_loader:
                img = self.pil_loader(img_path)

                # print("img_path:",img_path,"lesions:",lesions)
                bbox = random.choice(patch)
                p1 = self.generate_patch(img, bbox)

                if self.transform is not None:   #加上和lesion相同的数据增强
                    p1 = self.transform(p1)
                    p1 = p1.unsqueeze(0)
                    if healthy_patch == torch.Size([]):
                        healthy_patch = p1
                    else:
                        healthy_patch = torch.cat((healthy_patch,p1),dim=0)
            
            return patch_1, patch_2, healthy_patch

    def generate_patch(self, img, bbox):
        w, h = img.size
        #i = 0
        # if i == 0 and len(bbox) == 4 :
        #     print("bbox:",bbox)
        # if len(bbox) != 4:
        #     print("bbox:",bbox)
        #     print("img_size:",img.size)
        x1, y1, x2, y2 = bbox
        # b_w = random.randint(32, 128)
        # b_h = random.randint(32, 128)
        # x1 = random.randint(10, w-b_w)
        # y1 = random.randint(10, h-b_h)
        # x2 = x1 + b_w
        # y2 = y1 + b_h
        b_w = bbox[2] - bbox[0]
        b_h = bbox[3] - bbox[1]

        x_space = self.patch_size - b_w
        if x1 < w - b_w:
            l_shift = int(random.random() * min(x1, x_space))
            new_x1 = x1 - l_shift
            new_x2 = x2 + (x_space - l_shift)
        else:
            r_shift = int(random.random() * min(w - b_w, x_space))
            new_x1 = x1 - (x_space - r_shift)
            new_x2 = x2 + r_shift

        y_space = self.patch_size - b_h
        if y1 < h - b_h:
            t_shift = int(random.random() * min(y1, y_space))
            new_y1 = y1 - t_shift
            new_y2 = y2 + (y_space - t_shift)
        else:
            d_shift = int(random.random() * min(h - b_h, y_space))
            new_y1 = y1 - (y_space - d_shift)
            new_y2 = y2 + d_shift

        patch = img.crop((new_x1, new_y1, new_x2, new_y2))
        return patch

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class CustomizedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        super(CustomizedImageFolder, self).__init__(root, transform, target_transform, loader=loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_1 = self.transform(sample)
            sample_2 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample_1, sample_2

class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.pil_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class TwoCropTransform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
