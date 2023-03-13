import os.path
from options.train_options import TrainOptions
from torchvision.utils import save_image
from networks import define_G, define_D, GANLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from util.util import load_checkpoint, save_checkpoint
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self, root_A, transform=None):
        super(MyDataset, self).__init__()
        self.transform = transform
        A_images = os.listdir(root_A)
        
        self.A_images = [os.path.join(root_A, img) for img in A_images]
        self.length_dataset = len(self.A_images)
        
    def __len__(self):
        return self.length_dataset
        
    def add_elements_to_A(self, new_root_A):
        new_A_images = os.listdir(new_root_A)
        
        for img in new_A_images:
          self.A_images.append(os.path.join(new_root_A,img))
        self.length_dataset = len(self.A_images)
        
    
    def __getitem__(self, index):
        name = self.A_images[index].split('/')[-1]
        A_img = self.A_images[index]

        A_img = np.array(Image.open(A_img).convert("RGB"))

        if self.transform:
            augmentataions = self.transform(image = A_img)
            A_img = augmentataions["image"]

        return name, A_img


def test_fn(netG_B, dataloader):
    netG_B.eval()
    # Generators: netG_A: B -> A;netG_B: A -> B.
    # Discriminators: netD_A: netG_A(B)vs.A;netD_B: netG_B(A)vs.B.
    loop = tqdm(dataloader, leave=True)

    for idx, (name, real_A) in enumerate(loop):
        real_A = real_A.to(device)

        fake_B = netG_B(real_A)
        fake_B = fake_B * 0.5 + 0.5

        for index in range(len(fake_B)):
            img_name = name[index]
            img = fake_B[index]
            plt.imsave("./output/cluster6/"+img_name, np.array(img.detach().cpu().permute(1, 2, 0)))




if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    # Generators: G_A: B -> A; G_B: A -> B.
    netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids).to(device)
    netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids).to(device)

    optimizer_G = optim.Adam(
        list(netG_A.parameters()) + list(netG_B.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
    )
    LOAD_MODEL = True
    if LOAD_MODEL:
        load_checkpoint(
            opt.checkpoint_gen_B, netG_B, optimizer_G, opt.lr,
        )

    test_transform = A.Compose(
        [
            A.Resize(width=256, height=256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
    )

    
    fold_list = os.listdir(opt.dataroot)
    fold_list = [fold for fold in fold_list if os.path.isdir(os.path.join(opt.dataroot, fold))]
    fold_list.remove("cluster4")
    fold_list.remove("cluster_1")

    division_fold = "/home/ubuntu/IQA_EYE/LocalAggregation-Pytorch-master/result/cluster6/cluster4/"
    division_list = os.listdir(division_fold)
    division_list = [fold for fold in division_list if os.path.isdir(os.path.join(division_fold, fold))]
    division_list.remove("cluster_1")
    
    
    test_dataset = None
    
    for fold in division_list:
        if test_dataset == None:
            test_dataset = MyDataset(os.path.join(division_fold, fold),
                                      transform=test_transform)
        else:
            test_dataset.add_elements_to_A(os.path.join(division_fold, fold))

    for fold in fold_list:
        if test_dataset == None:
            test_dataset = MyDataset(os.path.join(opt.dataroot, fold),
                                      transform=test_transform)
        else:
            test_dataset.add_elements_to_A(os.path.join(opt.dataroot, fold))
    
    print(len(test_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    test_fn(netG_B, test_dataloader)

