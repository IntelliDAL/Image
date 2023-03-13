import os.path
from options.train_options import TrainOptions
from dataset import MyDataset, FundusDataset
from torchvision.utils import save_image
from networks import define_G, define_D, GANLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim
from util.util import load_checkpoint, save_checkpoint, feature_sample
import albumentations as A
from albumentations.pytorch import ToTensorV2
from util.icc_loss import ICCLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(netD_A, netD_B, netG_A, netG_B, dataloader, optimizer_D, optimizer_G,
             criterion_contrastive, criterionGAN, criterionCycle, criterionIdt, d_scaler, g_scaler):

    # Generators: netG_A: B -> A;netG_B: A -> B.
    # Discriminators: netD_A: netG_A(B)vs.A;netD_B: netG_B(A)vs.B.
    L_reals = 0
    L_fakes = 0
    loop = tqdm(dataloader, leave=True)

    for idx, (real_A, real_B, A_label) in enumerate(loop):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # Train Discriminator A and B
        with torch.cuda.amp.autocast():
            fake_A, emb_real_B = netG_A(real_B)
            pred_real = netD_A(real_A)
            pred_fake = netD_A(fake_A.detach())
            D_A_real_loss = criterionGAN(pred_real, True)
            D_A_fake_loss = criterionGAN(pred_fake, False)
            D_A_loss = D_A_real_loss + D_A_fake_loss
            
            L_reals += pred_real.mean().item()
            L_fakes += pred_fake.mean().item()

            fake_B, emb_real_A = netG_B(real_A)
            pred_real = netD_B(real_B)
            pred_fake = netD_B(fake_B.detach())
            D_B_real_loss = criterionGAN(pred_real, True)
            D_B_fake_loss = criterionGAN(pred_fake, False)
            D_B_loss = D_B_real_loss + D_B_fake_loss

            D_loss = (D_A_loss + D_B_loss) / 2

        optimizer_D.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(optimizer_D)
        d_scaler.update()

        # Train Generator A and B
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            pred_fakeA = netD_A(fake_A)
            pred_fakeB = netD_B(fake_B)
            loss_G_A = criterionGAN(pred_fakeA, True)
            loss_G_B = criterionGAN(pred_fakeB, True)

            # cycle loss
            cycle_A, emb_fake_B = netG_A(fake_B, A_label)
            cycle_B, emb_fake_A = netG_B(fake_A)
            emb_fake_A = feature_sample(emb_fake_A)
            emb_fake_B = feature_sample(emb_fake_B)
            emb_real_A = feature_sample(emb_real_A)
            emb_real_B = feature_sample(emb_real_B)

            cycle_A_loss = criterionCycle(real_A, cycle_A) + criterion_contrastive(emb_real_A, emb_fake_B).mean()
            cycle_B_loss = criterionCycle(real_B, cycle_B) + criterion_contrastive(emb_real_B, emb_fake_A).mean()

            # identity loss
            identity_A, _ = netG_A(real_A)
            identity_B, _ = netG_B(real_B)
            identity_A_loss = criterionIdt(real_A, identity_A)
            identity_B_loss = criterionIdt(real_B, identity_B) 

            # add all together
            G_loss = (
                loss_G_A
                + loss_G_B
                + cycle_B_loss * opt.lambda_cycle
                + cycle_A_loss * opt.lambda_cycle
                + identity_B_loss * opt.lambda_identity
                + identity_A_loss * opt.lambda_identity
            )

        optimizer_G.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(optimizer_G)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_B * 0.5 + 0.5, f"saved_images/fake_high_{idx}.png")
            save_image(fake_A * 0.5 + 0.5, f"saved_images/fake_low_{idx}.png")

        loop.set_postfix(L_real=L_reals / (idx + 1), L_fake=L_fakes / (idx + 1))


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    # Generators: G_A: B -> A; G_B: A -> B.
    netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids).to(device)
    netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids).to(device)

    # Discriminators: D_A: G_A(B) vs.A; D_B: G_B(A) vs.B.
    netD_A = define_D(opt.output_nc, opt.ndf, opt.netD,
                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids).to(device)
    netD_B = define_D(opt.input_nc, opt.ndf, opt.netD,
                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids).to(device)

    # Optimizers of Generators and Discriminators
    optimizer_D = optim.Adam(
        list(netD_A.parameters()) + list(netD_B.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
    )

    optimizer_G = optim.Adam(
        list(netG_A.parameters()) + list(netG_B.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
    )

    criterionGAN = GANLoss(opt.gan_mode).to(device)
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()
    criterion_contrastive = ICCLoss(opt.batch_size // len(opt.gpu_ids), 0.07)

    if opt.load_model:
        load_checkpoint(
            opt.checkpoint_gen_A, netG_A, optimizer_G, opt.lr,
        )
        load_checkpoint(
            opt.checkpoint_gen_B, netG_B, optimizer_G, opt.lr,
        )
        load_checkpoint(
            opt.checkpoint_critic_A, netD_A, optimizer_D, opt.lr,
        )
        load_checkpoint(
            opt.checkpoint_critic_A, netD_B, optimizer_D, opt.lr,
        )

    train_transform = A.Compose(
        [
            A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )

    fold_list = os.listdir(opt.dataroot)
    fold_list = [fold for fold in fold_list if os.path.isdir(os.path.join(opt.dataroot, fold))]
    fold_list.remove("target")
    
    train_dataset = None

    for fold in fold_list:
        if train_dataset == None:
            # train_dataset = MyDataset(os.path.join(opt.dataroot, fold), os.path.join(opt.dataroot, "target"),
            #                           transform=train_transform)
            train_dataset = FundusDataset(os.path.join(opt.dataroot, fold), os.path.join(opt.dataroot, "target"),
                                     transform=train_transform)
        else:
            train_dataset.add_elements_to_A(os.path.join(opt.dataroot, fold))
                              
    print("High Quality: {}, Low Quality: {}".format(train_dataset.B_len, train_dataset.A_len))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(opt.n_epochs):
        print("{}/{}".format(epoch, opt.n_epochs))
        train_fn(netD_A, netD_B, netG_A, netG_B, train_dataloader, optimizer_D, optimizer_G,
                criterion_contrastive, criterionGAN, criterionCycle, criterionIdt, d_scaler, g_scaler)

        if opt.save_model:
            save_checkpoint(netG_A, optimizer_G, filename=opt.checkpoint_gen_A)
            save_checkpoint(netG_B, optimizer_G, filename=opt.checkpoint_gen_B)
            save_checkpoint(netD_A, optimizer_D, filename=opt.checkpoint_critic_A)
            save_checkpoint(netD_B, optimizer_D, filename=opt.checkpoint_critic_B)


