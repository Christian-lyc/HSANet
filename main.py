import pydicom

import argparse
import os
import torch
from dataset import DenoiseDataset
import torchvision.transforms as transforms
from model import SwinUNet
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from unet import UNet
from measure import compute_measure
from loader import get_loader

parser = argparse.ArgumentParser(description='PyTorch LDCT denoising')
parser.add_argument('--epochs',default=1,type=int)
parser.add_argument('--batch_size',default=2,type=int)
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--weight_decay',default=1e-4,type=float)
parser.add_argument('--data',default='/home/yichao/Desktop/LDCT/CT',type=str)
parser.add_argument('--patch_n', type=int, default=10)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-160.0)
parser.add_argument('--trunc_max', type=float, default=240.0)
args = parser.parse_args()

trunc_min=args.trunc_min
trunc_max=args.trunc_max
norm_range_max=args.norm_range_max
norm_range_min=args.norm_range_min
def denormalize_( image):
        image = image * (norm_range_max - norm_range_min) + norm_range_min
        return image

def trunc( mat):
        mat[mat <= trunc_min] = trunc_min
        mat[mat >= trunc_max] = trunc_max
        return mat

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traindir = os.path.join(args.data, 'tr')
    valdir = os.path.join(args.data, 'te')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = get_loader(mode='train',
                             load_mode=0,
                             saved_path=traindir,
                             patch_n=args.patch_n,
                             patch_size=args.patch_size,
                             transform=args.transform,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    test_loader = get_loader(mode='test',
                             load_mode=0,
                             saved_path=valdir,
                             patch_n= None,
                             patch_size=None,
                             transform=args.transform,
                             batch_size=1,
                             num_workers=args.num_workers)

    #traindataset= DenoiseDataset(traindir=traindir,transform=transform)
    #valdataset= DenoiseDataset(valdir=valdir,transform=transform)

    #trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True,
    #                                          pin_memory=True)  #
    #valloader = torch.utils.data.DataLoader(valdataset, batch_size=1, shuffle=False,
    #                                         pin_memory=True)

    model = SwinUNet(64,64,1,32,num_blocks=2).cuda()
    NumOfParam = count_parameters(model)
    print('trainable parameter:', NumOfParam)

#    unet=UNet(1,1).cuda()
    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer_u = torch.optim.Adam(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ###
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=args.weight_decay)
    iter_num=0
    for epoch in range(args.epochs):
        for low,full in train_loader:
            model.train()
            low = low.unsqueeze(1).to(torch.float32).to(device)
            full=full.unsqueeze(1).to(torch.float32)
            if args.patch_size:
                low = low.view(-1, 1, args.patch_size, args.patch_size)
                full = full.view(-1, 1, args.patch_size, args.patch_size)
            y=model(low)
            #unet.eval()
            #with torch.no_grad():
            #    y_noise=unet(low.to(device))
            loss = criterion(y,full.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''
            model.eval()
            with torch.no_grad():
                y=model(low.to(device))
            unet.train()
            y_noise=unet(low.to(device))
            loss_noise = criterion(y-y_noise,full.to(device))
            optimizer_u.zero_grad()
            loss_noise.backward()
            optimizer_u.step()
            '''
            lr_=args.lr*(1-iter_num/40000)**0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1


        if epoch % 5 == 0:
            ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
            pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
            for low,full in test_loader:
                shape_ = low.shape[-1]
                model.eval()
                full=full.unsqueeze(0).to(torch.float64).to(device)
                low=low.unsqueeze(0).to(torch.float32)
                y=model(low.to(device))
                low=low.to(torch.float64)
                low = trunc(denormalize_(low.view(shape_, shape_).cpu().detach()))
                full = trunc(denormalize_(full.view(shape_, shape_).cpu().detach()))
                y = trunc(denormalize_(y.to(torch.float64).view(shape_, shape_).cpu().detach()))

                data_range = trunc_max - trunc_min

                original_result, pred_result = compute_measure(low, full, y, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                #psnr_value=psnr(y.cpu().numpy(),full.cpu().numpy())
                #print('epochs=',epoch,'psnr',psnr_value)

            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(test_loader),
                                                                                            ori_ssim_avg/len(test_loader),
                                                                                            ori_rmse_avg/len(test_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(test_loader),
                                                                                                  pred_ssim_avg/len(test_loader),
                                                                                                  pred_rmse_avg/len(test_loader)))





if __name__ == '__main__':
    main()

#dcm_path = '/home/yichao/Desktop/LDCT/L004-20250527T075323Z-1-001/L004/08-21-2018-84608/1.000000-Low Dose Images-35583/1-01.dcm'
#ds = pydicom.dcmread(dcm_path)
#image = ds.pixel_array
