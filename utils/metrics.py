import torch
import math
import numpy as np
from skimage.metrics import structural_similarity
import lpips

def MAE(vec1, vec2, mask=None, normalize=True):
    '''
    Input : N x 3  or  H x W x 3 .   [-1,1]
    Output : MAE, AE
    '''
    vec1, vec2 = vec1.copy(), vec2.copy()
    mask = mask.copy() if mask is not None else mask
    if normalize:
        norm1 = np.linalg.norm(vec1.astype(np.float64), axis=-1)
        norm2 = np.linalg.norm(vec2.astype(np.float64), axis=-1)
        vec1 /= norm1[...,None] + 1e-5
        vec2 /= norm2[...,None] + 1e-5
        vec1[norm1==0] = 0
        vec2[norm2==0] = 0
    dot_product = (vec1.astype(np.float64) * vec2.astype(np.float64)).sum(-1).clip(-1, 1)
    if mask is not None:
        dot_product = dot_product[mask.squeeze(1).astype(bool)]
    angular_err = np.arccos(dot_product) * 180.0 / math.pi
    l_err_mean  = angular_err.mean()
    return l_err_mean, angular_err

def PSNR(img1, img2, mask=None):
    '''
    Input : H x W x 3   [0,1]
    Output : PSNR
    '''
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    if mask is not None:
        img1, img2 = img1[mask.astype(bool)], img2[mask.astype(bool)]
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        psnr = 100
    else:
        psnr = - 10.0 * math.log10(mse)
    return psnr
   
def SSIM(img1, img2, mask=None, data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5,  use_sample_covariance=False):
    '''
    Input : H x W x 3   [0,1]
    Output : SSIM
    '''
    H, W = np.sqrt(img1.shape[0]).astype(int), np.sqrt(img1.shape[0]).astype(int)
    img1, img2 = img1.reshape(H, W, 3), img2.reshape(H, W, 3)
    
    ssim = structural_similarity(img1, img2,
                        data_range=data_range, channel_axis=channel_axis, 
                        gaussian_weights=gaussian_weights, sigma=sigma, 
                        use_sample_covariance=use_sample_covariance)
    return ssim
   
class LPIPS():
    def __init__(self, net='alex'):
        '''
        Input : H x W x 3   [0,1]
        Output : LPIPS
        '''
        self.loss_fn = lpips.LPIPS(net=net).cuda()

    def __call__(self, img1, img2, mask=None):
        # img1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()*2.-1
        # img2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()*2.-1
        
        # reshape H x W x 3
        H, W = np.sqrt(img1.shape[0]).astype(int), np.sqrt(img1.shape[0]).astype(int)
        img1, img2 = img1.reshape(H, W, 3), img2.reshape(H, W, 3)

        img1 = img1.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        img2 = img2.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

        err = self.loss_fn.forward(img1,img2)
        return err.item()