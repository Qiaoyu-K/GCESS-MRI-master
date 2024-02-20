import math
import Tool.utils_network as Tools
import torch


def to_single(x):
    x = Tools.complex2double(x)
    x_real = sos(x[:, 0, ...].squeeze())
    x_imag = sos(x[:, 1, ...].squeeze())
    output = torch.stack((x_real, x_imag), dim=1)
    return output


def sos(x, dim):
    pnorm = 2
    out = torch.sqrt(torch.sum(abs(x ** pnorm), axis=dim))
    return out.float()


def RLNE(im_ori, im_rec):
    L2_error = im_ori - im_rec
    out = torch.norm(L2_error, p=2) / torch.norm(im_ori, p=2)
    return out


def PSNR(im_ori, im_rec):
    mse = torch.mean(abs(im_ori - im_rec) ** 2)
    peakval = torch.max(abs(im_ori))
    out = 10 * math.log10(abs(peakval) ** 2 / mse)
    return out
