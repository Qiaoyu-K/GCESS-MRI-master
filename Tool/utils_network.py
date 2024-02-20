from scipy.io import loadmat
import torch
import time
from numpy.matlib import repmat
import h5py
import numpy as np
import h5py as h5


def getData_VN(filename, mask_filename, number):
    with h5.File(filename) as f:
        trnCsm, trnOrg = f['csm_final'][..., :number], f['image_final'][..., :number]

    trnCsm = trnCsm['real'] + trnCsm['imag'] * 1j
    trnOrg = trnOrg['real'] + trnOrg['imag'] * 1j
    trnMask = loadmat(mask_filename)
    mask_key = '_'.join(mask_filename.split('\\')[-1].split('_')[:-2])
    trnMask = trnMask[mask_key]

    trnCsm = torch.tensor(trnCsm.transpose([3, 0, 2, 1]), dtype=torch.complex64)  # patch,channel,w,h
    trnOrg = torch.tensor(trnOrg.transpose([2, 1, 0]), dtype=torch.complex64)
    trnMask = torch.tensor(trnMask)

    return trnOrg, trnCsm[:number], trnMask


def getTestdata_VN(filename, mask_filename, start):
    with h5.File(filename) as f:
        testCsm, testOrg = f['csm_final'][:], f['image_final'][:]
    testCsm = testCsm['real'] + testCsm['imag'] * 1j
    testOrg = testOrg['real'] + testOrg['imag'] * 1j
    testMask = loadmat(mask_filename)
    mask_key = '_'.join(mask_filename.split('\\')[-1].split('_')[:-2])
    testMask = testMask[mask_key]
    testCsm = torch.tensor(testCsm.transpose([3, 0, 2, 1]), dtype=torch.complex64)
    testOrg = torch.tensor(testOrg.transpose([2, 1, 0]), dtype=torch.complex64)
    testMask = torch.tensor(testMask)

    return testOrg[start:], testCsm[start:], testMask


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)


def undersampling_rate(mask):
    rate_ = mask.sum() / (mask.shape[-1] * mask.shape[-2])
    return rate_


def complex2double(image, dim=1):
    output = torch.stack((image.real.float(), image.imag.float()), dim=dim)
    return output


def double2complex(image_real=0, image_imag=0, image='none'):
    if image == 'none':
        output = torch.complex(image_real, image_imag)
    else:
        output = torch.complex(image[:, 0, ...], image[:, 1, ...])
    return output


def tc_fft2c(x):
    kx = int(x.shape[-2])
    ky = int(x.shape[-1])
    axes = (-2, -1)
    k_space = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), dim=axes)) / np.sqrt(kx * ky)
    return k_space


def tc_ifft2c(x):
    kx = int(x.shape[-2])
    ky = int(x.shape[-1])
    axes = (-2, -1)
    image = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x), dim=axes)) * np.sqrt(kx * ky)

    return image


def pre_data_adj(index, adj_path):
    file_real = loadmat(adj_path + '/final_adj_%d_real.mat' % index)
    file_imag = loadmat(adj_path + '/final_adj_%d_imag.mat' % index)
    adj_key = adj_path.split('\\')[-1]
    adj_real = file_real[adj_key + '_']
    adj_imag = file_imag[adj_key + '_']

    return torch.tensor(adj_real), torch.tensor(adj_imag)
