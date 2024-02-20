import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import argparse
import numpy as np
from pyds_lib.utils import graph_generation_network, adj_restore
from Tool import utils_network as Tools, evaluation as eval
from model import model_GCESS as model_gcess
from pytorch_msssim import SSIM

parser = argparse.ArgumentParser()
parser.add_argument("--n_block_gcess", type=int, default=10)
parser.add_argument("--dataset_path", default='')
parser.add_argument("--checkpt_file_gcess", default='')
parser.add_argument("--adj_path", default='')

parser.add_argument("--mask_path", default='')

parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False


class net():
    def __init__(self):
        self.loss1 = nn.MSELoss()
        self.dataset_path = opt.dataset_path
        self.mask_path = opt.mask_path
        self.start_number = 320
        self.adj_path = opt.adj_path
        self.model_gcess = model_gcess.GCESS(opt.n_block_gcess)
        label_data, Csm, Mask = Tools.getTestdata_VN(self.dataset_path, self.mask_path, self.start_number, )  # VN
        # slices channel w h
        label_data = label_data.unsqueeze(1) * Csm
        self.number = label_data.shape[0]
        self.checkpt_file_gcess = opt.checkpt_file_gcess
        k_space = Tools.tc_fft2c(label_data)
        k_space_u = k_space * Mask
        image_u = Tools.tc_ifft2c(k_space_u)
        self.Mask = torch.tensor(Mask, device='cuda')
        self.undersampling_rate = Tools.undersampling_rate(Mask)
        print('self.undersampling_rate: ', self.undersampling_rate)
        self.image_u = torch.tensor(image_u, device='cuda')
        label_data = torch.sum(label_data * torch.conj(Csm), dim=1)
        self.label_data = torch.tensor(label_data, device='cuda')
        self.Csm = torch.tensor(Csm, device='cuda')
        torch.cuda.empty_cache()
        self.index = torch.arange(self.start_number, self.start_number + int(self.label_data.shape[0]), 1)
        torch.cuda.empty_cache()
        test_data = TensorDataset(self.image_u,
                                  self.label_data,
                                  self.Csm,
                                  self.index)
        self.test_data = DataLoader(test_data, batch_size=1, shuffle=False)
        print(self.label_data.shape[0])
        torch.cuda.empty_cache()

        if cuda:
            self.model = self.model.cuda()
            self.model_gcess = self.model_gcess.cuda()

    def test(self):
        checkpt_file_gcess = self.checkpt_file_gcess
        self.model_gcess.load_state_dict(torch.load(checkpt_file_gcess))
        print('-->start test')
        rec_data_gcess = []
        rec_label_data = []
        all_PSNR_gcess = []
        all_SSIM_gcess = []
        for index, data in enumerate(self.test_data):
            test_data, label_data, Csm, index_1 = data
            print(index_1)
            adj_real, adj_imag = Tools.pre_data_adj(index_1, self.adj_path)

            adj = adj_restore(adj_real, adj_imag)
            regu = torch.max(abs(test_data))
            ssim = SSIM(data_range=regu, channel=1)

            test_data, label_data = test_data / regu, (
                (label_data).view(test_data.shape[0], 1, test_data.shape[-2], test_data.shape[-1])) / regu
            val_data_k = Tools.tc_fft2c(test_data).transpose(1, 0)
            test_data = torch.sum(test_data * torch.conj(Csm), dim=1)

            test_data = Tools.complex2double(test_data)

            with torch.no_grad():
                graph = graph_generation_network(label_data)
                output = self.model(test_data, val_data_k, self.Mask, Csm)
                output_gcess = self.model_gcess(test_data, adj.squeeze_(), graph, val_data_k, self.Mask, Csm)
                output = Tools.double2complex(image=output)
                output_gcess = Tools.double2complex(image=output_gcess)
                test_data = Tools.double2complex(image=test_data)

            label_data = torch.tensor(label_data, dtype=torch.complex64)
            torch.cuda.empty_cache()
            PSNR1 = eval.PSNR(abs(label_data), abs(test_data))
            RLNE_gcess = eval.RLNE(abs(label_data), abs(output_gcess))
            PSNR_gcess = eval.PSNR(abs(label_data), abs(output_gcess))
            SSIM_gcess = ssim(abs(label_data), abs(output_gcess.unsqueeze(1)))

            all_PSNR_gcess.append(PSNR_gcess)
            all_SSIM_gcess.append(SSIM_gcess.cpu().detach().numpy())

            print('test_gcess---------', " [RLNE: %f] : [PSNR: %f/%f] " % (RLNE_gcess, PSNR_gcess, PSNR1))
            rec_label_data.append(label_data.cpu().detach().numpy())
            torch.cuda.empty_cache()
        all_PSNR_gcess = np.array(all_PSNR_gcess)
        all_SSIM_gcess = np.array(all_SSIM_gcess)
        print('test_gcess---------',
              " [mean_SSIM: %f] : [PSNR: %f] " % (np.mean(all_SSIM_gcess), np.mean(all_PSNR_gcess)))


if __name__ == "__main__":
    network = net()  # 实例化
    network.test()
