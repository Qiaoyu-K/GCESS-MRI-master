import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pyds_lib.utils import graph_generation_network, adj_restore
from Tool import utils_network as Tools, evaluation as eval

from model import model_GCESS as model
from pytorch_msssim import SSIM

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.00015, help="adam: learning rate")
parser.add_argument("--n_block", type=int, default=10)
parser.add_argument("--dataset_path", default='')
parser.add_argument("--mask_path", default='')
parser.add_argument("--adj_path", default='')
parser.add_argument("--model_save_path", type=str, default="")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False


class net():
    def __init__(self):
        self.loss = nn.MSELoss()
        self.start = 0
        self.epoch = opt.epochs
        self.dataset_path = opt.dataset_path
        self.mask_path = opt.mask_path
        self.adj_path = opt.adj_path
        self.train_number = 10
        self.label_data, self.Csm, self.Mask = Tools.getData_VN(self.dataset_path, self.mask_path,
                                                                self.train_number)  # VN
        self.label_data = self.label_data.unsqueeze(1) * self.Csm
        self.number = self.label_data.shape[0]
        self.model = model.GCESS(opt.n_block)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.path = opt.model_save_path

        k_space = Tools.tc_fft2c(self.label_data)
        k_space_u = k_space * self.Mask
        self.Mask = torch.tensor(self.Mask, device='cuda')

        image_u = Tools.tc_ifft2c(k_space_u)
        self.Mask = torch.tensor(self.Mask, device='cuda')

        self.undersampling_rate = Tools.undersampling_rate(self.Mask)
        print('self.undersampling_rate: ', self.undersampling_rate)
        self.image_u = torch.tensor(image_u, device='cuda')
        self.label_data = torch.sum(self.label_data * torch.conj(self.Csm), dim=1)

        self.label_data = torch.tensor(self.label_data, device='cuda')
        self.Csm = torch.tensor(self.Csm, device='cuda')

        self.index = torch.arange(0, self.number, 1)
        torch.cuda.empty_cache()
        i = -(self.number // 8)
        train_data = TensorDataset(self.image_u[: i, ...],
                                   self.label_data[: i, ...],
                                   self.Csm[: i, ...],
                                   self.index[: i, ...])
        val_data = TensorDataset(self.image_u[i:, ...],
                                 self.label_data[i:, ...],
                                 self.Csm[i:, ...],
                                 self.index[i:, ...])
        print('train number: ', self.label_data.shape[0] + i, 'value number: ', -i)

        self.train_data = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)  # 0.35
        self.initialize_weights()

        del train_data, val_data, self.label_data, self.image_u
        torch.cuda.empty_cache()

        if cuda:
            self.model = self.model.cuda()

    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, model.gcn_module):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                module.bias.data.zero_()
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                module.bias.data.zero_()
            if isinstance(module, model.IterBlock):
                nn.init.normal_(module.arf, mean=0, std=0.1)

    def train(self):
        loss_min = 1
        for epoch in range(self.start, self.epoch):
            print('-->start training epoch: ', epoch)
            all_loss = []
            all_RLNE = []
            all_PSNR = []
            all_SSIM = []
            all_PSNR1 = []
            all_SSIM1 = []
            for index, data in enumerate(self.train_data):
                train_data, label_data, Csm, index_1 = data

                adj_real, adj_imag = Tools.pre_data_adj(index_1, self.adj_path)  # GCN_ADJ_CNN
                adj = adj_restore(adj_real, adj_imag)

                regu = torch.max(abs(train_data))
                ssim = SSIM(data_range=regu, channel=1)
                train_data, label_data = train_data / regu, (
                    (label_data).view(train_data.shape[0], 1, train_data.shape[-2],
                                      train_data.shape[-1])) / regu

                train_data_k = Tools.tc_fft2c(train_data).transpose(1, 0)
                train_data = torch.sum(train_data * torch.conj(Csm), dim=1)
                train_data = Tools.complex2double(train_data)

                graph = graph_generation_network(label_data)
                output = self.model(train_data, adj.squeeze_(), graph, train_data_k, self.Mask, Csm)

                output = Tools.double2complex(image=output)
                train_data = Tools.double2complex(image=train_data)

                self.optimizer.zero_grad()
                loss = self.loss(abs(output), abs(label_data.squeeze(1)))
                loss.backward()
                self.optimizer.step()

                train_data = train_data * regu
                label_data = label_data * regu
                output = output * regu

                RLNE = eval.RLNE(label_data, output)
                PSNR = eval.PSNR(label_data, output)
                PSNR1 = eval.PSNR(label_data, train_data)

                SSIM_ = ssim(abs(label_data), abs(output.unsqueeze(1)))
                SSIM1 = ssim(abs(label_data), abs(train_data.unsqueeze(1)))

                all_loss.append(loss.item())
                all_RLNE.append(RLNE.cpu().detach().numpy())
                all_PSNR.append(PSNR)
                all_SSIM.append(SSIM_.cpu().detach().numpy())
                all_PSNR1.append(PSNR1)
                all_SSIM1.append(SSIM1.cpu().detach().numpy())
                torch.cuda.empty_cache()
                del output, Csm, adj, adj_imag, adj_real, train_data, label_data

            all_loss = np.array(all_loss)
            all_RLNE = np.array(all_RLNE)
            all_PSNR = np.array(all_PSNR)
            all_SSIM = np.array(all_SSIM)
            all_PSNR1 = np.array(all_PSNR1)
            all_SSIM1 = np.array(all_SSIM1)

            print('train---------', "[Epoch %d/%d] : [loss: %f] : [RLNE: %f] : [PSNR: %f/%f] : [SSIM: %f/%f]  " % (
                epoch + 1, self.epoch, np.mean(all_loss), np.mean(all_RLNE), np.mean(all_PSNR),
                np.mean(all_PSNR1), np.mean(all_SSIM), np.mean(all_SSIM1)))

            self.scheduler.step()

            if epoch % 1 == 0:
                loss_val = self.val()
                if loss_val < loss_min:
                    loss_min = loss_val
                    torch.save(self.model.state_dict(),
                               '%s/model_epoch_%04d.pth' % (
                                   'save_models/GCESS', epoch + 1))

    def val(self):
        all_PSNR1 = []
        all_SSIM1 = []
        all_loss = []
        all_RLNE = []
        all_PSNR = []
        all_SSIM = []
        for index, data in enumerate(self.val_data):
            val_data, label_data, Csm, index_1 = data
            adj_real, adj_imag = Tools.pre_data_adj(index_1, self.adj_path)  # GCN_ADJ_CNN

            adj = adj_restore(adj_real, adj_imag)
            regu = torch.max(abs(val_data))
            ssim = SSIM(data_range=regu, channel=1)
            val_data, label_data = val_data / regu, (
                (label_data).view(val_data.shape[0], 1, val_data.shape[-2], val_data.shape[-1])) / regu
            val_data_k = Tools.tc_fft2c(val_data).transpose(1, 0)
            val_data = torch.sum(val_data * torch.conj(Csm), dim=1)
            val_data = Tools.complex2double(val_data)

            graph = graph_generation_network(label_data)
            with torch.no_grad():
                output = self.model(val_data, adj.squeeze_(), graph, val_data_k, self.Mask, Csm)
                output = Tools.double2complex(image=output)
                val_data = Tools.double2complex(image=val_data)

            label_data = torch.tensor(label_data, dtype=torch.complex64)
            loss = self.loss(abs(output), abs(label_data.squeeze_(1)))
            torch.cuda.empty_cache()
            RLNE = eval.RLNE(abs(label_data), abs(output))
            PSNR = eval.PSNR(abs(label_data), abs(output))
            PSNR1 = eval.PSNR(abs(label_data), abs(val_data))

            SSIM_ = ssim(abs(label_data.unsqueeze_(1)), abs(output.unsqueeze_(1)))
            SSIM1 = ssim(abs(label_data), abs(val_data.unsqueeze_(1)))

            all_loss.append(loss.item())
            all_RLNE.append(RLNE.cpu().detach().numpy())
            all_PSNR.append(PSNR)
            all_SSIM.append(SSIM_.cpu().detach().numpy())
            all_PSNR1.append(PSNR1)
            all_SSIM1.append(SSIM1.cpu().detach().numpy())

            del output, Csm, adj, adj_imag, adj_real, val_data, label_data

        all_loss = np.array(all_loss)
        all_RLNE = np.array(all_RLNE)
        all_PSNR = np.array(all_PSNR)
        all_SSIM = np.array(all_SSIM)

        all_PSNR1 = np.array(all_PSNR1)
        all_SSIM1 = np.array(all_SSIM1)

        print('val---------------------------', "[loss: %f] : [RLNE: %f] : [PSNR: %f/%f] : [SSIM: %f/%f]  " % (
            np.mean(all_loss), np.mean(all_RLNE), np.mean(all_PSNR),
            np.mean(all_PSNR1), np.mean(all_SSIM), np.mean(all_SSIM1)))

        torch.cuda.empty_cache()
        return np.mean(all_loss)


if __name__ == "__main__":
    network = net()  # 实例化
    network.train()
