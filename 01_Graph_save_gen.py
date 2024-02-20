from scipy.io import savemat
import torch
import time
import numpy as np
from pyds_lib.utils import getData_VN, graph_generation

cuda_number = 0
device = 'cuda:%d' % cuda_number


def pre_data_adj_VN(filename, save_path):
    org, Csm, mask = getData_VN(filename)
    org = torch.tensor(org)
    # slices channel w h
    number = int(abs(org).size(-1) / 2 - 2)

    for k in range(org.shape[0]):
        index_f_Real = []
        index_f_Imag = []
        adj_f_Real = []
        adj_f_Imag = []
        graph = graph_generation(org[k].cuda())
        tic = time.time()
        adj_Real = graph.to_final_adj(torch.tensor(org[k].real), std=True)
        toc = time.time()
        print('adj: ', toc - tic, 's')
        adj_Real = np.array(adj_Real.cpu())
        index = np.zeros([number ** 2, 8])
        adj_ = np.zeros([number ** 2, 8]) * (0 + 0j)
        for m in range(adj_Real.shape[0]):
            for n in range(8):
                max_index = np.argmax(abs(adj_Real[m, :]))
                index[m, n] = max_index
                adj_[m, n] = adj_Real[m, max_index]
                adj_Real[m, max_index] = 0

        index_f_Real.append(index)
        adj_f_Real.append(adj_)
        # adj_imag
        adj_imag = np.array(graph.to_final_adj(torch.tensor(org[k].imag), std=True).cpu())
        index = np.zeros([number ** 2, 8])
        adj_ = np.zeros([number ** 2, 8]) * (0 + 0j)

        for m in range((adj_imag).shape[0]):
            for n in range(8):
                max_index = np.argmax(abs(adj_imag[m, :]))
                index[m, n] = max_index
                adj_[m, n] = adj_imag[m, max_index]
                adj_imag[m, max_index] = 0
        index_f_Imag.append(index)
        adj_f_Imag.append(adj_)

        index_f_Real = np.array(index_f_Real)
        adj_f_Real = np.array(adj_f_Real)
        save_Real = np.stack([index_f_Real, adj_f_Real], axis=0)
        savemat(save_path + '/' + '/final_adj_%s_real.mat' % (k), {'vn_datasets_adj': save_Real})
        print('savemat_Real: ', k)
        index_f_Imag = np.array(index_f_Imag)
        adj_f_Imag = np.array(adj_f_Imag)
        save_Imag = np.stack([index_f_Imag, adj_f_Imag], axis=0)
        savemat(save_path + '/' + '/final_adj_%s_imag.mat' % (k), {'vn_datasets_adj': save_Imag})
        print('savemat_Imag: ', k)


pre_data_adj_VN('', save_path='')
