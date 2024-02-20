import torch.nn as nn
import torch

from Tool import utils_network as Tools

import Tool.cnn_3_64 as net


class gcn_module(nn.Module):
    def __init__(self, in_fea, out_fea):
        super(gcn_module, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_fea, out_fea), requires_grad=True)
        self.bias = nn.Parameter(torch.FloatTensor(out_fea), requires_grad=True)

    def forward(self, x, adj, j=0):
        output = torch.zeros(x.size(0), x.size(1), x.size(2), self.weight.size(1), device='cuda')
        for j in range(x.size(0)):
            t = x[j, ...].contiguous().view(-1, x.size(3))  # x.size(2)=36
            support = torch.mm(t, self.weight)
            support = support.view(x.size(1), x.size(2), -1)  # (x.size(0), x.size(1), 36 or 64)
            # support = torch.tensor(support, dtype=torch.float64)
            out = torch.zeros_like(support)
            for i in range(x.size(1)):
                out[i] = torch.mm(adj[j], support[i])
            output[j] = out

        torch.cuda.empty_cache()
        return output


class IterBlock(nn.Module):
    def __init__(self):
        super(IterBlock, self).__init__()
        self.block1 = gcn_module(36, 64)
        self.block2 = net.cnn()
        self.block3 = gcn_module(64, 36)
        self.relu = nn.ReLU(inplace=True)
        self.arf = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, input_data, adj, graph, train_data_k, mask, csm):
        # i = 0
        csm = csm.transpose(1, 0)
        tmp2 = self.block2(input_data)

        feature1 = graph.image2feature(input_data[:, 0, :, :])[0]
        feature2 = graph.image2feature(input_data[:, 1, :, :])[0]

        feature = torch.stack([feature1, feature2], dim=0)
        tmp3 = self.relu(self.block1(feature, adj))
        tmp3 = self.block3(tmp3, adj)

        tmp3_0 = graph.feature2image(tmp3[0])
        tmp3_1 = graph.feature2image(tmp3[1])
        tmp3 = torch.cat([tmp3_0, tmp3_1], dim=1)
        output = input_data + (1 - self.arf) * tmp2 + self.arf * tmp3
        # dc
        output = Tools.double2complex(image=output)
        output = output * csm
        output_k = Tools.tc_fft2c(output)

        k_space_u = mask * train_data_k
        output_k_con = (1 - mask) * output_k
        final_k_space = output_k_con + 0.999999 * k_space_u

        output = Tools.tc_ifft2c(final_k_space)
        output = torch.sum(output * torch.conj(csm), dim=0)
        output = Tools.complex2double(image=output)

        torch.cuda.empty_cache()

        return output


class GCESS(nn.Module):
    def __init__(self, block_num):
        super(GCESS, self).__init__()
        self.block1 = nn.ModuleList([IterBlock() for i in range(int(block_num / 2))])
        self.block2 = nn.ModuleList([IterBlock() for i in range(int(block_num / 2))])

    def forward(self, input_data, adj, graph, lable_data_k, mask, csm):
        x = input_data
        for index, module in enumerate(self.block1):
            x = module(x, adj, graph, lable_data_k, mask, csm)
        for index, module in enumerate(self.block2):
            x = module(x, adj, graph, lable_data_k, mask, csm)
        return x


if __name__ == '__main__':
    pass
