from torch.linalg import multi_dot

import torch
import torch.nn.functional as F

from torch.nn.modules.utils import _single, _pair, _triple


class MyConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input_):

        hout = ((input_.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) //
                self.stride[0]) + 1
        wout = ((input_.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) //
                self.stride[1]) + 1

        inputUnfolded = F.unfold(input_, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation,
                                 stride=self.stride)

        i_t = inputUnfolded.transpose(1, 2)

        convolvedOutput = multi_dot([i_t.squeeze(), self.weight.view(self.weight.size(0), -1).t()]).unsqueeze(0).transpose(
            1, 2)

        if self.bias is not None:
            convolvedOutput += self.bias.view(-1, 1)

        convolutionReconstruction = convolvedOutput.view(input_.shape[0], self.out_channels, hout, wout)


        return convolutionReconstruction


class MyConv2dSVD(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2dSVD, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input_):
        hout = ((input_.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) //
                self.stride[0]) + 1
        wout = ((input_.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) //
                self.stride[1]) + 1

        inputUnfolded = F.unfold(input_, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation,
                                 stride=self.stride)

        u, s_r, v = torch.split(self.weight,
                                [self.in_channels * self.kernel_size[0] * self.kernel_size[1], 1, self.out_channels])

        s = s_r.reshape(-1)
        s_diag = torch.diag(s)
        v_t = v.t()

        convolvedOutput = multi_dot([inputUnfolded.transpose(1, 2).squeeze(), u, s_diag, v_t]).unsqueeze(0).transpose(1, 2)

        if self.bias is not None:
            convolvedOutput += self.bias.view(-1, 1)

        convolutionReconstruction = convolvedOutput.view(input_.shape[0], self.out_channels, hout, wout)
        return convolutionReconstruction
