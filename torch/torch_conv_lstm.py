import h5py
import yaml
import math
import torch
from torch.utils import data
from torch import nn
from torch.nn import functional as F

def read_config(file_name):
    """ To read the parameters from the yaml file. """
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)
    return config

class TorchDataset(data.Dataset):
    """ A Torch Dataset class with the HDF5 datasets.

    :param data_path: path of the HDF5 dataset.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        DTYPE = torch.float32
        f = h5py.File(data_path, 'r')
        self.lang_mask = f['lang_mask']
        self.language = f['language']
        self.mask = f['mask']
        self.motor = f['motor']
        self.vision = f['vision']

    def __len__(self):
        return self.vision.shape[0]

    def __getitem__(self, index):
        """ Returns the training data corresponding to 'index'.

        :param index: index of the data to return
        :type index: int
        :return: vision, motor, language, mask, lang_mask
        :rtype: tuple(Numpy array)
        """
        return (self.vision[index], self.motor[index], self.language[index],
                self.mask[index], self.lang_mask[index])


class TorchDataLoader(data.DataLoader):
    """ A Torch DataLoader class with the HDF5 datasets.

    :param data_path: path of the HDF5 dataset.
    :param batch_size: batch size for the DataLoader
    :param shuffle: whether to shuffle the data
    """
    def __init__(self, data_path, batch_size, shuffle=True):
        dataset = TorchDataset(data_path)
        super(TorchDataLoader, self).__init__(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=True)


def padding(input_dim, output_dim, stride, kernel_size, dilation):
    """Calculate the padding for a convolution.

    The padding is calculated to attain the desired input and output
    dimensions.

    :param input_dim: dimension of the (square) input
    :type input_dim: int
    :param output_dim: height or width of the square convolution output
    :type output_dim: int
    :param stride: stride
    :type stride: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param dilation: dilation
    :type dilation: int
    :returns: padding for the convolution, residual for the transpose convolution
    :rtype: int, int
    """
    pad = math.ceil(0.5 * (
        stride * (output_dim - 1) - input_dim + dilation * (kernel_size - 1) + 1))
    err_msg = "kernel, stride, dilation and input/output sizes do not match"
    if pad >= 0:
        r = (input_dim + 2 * pad - dilation * (kernel_size - 1) - 1) % stride
        # verify that the padding is correct
        assert ( output_dim ==
            math.floor((input_dim + 2 * pad - dilation * (kernel_size - 1) - 1) / stride) + 1
            ), err_msg
        return pad, r
    else:
        raise ValueError(err_msg)


class ConvLSTMCell(nn.Module):
    """ A convolutional LSTM cell for the visual data. """
    def __init__(self,
                 input_channels, 
                 hidden_channels,
                 inp_kernel_size,
                 hid_kernel_size,
                 inp_stride=1,
                 inp_padding=0,
                 ik_dilation=1,
                 bias=True):
        """ Initialize the ConvLSTM cell.

        :param input_channels: number of channels in the input tensor
        :type input_channels: int
        :param hidden_channels: number of channels in the hidden state
        :type hidden_channels: int
        :param inp_kernel_size: kernel size for input convolutions
        :type inp_kernel_size: int
        :param hid_kernel_size: kernel size for hidden state convolutions
        :type hid_kernel_size: int
        :param inp_stride: stride for the input convolutions
        :type inp_stride: int
        :param inp_padding: padding for the input convolutions
        :type inp_padding: int
        :param ik_dilation: kernel dilation for the input convolutions
        :type ik_dilation: int
        :param bias: whether to use bias in the convolutional layers
        :type bias: bool
        """
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.inp_kernel_size = inp_kernel_size
        self.hid_kernel_size = hid_kernel_size
        self.inp_stride = inp_stride
        self.inp_padding = inp_padding
        self.ik_dilation = ik_dilation
        self.bias = bias

        self.inp_conv = nn.Conv2d(input_channels,
                                  4 * hidden_channels,
                                  kernel_size=self.inp_kernel_size,
                                  stride=self.inp_stride,
                                  padding=self.inp_padding,
                                  dilation=self.ik_dilation,
                                  bias=self.bias,
        )
        self.hid_conv = nn.Conv2d(hidden_channels,
                                  4 * hidden_channels,
                                  kernel_size=self.hid_kernel_size,
                                  stride=1,
                                  padding="same",
                                  dilation=1,
                                  bias=self.bias,
        )
        self.i_bias = nn.Parameter(torch.zeros((self.hidden_channels, 1, 1)))
        self.f_bias = nn.Parameter(torch.zeros((self.hidden_channels, 1, 1)))
        self.o_bias = nn.Parameter(torch.zeros((self.hidden_channels, 1, 1)))
        self.g_bias = nn.Parameter(torch.zeros((self.hidden_channels, 1, 1)))

    def forward(self, input_tensor, cur_state):
        """ Advance one time step.

        :param input_tensor: input tensor for the current time step
        :type input_tensor: torch.Tensor
        :param cur_state: current hidden and cell states
        :type cur_state: tuple(torch.Tensor, torch.Tensor)
        :return: next hidden and cell states
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        h_cur, c_cur = cur_state
    
        inp_convs = self.inp_conv(input_tensor)
        print(f"inp_convs: {inp_convs.shape}")
        hid_convs = self.hid_conv(h_cur)
        print(f"hid_convs: {hid_convs.shape}")
        ic, fc, oc, gc = torch.split(inp_convs + hid_convs, self.hidden_channels, dim=1)
        print(f"ic: {ic.shape}")
        i = F.sigmoid(ic + self.i_bias)
        f = F.sigmoid(fc + self.f_bias)
        o = F.sigmoid(oc + self.o_bias)
        g = F.tanh(gc + self.g_bias)
        c_next = f * c_cur + i * g
        h_next = o * F.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width),
                torch.zeros(batch_size, self.hidden_channels, height, width))

    def init_hidden_from_normal(self, batch_size, image_size, std=0.1):
        """ Initialize the hidden state from a normal distribution. """
        height, width = image_size
        return (std * torch.randn(batch_size, self.hidden_channels, height, width),
                std * torch.randn(batch_size, self.hidden_channels, height, width))


class PredictorCell(nn.Module):
    """ Predicts the next image and updates the hidden state.

    This class uses a ConvLSTM cell to update the hidden state, and a transpose
    convolutional layer to predict the next image.
    """
    def __init__(self, conv_params, conv_params_t):
        """ Initialize the PredictorCell.

        :param conv_params: parameters for the ConvLSTM cell. Dictionary with:
            - input_size: number of channels in the input tensor
            - hidden_size: number of channels in the hidden state
            - kernel_size: size of the convolutional kernel
            - stride: stride of the convolutional layer
            - dilation: kernel dilation
            - padding: padding of the convolutional layer
            - bias: whether to use bias in the convolutional layers
        :type conv_params: dict
        :param conv_params_t: parameter dictionary for the transpose convolution.
            - input_size: number of channels in the input tensor
            - output_size: number of channels in the output tensor
            - kernel_size: size of the convolutional kernel
            - stride: stride of the "direct" convolution
            - padding: padding of the "direct" convolution
        """
        super(PredictorCell, self).__init__()
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv_lstm = ConvLSTMCell(conv_params['input_size'],
                                      conv_params['hidden_size'],
                                      conv_params['kernel_size'],
                                      stride=conv_params['stride'],
                                      padding=conv_params['padding'],
                                      dilation=conv_params['dilation'],
                                      bias=conv_params['bias'],
        )
        self.transp_conv = nn.ConvTranspose2d(in_channels=hidden_size,
                                              out_channels=input_size,
                                              kernel_size=kernel_size,
                                              padding=self.padding,
                                              bias=bias)




