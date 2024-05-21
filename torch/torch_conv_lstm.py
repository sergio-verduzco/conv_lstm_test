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


def padding_fun(input_dim, output_dim, stride, kernel_size, dilation):
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
        hid_convs = self.hid_conv(h_cur)
        ic, fc, oc, gc = torch.split(inp_convs + hid_convs, self.hidden_channels, dim=1)
        i = F.sigmoid(ic + self.i_bias)
        f = F.sigmoid(fc + self.f_bias)
        o = F.sigmoid(oc + self.o_bias)
        g = F.tanh(gc + self.g_bias)
        c_next = f * c_cur + i * g
        h_next = o * F.tanh(c_next)

        return h_next, c_next

    def hid_shape(self, inp_size):
        """ Calculate the shape of the hidden state for a given input size.

        :param inp_size: height and width of the input images
        :type inp_size: tuple(int)
        :return: height and width of the hidden state
        :rtype: tuple(int)
        """
        height, width = inp_size
        height = (height + 2 * self.inp_padding - self.ik_dilation * (self.inp_kernel_size - 1) - 1) // self.inp_stride + 1
        width = (width + 2 * self.inp_padding - self.ik_dilation * (self.inp_kernel_size - 1) - 1) // self.inp_stride + 1
        return height, width

    def init_hidden(self, batch_size, image_size):
        """ Initialize the hidden state to zeros.

        :param batch_size: number of inputs in the batch
        :type batch_size: int
        :param image_size: height and width of the input images
        :type image_size: tuple(int)
        """
        height, width = self.hid_shape(image_size)
        return (torch.zeros(batch_size, self.hidden_channels, height, width),
                torch.zeros(batch_size, self.hidden_channels, height, width))

    def init_hidden_from_normal(self, batch_size, image_size, std=0.05):
        """ Initialize the hidden state from a normal distribution. """
        height, width = self.hid_shape(image_size)
        return (std * torch.randn(batch_size, self.hidden_channels, height, width),
                std * torch.randn(batch_size, self.hidden_channels, height, width))


class PredictorCell(nn.Module):
    """ Predicts the next image and updates the hidden state.

    This class uses a ConvLSTM cell to update the hidden state, and a transpose
    convolutional layer to predict the next image.
    """
    def __init__(self, conv_params, conv_params_t):
        """ Initialize the PredictorCell.

        :param conv_params: parameters for ConvLSTMCell. Dictionary with:
            - input_channels: number of channels in the input tensor
            - hidden_channels: number of channels in the hidden state
            - inp_kernel_size: kernel size for input convolutions
            - hid_kernel_size: kernel size for hidden state convolutions
            - inp_stride: stride for the input convolutions
            - inp_padding: padding for the input convolutions
            - ik_dilation: kernel dilation for the input convolutions
            - bias: whether to use bias in the convolutional layers
        :type conv_params: dict
        :param conv_params_t: parameter dictionary for the transpose convolution.
            - kernel_size: size of the convolutional kernel
            - ik_dilation: kernel dilation for the transpose convolutions
            - inp_padding: input padding for the transpose convolutions
            - output_padding: additional size for the output shape
            - bias: whether to use bias in the convolutional layer
        :type conv_params_t: dict
        """
        super(PredictorCell, self).__init__()
        self.conv_lstm = ConvLSTMCell(conv_params['input_channels'],
                                      conv_params['hidden_channels'],
                                      conv_params['inp_kernel_size'],
                                      conv_params['hid_kernel_size'],
                                      inp_stride=conv_params['inp_stride'],
                                      inp_padding=conv_params['inp_padding'],
                                      ik_dilation=conv_params['ik_dilation'],
                                      bias=conv_params['bias'],
        )
        self.transp_conv = nn.ConvTranspose2d(conv_params['hidden_channels'],
                                              conv_params['input_channels'],
                                              conv_params_t['kernel_size'],
                                              stride=conv_params['inp_stride'],
                                              padding=conv_params_t['inp_padding'],
                                              output_padding=conv_params_t['output_padding'],
                                              bias=conv_params_t['bias'],
        )
        
    def forward(self, input_tensor, cur_state):
        """ Advance one time step.

        :param input_tensor: input tensor for the current time step
        :type input_tensor: torch.Tensor
        :param cur_state: current hidden and cell states
        :type cur_state: tuple(torch.Tensor, torch.Tensor)
        :return: next input, next hidden and cell states
        :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor)
        """
        h_cur, c_cur = cur_state
        h_next, c_next = self.conv_lstm(input_tensor, (h_cur, c_cur))
        next_input = self.transp_conv(h_next)
        return next_input, h_next, c_next

    def init_hidden(self, batch_size, image_size):
        return self.conv_lstm.init_hidden(batch_size, image_size)

    def init_hidden_from_normal(self, batch_size, image_size, std=0.1):
        return self.conv_lstm.init_hidden_from_normal(batch_size, image_size, std=std)


class Predictor(nn.Module):
    """ Autoregressive prediction for sequences of images with ConvLSTM. """
    def __init__(self, conv_params, conv_params_t):
        """ Initialize the Predictor.

        :param conv_params: Same as for the PredictorCell class.
        :type conv_params: dict
        :param conv_params_t: Same as for the PredictorCell class.
        :type conv_params_t: dict
        """
        super(Predictor, self).__init__()
        self.predictor_cell = PredictorCell(conv_params, conv_params_t)

    @torch.compile(backend="cudagraphs")
    def forward(self, first_imgs, T, normal_init=True):
        """ Predict a sequence of T images.

        :param first_imgs: Tensor with shape (batches, 3, height, width)
        :type first_imgs: torch.Tensor
        :param T: number of images to predict
        :type T: int
        :normal_init: Initialize the ConvLSTM hidden state from a normal?
        :type normal_init: bool
        :return: Tensor with shape (batches, T, 3, height, width)
        :rtype: torch.Tensor
        """
        shape = first_imgs.shape
        im_size = (shape[2], shape[3])
        if normal_init:
            h, c = self.predictor_cell.init_hidden_from_normal(shape[0], im_size)
        else:
            h, c = self.predictor_cell.init_hidden(shape[0], im_size)
        h = h.to(first_imgs.device)
        c = c.to(first_imgs.device)
        
        with torch.no_grad():
            pred_sequence = [first_imgs.clone()]

        for img_n in range(1, T):
            img, h, c = self.predictor_cell(pred_sequence[-1], (h, c))
            pred_sequence.append(img)

        return torch.stack(pred_sequence, dim=1)




