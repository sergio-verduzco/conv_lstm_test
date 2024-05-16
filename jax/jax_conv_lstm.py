import sys
import yaml

import h5py
import jax
import jax.numpy as jnp
from jax import lax
from jax import random
from jax.tree_util import tree_map
import numpy as np
from torch.utils import data
import math
from functools import partial
from jax.tree_util import Partial
from typing import NamedTuple
import pickle

def read_config(file_name):
    """ To read the parameters from the yaml file. """
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)
    return config

class JaxDataset(data.Dataset):
    """ A Torch Dataset class with the HDF5 datasets.

    :param data_path: path of the HDF5 dataset.
    """
    def __init__(self, data_path):
        self.data_path = data_path
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

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=True,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

def kernel_initializer(key, shape):
    """ Randomly initialize the kernel for the convolutions.

        :param key: random number generator
        :type key: jax.random key
        :param shape: shape of the kernel as (IOHW)
        :type shape: tuple(int)
        :returns: initialized kernel
        :rtype: jnp.array
    """
    return jax.random.normal(key, shape) / np.sqrt(np.prod(shape[2:]))

def bias_initializer(key, shape):
    """" Randomly initialize the bias for the convolutions.

         :param key: random number generator
         :type key : jax.random key
         :param shape: shape of the bias array
         :type length: int
         :returns: bias array
         :rtype: jnp.array
    """
    return jax.random.normal(key, shape) / np.sqrt(np.prod(shape))

def padding2(input_dim, output_dim, stride, kernel_size, dilation, inp_dilation):
    """Calculate the padding for a convolution.

    The padding is calculated to attain the desired input and output
    dimensions. If an asymmetric padding is needed, the function will
    return a non-zero residual value r.

    You can specify the asymmetric padding in JAX's conv_general_dilated by using
    a padding argument like: ((pad + r, pad), (pad + r, pad))

    :param input_dim: size of the input along the relevant dimension
    :type input_dim: int
    :param output_dim: height or width of the convolution output
    :type output_dim: int
    :param stride: stride
    :type stride: int
    :param kernel_size: kernel size
    :type kernel_size: int
    :param dilation: dilation
    :type dilation: int
    :param inp_dilation: dilation of the input
    :type inp_dilation: int
    :returns: padding for the convolution, residual value
    :rtype: int, int
    """
    pad = 0.5 * (stride * (output_dim - 1) - input_dim - (input_dim - 1) * (inp_dilation - 1)
             + dilation * (kernel_size - 1) + 1)
    r = math.ceil(pad % 1)
    return math.floor(pad), r

def create_random_params(key, config):
    """ Create the random parameters for the model.

        :param key: random number generator
        :type key: jax.random key
        :param config: configuration dictionary from the yaml file
        :type config: dict
        :returns: initialized parameters,
                  param dictionary for the direct convolution,
                  param dictionary for the transpose convolution,
                  shape of the hidden state
        :rtype: tuple(dict)
    """
    # Initialize the filters and biases
    h_channels = config['h_channels']  # number of channels in the hidden state
    ksx = config['inp_kernel_size']
    ksh = config['hid_kernel_size']
    kst = config['trans_kernel_size']
    # kernel_shape_x = (n_input_channels, n_output_channels, height, width)
    kernel_shape_x = (3, h_channels, ksx, ksx)  # kernel for input convolutions
    kernel_shape_h = (h_channels, h_channels, ksh, ksh)  # for hidden state convolutions
    kernel_shape_t = (h_channels, 3, kst, kst)  # kernel for the transpose convolution
    bias_shape = (h_channels,)
    strides = (config['s'], config['s'])  # stride for direct convolutions
    p = config['p'] 
    pad = ((p, p), (p, p))  # padding for direct convolutions
    d = config['d']
    inp_dilation = (d, d)  # dilation for the input in the direct convolutions
    kd = config['kd']
    ker_dilation = (kd, kd)  # kernel dilation in the direct convolutions
    vision = config['vision']
    print(f"Vision shape: {vision.shape}")
    gate_names = ['i', 'f', 'o', 'g']
    key, *l0_keys = random.split(key, 4)
    key0, *gate_keys = random.split(l0_keys[0], 2 * len(gate_names) + 1)
    params = {}

    for idx, name in enumerate(gate_names):
        key_x = gate_keys[2 * idx]
        key_h = gate_keys[2 * idx + 1]
        params['wx' + name] = kernel_initializer(key_x, kernel_shape_x)
        params['wh' + name] = kernel_initializer(key_h, kernel_shape_h)

    key1, *bias_keys = random.split(l0_keys[1], len(gate_names) + 1)
    for name, key in zip(gate_names, bias_keys):
        params['b' + name] = bias_initializer(key, bias_shape)

    # vision has shape (N, T, C, H, W). We will remove the temporal dimension T.
    dnx = lax.conv_dimension_numbers((vision.shape[0],) + vision.shape[2:],
                                    kernel_shape_x,     # only ndim matters, not shape 
                                    ('NCHW', 'IOHW', 'NCHW'))  # dimension meanings
    # calculate the shape of the hidden state
    h_shape = jax.eval_shape(partial(lax.conv_general_dilated, window_strides=strides,
                        padding=pad, lhs_dilation=inp_dilation, rhs_dilation=ker_dilation,
                        dimension_numbers=dnx), vision[:, 0, :, :, :], params['wxi']).shape

    print(f"hidden state shape: {h_shape}")
    print("(n_batch, n_channels, height, width)")

    # dimension numbers for the hidden state recursive convolutions
    dnh = lax.conv_dimension_numbers(h_shape, kernel_shape_h, ('NCHW', 'IOHW', 'NCHW')) 

    # kernel for the transposed convolution
    params['whx'] = kernel_initializer(l0_keys[2], kernel_shape_t)

    # Create parameters for the transpose convolution (assume square input and output)
    strides_t = (1, 1)  # stride for the transpose convolution
    inp_dilation_t = strides  # dilation for the input in the transpose convolution
    ker_dilation_t = (1, 1)  # kernel dilation in the transpose convolution
    p_t, r_t = padding2(h_shape[2],
                        vision.shape[3],
                        strides_t[0],
                        kernel_shape_t[2],
                        ker_dilation[0],
                        inp_dilation_t[0])
    pad_t = ((p_t + r_t, p_t), (p_t + r_t, p_t))  # padding for the transpose convolution

    dnt = lax.conv_dimension_numbers(h_shape, kernel_shape_t, ('NCHW', 'IOHW', 'NCHW'))

    conv_params = {'strides': strides,
                'padding': pad,
                'lhs_dilation': inp_dilation,
                'rhs_dilation': ker_dilation,
                'dnx': dnx,
                'dnh': dnh,
                }
    conv_params_t = {'strides': strides_t,
                    'padding': pad_t,
                    'lhs_dilation': inp_dilation_t,
                    'rhs_dilation': ker_dilation_t,
                    'dnt': dnt,
                    }
    return params, conv_params, conv_params_t, h_shape

def conv_lstm_forward(params, x, h, c, conv_params):
    """ A functional implementation of a convolutional LSTM.
    
        :param params: parameters of the model. A named tuple with following fields:
                       'wxi': kernel for input gate input convolution,
                       'whi': kernel for input gate hidden state convolution,
                       'bi': bias for input gate,
                       'wxf': kernel for forget gate input convolution,
                       'whf': kernel for forget gate hidden state convolution,
                       'bf': bias for forget gate,
                       'wxo': kernel for output gate input convolution,
                       'who': kernel for output gate hidden state convolution,
                       'bo': bias for output gate
                       'wxg': kernel for cell state input convolution,
                       'whg': kernel for cell state hidden state convolution,
                       'bg': bias for cell state,
        :type params: NamedTuple
        :param x: input to the LSTM cell
        :type x: jnp.array
        :param h: hidden state of the LSTM cell
        :type h: jnp.array
        :param c: cell state of the LSTM cell
        :type c: jnp.array
        :param conv_params: convolution hyperparameters NamedTuple with these entries:
                        'strides': tuple(int) with the strides for the input convolutions,
                        'padding': (str | Sequence[tuple(int, int)]) with padding
                        'lhs_dilation': tuple(int) with the input dilation
                        'rhs_dilation': tuple(int) with the kernel dilation
                        'dnx': dimension numbers for input convolutions,
                        'dnh': dimension numbers for hidden state convolutions,
        :returns: new hidden state and cell state of the LSTM cell
        :rtype: tuple(jnp.array)
    """
    s = conv_params.strides
    p = conv_params.padding
    ld = conv_params.lhs_dilation
    rd = conv_params.rhs_dilation
    dnx = conv_params.dnx
    dnh = conv_params.dnh
    # i = sigmoid(Conv(x, wxi) + Conv(h, whi) + bi)
    i = jax.nn.sigmoid(lax.conv_general_dilated(x, params.wxi, s, p, ld, rd, dnx) +
                       lax.conv_general_dilated(h,
                                                params.whi,
                                                (1, 1),
                                                'SAME',
                                                (1, 1),
                                                (1, 1), 
                                                dnh) + params.bi[:, None, None])
    # f = sigmoid(Conv(x, wxf) + Conv(h, whf) + bf)
    f = jax.nn.sigmoid(lax.conv_general_dilated(x, params.wxf, s, p, ld, rd, dnx) +
                       lax.conv_general_dilated(h,
                                                params.whf,
                                                (1, 1),
                                                'SAME',
                                                (1, 1),
                                                (1, 1),
                                                dnh) + params.bf[:, None, None])
    # o = sigmoid(Conv(x, wxo) + Conv(h, who) + bo)
    o = jax.nn.sigmoid(lax.conv_general_dilated(x, params.wxo, s, p, ld, rd, dnx) +
                       lax.conv_general_dilated(h,
                                                params.who,
                                                (1, 1),
                                                'SAME',
                                                (1, 1),
                                                (1, 1),
                                                dnh) + params.bo[:, None, None])
    # g = tanh(Conv(x, wxg) + Conv(h, whg) + bg)
    g = jax.lax.tanh(lax.conv_general_dilated(x, params.wxg, s, p, ld, rd, dnx) +
                     lax.conv_general_dilated(h,
                                              params.whg,
                                              (1, 1),
                                              'SAME',
                                              (1, 1),
                                              (1, 1),
                                              dnh) + params.bg[:, None, None])
    c = f * c + i * g
    h = o * jax.lax.tanh(c)

    return h, c
    

def prediction_step(params, x, h, c, conv_params, conv_params_t):
    """ A step of the convLSTM including a predicted next input.
    
        :param params: as in conv_lstm_forward, with the following extra entries:
                       'whx': kernel for hidden state to input convolution,
        :param x: input to the LSTM cell
        :type x: jnp.array
        :param h: hidden state of the LSTM cell
        :type h: jnp.array
        :param c: cell state of the LSTM cell
        :type c: jnp.array
        :param conv_params: as in conv_lstm_forward
        :type conv_params: NamedTuple
        :param conv_params_t: hyperparameters named tuple for the transposed convolution:
                    'strides': tuple(int) with the strides for the input convolutions,
                    'padding': (str | Sequence[tuple(int, int)]) with padding
                    'lhs_dilation': tuple(int) with the input dilation
                    'rhs_dilation': tuple(int) with the kernel dilation
                    'dnt': dimension numbers for transpose convolution
        :type conv_params_t: NamedTuple
        :returns: new hidden state, cell state of the LSTM cell, and predicted x
        :rtype: tuple(jnp.array)
    """
    h, c = conv_lstm_forward(params, x, h, c, conv_params)
    next_x = lax.conv_general_dilated(h,
                                      params.whx,
                                      conv_params_t.strides,
                                      conv_params_t.padding,
                                      conv_params_t.lhs_dilation,
                                      conv_params_t.rhs_dilation,
                                      conv_params_t.dnt,
                                     )
    return h, c, next_x


def prediction_n_steps(params, vision, h, c, conv_params, conv_params_t):
    """ All temporal steps of autoregressive prediction with a convolutional LSTM.

        All batches are done in parallel.

        :param params: parameters of the model, as in `prediction_step`
        :type params: NamedTuple
        :param vision: All inputs to the LSTM cell
        :type vision: jnp.array with shape (batches, T, channels, height, width)
        :param h: initial hidden state of the LSTM cell
        :type h: jnp.array
        :param c: initial cell state of the LSTM cell
        :type c: jnp.array
        :param conv_params: as in conv_lstm_forward
        :type conv_params: NamedTuple
        :param conv_params_t: in prediction_step
        :type conv_params_t: NamedTuple
        :returns: array with the predicted vision data
        :rtype: jnp.array with shape (batches, T, channels, height, width)
    """
    n = vision.shape[1]  # This will prevent jitting
    x_pred = jnp.zeros_like(vision)
    x = vision[:, 0, :, :, :]
    x_pred = x_pred.at[:, 0, :, :, :].set(x)

    for i in range(1, n):
        h, c, x = prediction_step(params, x, h, c, conv_params, conv_params_t)
        x_pred = x_pred.at[:, i, :, :, :].set(x)

    return x_pred

class ConvLSTMParams(NamedTuple):
    wxi: jnp.array
    whi: jnp.array
    bi: jnp.array
    wxf: jnp.array
    whf: jnp.array
    bf: jnp.array
    wxo: jnp.array
    who: jnp.array
    bo: jnp.array
    wxg: jnp.array
    whg: jnp.array
    bg: jnp.array
    whx: jnp.array

class ConvParams(NamedTuple):
    strides: tuple
    padding: tuple
    lhs_dilation: tuple
    rhs_dilation: tuple
    dnx: tuple
    dnh: tuple

class ConvParamsT(NamedTuple):
    strides: tuple
    padding: tuple
    lhs_dilation: tuple
    rhs_dilation: tuple
    dnt: tuple

def params_to_nt(params, conv_params, conv_params_t):
    """ Convert the parameters to named tuples.

        :param params: parameters of the model
        :type params: dict(str, jnp.array)
        :param conv_params: convolution hyperparameters dictionary
        :type conv_params: dict(str, tuple) | dict(str, ConvDimensionNumbers)
        :param conv_params_t: hyperparameters
        :type conv_params_t: dict(str, tuple) | dict(str, ConvDimensionNumbers)
        :returns: named tuples with the parameters
        :rtype: tuple(NamedTuple)
    """
    return (ConvLSTMParams(**{k: v for k, v in params.items()}),
            ConvParams(**{k: v for k, v in conv_params.items()}),
            ConvParamsT(**{k: v for k, v in conv_params_t.items()}))

def loss(params, vision, h, c, conv_params, conv_params_t):
    """ Calculate the loss of the model.

        :param params: parameters of the model
        :type params: dict(str, jnp.array)
        :param vision: input data
        :type vision: jnp.array
        :param h: initial hidden state
        :type h: jnp.array
        :param c: initial cell state
        :type c: jnp.array
        :param conv_params: convolution hyperparameters dictionary
        :type conv_params: dict(str, tuple) | dict(str, ConvDimensionNumbers)
        :param conv_params_t: hyperparameters dictionary for the transposed convolution
        :type conv_params_t: dict(str, tuple) | dict(str, ConvDimensionNumbers)
        :returns: loss value
        :rtype: jnp.array
    """
    x_pred = prediction_n_steps(params, vision, h, c, conv_params, conv_params_t)
    return jnp.mean((x_pred - vision) ** 2)

@Partial(jax.jit, static_argnums=(4, 5))
def sgd_update(params, vision, h, c, conv_params, conv_params_t, lr=1e-3):
    """ Update the parameters of the model using SGD.

        :param params: parameters of the model
        :type params: dict(str, jnp.array)
        :param vision: input data
        :type vision: jnp.array
        :param h: initial hidden state
        :type h: jnp.array
        :param c: initial cell state
        :type c: jnp.array
        :param conv_params: convolution hyperparameters dictionary
        :type conv_params: dict(str, tuple) | dict(str, ConvDimensionNumbers)
        :param conv_params_t: hyperparameters dictionary for the transposed convolution
        :type conv_params_t: dict(str, tuple) | dict(str, ConvDimensionNumbers)
        :param lr: learning rate
        :type lr: float
        :returns: loss value, new optimizer state and new parameters
        :rtype: tuple(jnp.array, dict, dict)
    """
    loss_val, grads = jax.value_and_grad(loss)(params, vision, h, c, conv_params, conv_params_t)
    #new_params = jax.tree.map(
    new_params = tree_map(
        lambda p, g: p - lr * g, params, grads
    )
    return loss_val, new_params


# ==============================================================================
# Main function
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python jax_conv_lstm.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = read_config(config_file)

    # Load the dataset, initialize the data loader
    dataset = JaxDataset(config['data_path'])
    print("Configuration dictionary:")
    print(config)
    data_loader = NumpyLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # explore the data_loader object
    for i, datum in enumerate(data_loader):
        vision, motor, language, mask, lang_mask = datum
        print(f"Batch {i}:")
        print(f"Vision shape: {vision.shape}. \t Vision type: {type(vision)}")
        print(f"Motor shape: {motor.shape}. \t Motor type: {type(motor)}")
        print(f"Language shape: {language.shape}. \t Language type: {type(language)}.")
        print(f"Mask shape: {mask.shape}. \t Mask type: {type(mask)}.")
        print(f"Lang_mask shape: {lang_mask.shape}. \t Lang_mask type: {type(lang_mask)}.")
        config['vision'] = vision[:, 0:1, :, :, :]  # for shape reference when creating parameters
        break

    # Initialize the parameters
    key = random.key(34)
    params, conv_params, conv_params_t, h_shape  = create_random_params(key, config)
    # convert parameters to named tuples    
    params_nt, conv_params_nt, conv_params_t_nt = params_to_nt(params, conv_params, conv_params_t)

    ## Gradient descent
    # initialize the hidden state and cell state
    key = random.key(23456)
    key_h, key_c = random.split(key)
    h = 0.1 * jax.random.normal(key_h, h_shape)
    c = 0.1 * jax.random.normal(key_c, h_shape)
    lr = config['learning_rate']
    losses = []

    # Run training epochs
    for epoch in range(config['n_epochs']):
        running_loss = 0.0
        for datum in data_loader:
            vision, motor, language, mask, lang_mask = datum
            loss_val, params_nt = sgd_update(params_nt,
                                             vision,
                                             h,
                                             c,
                                             conv_params_nt,
                                             conv_params_t_nt,
                                             lr)
            running_loss += loss_val
        losses.append(running_loss / len(data_loader))
        if epoch % 4 == 0:
            print(f"Epoch {epoch}, loss: {losses[-1]}")

    # Save the parameters of the model
    with open(config['model_path'], 'wb') as file:
        pickle.dump(params_nt, file)


