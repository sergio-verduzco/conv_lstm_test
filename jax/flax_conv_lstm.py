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
from flax import linen as nn

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
        return (self.vision[index].transpose(0, 2, 3, 1),
                self.motor[index],
                self.language[index],
                self.mask[index],
                self.lang_mask[index])

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


