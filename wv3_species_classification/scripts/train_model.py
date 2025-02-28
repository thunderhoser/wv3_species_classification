"""Trains new model."""

import os
import copy
import time
import glob
import errno
import pickle
import argparse
import numpy
import pandas
import netCDF4
import sklearn.metrics
from sklearn.model_selection import KFold
import tensorflow
import tensorflow.linalg
import tensorflow.keras as tf_keras
import keras
import keras.callbacks
import keras.regularizers
from tensorflow.keras import backend as K
from keras.utils import to_categorical
from keras.metrics import TopKCategoricalAccuracy, top_k_categorical_accuracy
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, \
    BatchNormalization, LeakyReLU, Concatenate, Softmax

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

UNIQUE_SPECIES_NAMES_6CLASSES = [
    'POTR', 'Salix', 'PIEN', 'BEGL', 'PIFL', 'ABLA'
]
UNIQUE_SPECIES_NAMES_4CLASSES = ['PIEN', 'PIFL', 'ABLA', 'Other']
UNIQUE_SPECIES_NAMES_2CLASSES = ['PIFL', 'Other']

PATCH_IDS_KEY = 'patch_id_strings'
MULTISPECTRAL_RADIANCE_MATRIX_KEY = 'multispectral_radiance_matrix_w_m02'
MULTISPECTRAL_GRID_X_MATRIX_KEY = 'multispectral_grid_x_matrix_metres'
MULTISPECTRAL_GRID_Y_MATRIX_KEY = 'multispectral_grid_y_matrix_metres'
PANCHROMATIC_RADIANCE_MATRIX_KEY = 'panchromatic_radiance_matrix_w_m02'
PANCHROMATIC_GRID_X_MATRIX_KEY = 'panchromatic_grid_x_matrix_metres'
PANCHROMATIC_GRID_Y_MATRIX_KEY = 'panchromatic_grid_y_matrix_metres'
ELEVATION_MATRIX_KEY = 'elevation_matrix_m_asl'
ELEVATION_GRID_X_MATRIX_KEY = 'elevation_grid_x_matrix_metres'
ELEVATION_GRID_Y_MATRIX_KEY = 'elevation_grid_y_matrix_metres'

IMAGE_PATCH_KEYS = [
    MULTISPECTRAL_RADIANCE_MATRIX_KEY, PANCHROMATIC_RADIANCE_MATRIX_KEY,
    ELEVATION_MATRIX_KEY, PATCH_IDS_KEY
]

# NUM_GRID_ROWS_MULTISPECTRAL = 257
# NUM_GRID_COLUMNS_MULTISPECTRAL = 257
# NUM_CHANNELS_MULTISPECTRAL = 8
# NUM_GRID_ROWS_PANCHROMATIC = 1025
# NUM_GRID_COLUMNS_PANCHROMATIC = 1025

NUM_GRID_ROWS_MULTISPECTRAL = 129
NUM_GRID_COLUMNS_MULTISPECTRAL = 129
NUM_CHANNELS_MULTISPECTRAL = 8
NUM_GRID_ROWS_PANCHROMATIC = 513
NUM_GRID_COLUMNS_PANCHROMATIC = 513

NUM_CROSSVAL_FOLDS = 5
NUM_EXAMPLES_PER_BATCH = 64

METRIC_LIST = [
    TopKCategoricalAccuracy(k=1, name='top1_accuracy'),
    TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
    TopKCategoricalAccuracy(k=3, name='top3_accuracy')
]

PATCH_DIR_ARG_NAME = 'input_patch_dir_name'
NOISE_STDEV_ARG_NAME = 'noise_stdev'
NUM_NOISINGS_ARG_NAME = 'num_noisings'
NUM_CLASSES_ARG_NAME = 'num_classes'
USE_GERRITY_LOSS_ARG_NAME = 'use_gerrity_loss'
USE_CLASS_WEIGHTS_ARG_NAME = 'use_class_weights'
L2_WEIGHT_ARG_NAME = 'l2_weight'
NUM_DENSE_LAYERS_ARG_NAME = 'num_dense_layers'
DROPOUT_RATE_ARG_NAME = 'dropout_rate'
PUT_PIFL_FIRST_ARG_NAME = 'put_pifl_first_in_loss'
OUTPUT_DIR_ARG_NAME = 'top_output_dir_name'

PATCH_DIR_HELP_STRING = (
    'Name of directory containing NetCDF files with image patches.'
)
NOISE_STDEV_HELP_STRING = (
    'Standard deviation of Gaussian noise, used for data augmentation.'
)
NUM_NOISINGS_HELP_STRING = (
    'Number of times that each example is noised, used for data augmentation.'
)
NUM_CLASSES_HELP_STRING = 'Number of classes.  May be 6, 4, or 2.'
USE_GERRITY_LOSS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use negative Gerrity score (cross-entropy) '
    'as loss function.'
)
USE_CLASS_WEIGHTS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will weight minority classes more heavily in loss'
    ' function.'
)
L2_WEIGHT_HELP_STRING = 'L2 weight for all conv and dense layers.'
NUM_DENSE_LAYERS_HELP_STRING = 'Number of dense layers.'
DROPOUT_RATE_HELP_STRING = 'Dropout rate for dense layers.'
PUT_PIFL_FIRST_HELP_STRING = (
    'Boolean flag.  If 1, will put PIFL first in the loss function.  This '
    'matters only if the loss function is the Gerrity score, in which case '
    'PIFL is emphasized more than otherwise.'
)
OUTPUT_DIR_HELP_STRING = 'Name of top-level output directory.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_DIR_ARG_NAME, type=str, required=True,
    help=PATCH_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NOISE_STDEV_ARG_NAME, type=float, required=True,
    help=NOISE_STDEV_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_NOISINGS_ARG_NAME, type=int, required=True,
    help=NUM_NOISINGS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CLASSES_ARG_NAME, type=int, required=False, default=6,
    help=NUM_CLASSES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_GERRITY_LOSS_ARG_NAME, type=int, required=True,
    help=USE_GERRITY_LOSS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_CLASS_WEIGHTS_ARG_NAME, type=int, required=True,
    help=USE_CLASS_WEIGHTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + L2_WEIGHT_ARG_NAME, type=float, required=True,
    help=L2_WEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_DENSE_LAYERS_ARG_NAME, type=int, required=True,
    help=NUM_DENSE_LAYERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DROPOUT_RATE_ARG_NAME, type=float, required=True,
    help=DROPOUT_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PUT_PIFL_FIRST_ARG_NAME, type=int, required=True,
    help=PUT_PIFL_FIRST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_conv_layer(num_filters, l2_weight):
    """Creates convolutional layer.

    :param num_filters: Number of filters (output channels).
    :param l2_weight: Weight for L2 regularization.
    :return: conv_layer_object: Instance of `keras.layers.Conv2D`.
    """

    regularization_func = keras.regularizers.l1_l2(l1=0, l2=l2_weight)

    return Conv2D(
        filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=regularization_func,
        bias_regularizer=regularization_func
    )


def _get_batch_norm_layer():
    """Creates batch-normalization layer.

    :return: batch_norm_layer_object: Instance of
        `keras.layers.BatchNormalization`.
    """

    return BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True
    )


def _get_pooling_layer():
    """Creates pooling layer.

    :return: pooling_layer_object: Instance of `keras.layers.MaxPooling2D`.
    """

    return MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='same'
    )


def _get_dense_layer(num_output_neurons, l2_weight):
    """Creates dense layer.

    :param num_output_neurons: Number of output neurons.
    :param l2_weight: Weight for L2 regularization.
    :return: dense_layer_object: Instance of `keras.layers.Dense`.
    """

    regularization_func = keras.regularizers.l1_l2(l1=0, l2=l2_weight)

    return Dense(
        num_output_neurons, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=regularization_func,
        bias_regularizer=regularization_func
    )


def _get_a_for_gerrity_loss(confusion_tensor):
    """Returns vector a for Gerrity loss function.

    The equation for a is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param confusion_tensor: See doc for `_get_s_for_gerrity_score`.
    :return: a_value_tensor: See above.
    """

    num_examples = K.sum(confusion_tensor)
    num_examples_by_observed_class = K.sum(confusion_tensor, axis=0)
    cumul_freq_by_observed_class = K.cumsum(
        num_examples_by_observed_class / num_examples
    )

    return (
        (1. - cumul_freq_by_observed_class) /
        (cumul_freq_by_observed_class + K.epsilon())
    )


def _get_s_for_gerrity_loss(confusion_tensor, num_classes, i_matrix, j_matrix):
    """Returns matrix S for Gerrity loss function.

    The equation for S is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    K = number of classes

    :param confusion_tensor: K-by-K tensor, where entry [i, j] is the number of
        cases where the predicted label is i and correct label is j.
    :param num_classes: Number of classes.
    :param i_matrix: K-by-K numpy array of i-indices.
    :param j_matrix: K-by-K numpy array of j-indices.
    :return: s_tensor: See above.
    """

    a_value_tensor_1d = _get_a_for_gerrity_loss(confusion_tensor)
    a_reciprocal_tensor_1d = 1. / (a_value_tensor_1d + K.epsilon())

    cumul_a_reciprocal_tensor = K.cumsum(a_reciprocal_tensor_1d[:-1])
    cumul_a_reciprocal_tensor = K.concatenate((
        K.constant([0], shape=(1,)), cumul_a_reciprocal_tensor
    ))

    cumul_a_value_tensor = K.reverse(
        K.cumsum(K.reverse(a_value_tensor_1d[:-1], axes=0)),
        axes=0
    )
    cumul_a_value_tensor = K.concatenate((
        cumul_a_value_tensor, K.constant([0], shape=(1,))
    ))

    s_tensor = (
        tensorflow.gather(cumul_a_reciprocal_tensor, i_matrix)
        - (j_matrix - i_matrix)
        + tensorflow.gather(cumul_a_value_tensor, j_matrix)
    ) / (num_classes - 1)

    s_tensor_upper_triangle = tensorflow.linalg.band_part(s_tensor, 0, -1)
    return (
        s_tensor_upper_triangle -
        tensorflow.linalg.band_part(s_tensor, 0, 0) +
        K.transpose(s_tensor_upper_triangle)
    )


def mkdir_recursive_if_necessary(directory_name=None, file_name=None):
    """Creates directory if necessary (i.e., doesn't already exist).

    This method checks for the argument `directory_name` first.  If
    `directory_name` is None, this method checks for `file_name` and extracts
    the directory.

    :param directory_name: Path to local directory.
    :param file_name: Path to local file.
    """

    if directory_name is None:
        assert isinstance(file_name, str)
        directory_name = os.path.dirname(file_name)
    else:
        assert isinstance(directory_name, str)

    if directory_name == '':
        return

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise


def create_crossval_folds(patch_id_strings, num_folds, num_classes):
    """Creates cross-validation folds.

    E = number of examples
    F = number of folds

    :param patch_id_strings: length-E list of patch IDs.
    :param num_folds: Number of folds.
    :param num_classes: Number of classes.
    :return: fold_indices: length-E numpy array of indices ranging from
        0...(F - 1).
    """

    k_fold_object = KFold(n_splits=num_folds, shuffle=True)

    indiv_id_strings_by_patch = [s.split('_')[0] for s in patch_id_strings]
    indiv_id_strings_by_indiv, indiv_to_patch_indices = numpy.unique(
        numpy.array(indiv_id_strings_by_patch), return_inverse=True
    )

    species_id_strings_by_indiv = [
        s.split('_')[0][:-4] for s in indiv_id_strings_by_indiv
    ]
    num_individuals = len(indiv_id_strings_by_indiv)
    fold_indices_by_indiv = numpy.full(num_individuals, -1, dtype=int)

    if num_classes == 6:
        unique_species_names = copy.deepcopy(UNIQUE_SPECIES_NAMES_6CLASSES)
    else:
        species_id_strings_by_indiv = [
            s.replace('POTR', 'Other') for s in species_id_strings_by_indiv
        ]
        species_id_strings_by_indiv = [
            s.replace('Salix', 'Other') for s in species_id_strings_by_indiv
        ]
        species_id_strings_by_indiv = [
            s.replace('BEGL', 'Other') for s in species_id_strings_by_indiv
        ]

        if num_classes == 2:
            unique_species_names = copy.deepcopy(UNIQUE_SPECIES_NAMES_2CLASSES)

            species_id_strings_by_indiv = [
                s.replace('PIEN', 'Other') for s in species_id_strings_by_indiv
            ]
            species_id_strings_by_indiv = [
                s.replace('ABLA', 'Other') for s in species_id_strings_by_indiv
            ]
        else:
            unique_species_names = copy.deepcopy(UNIQUE_SPECIES_NAMES_4CLASSES)

    for k in range(len(unique_species_names)):
        these_indiv_indices = numpy.where(
            numpy.array(species_id_strings_by_indiv) ==
            unique_species_names[k]
        )[0]

        dummy_predictor_matrix = numpy.full(
            (len(these_indiv_indices), 1), 0.
        )
        current_fold_index = -1

        for _, these_subindices in k_fold_object.split(dummy_predictor_matrix):
            current_fold_index += 1
            fold_indices_by_indiv[these_indiv_indices[these_subindices]] = (
                current_fold_index
            )

    num_patches = len(patch_id_strings)
    fold_indices_by_patch = numpy.full(num_patches, -1, dtype=int)
    for i in range(num_individuals):
        fold_indices_by_patch[indiv_to_patch_indices == i] = (
            fold_indices_by_indiv[i]
        )

    assert numpy.all(fold_indices_by_patch >= 0)
    assert numpy.all(fold_indices_by_patch < num_folds)

    for i in range(num_patches):
        print('Patch {0:s} ... fold {1:d}'.format(
            patch_id_strings[i], fold_indices_by_patch[i]
        ))

    return fold_indices_by_patch


def read_image_patches(netcdf_file_name):
    """Reads image patches.

    P = number of patches
    mm = number of rows in elevation grid
    nn = number of columns in elevation grid
    m = number of rows in multispectral grid
    n = number of columns in multispectral grid
    M = number of rows in panchromatic grid
    N = number of columns in panchromatic grid
    C = number of spectral channels (wavelengths)

    :param netcdf_file_name: Path to input file.
    :return: patch_dict: Dictionary with the following keys.
    patch_dict['multispectral_radiance_matrix_w_m02']: P-by-m-by-n-by-C numpy
        array of radiances.
    patch_dict['multispectral_grid_x_matrix_metres']: P-by-n numpy array of
        x-coordinates.
    patch_dict['multispectral_grid_y_matrix_metres']: P-by-m numpy array of
        y-coordinates.
    patch_dict['panchromatic_radiance_matrix_w_m02']: P-by-M-by-N numpy array of
        radiances.
    patch_dict['panchromatic_grid_x_matrix_metres']: P-by-N numpy array of
        x-coordinates.
    patch_dict['panchromatic_grid_y_matrix_metres']: P-by-M numpy array of
        y-coordinates.
    patch_dict['elevation_matrix_m_asl']: P-by-mm-by-nn numpy array of
        elevations (metres above sea level).
    patch_dict['elevation_grid_x_matrix_metres']: P-by-nn numpy array of
        x-coordinates.
    patch_dict['elevation_grid_y_matrix_metres']: P-by-mm numpy array of
        y-coordinates.
    patch_dict['patch_id_strings']: length-P list of patch IDs.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    patch_dict = {
        MULTISPECTRAL_RADIANCE_MATRIX_KEY:
            dataset_object.variables[MULTISPECTRAL_RADIANCE_MATRIX_KEY][:],
        MULTISPECTRAL_GRID_X_MATRIX_KEY:
            dataset_object.variables[MULTISPECTRAL_GRID_X_MATRIX_KEY][:],
        MULTISPECTRAL_GRID_Y_MATRIX_KEY:
            dataset_object.variables[MULTISPECTRAL_GRID_Y_MATRIX_KEY][:],
        PANCHROMATIC_RADIANCE_MATRIX_KEY:
            dataset_object.variables[PANCHROMATIC_RADIANCE_MATRIX_KEY][:],
        PANCHROMATIC_GRID_X_MATRIX_KEY:
            dataset_object.variables[PANCHROMATIC_GRID_X_MATRIX_KEY][:],
        PANCHROMATIC_GRID_Y_MATRIX_KEY:
            dataset_object.variables[PANCHROMATIC_GRID_Y_MATRIX_KEY][:],
        ELEVATION_MATRIX_KEY:
            dataset_object.variables[ELEVATION_MATRIX_KEY][:],
        ELEVATION_GRID_X_MATRIX_KEY:
            dataset_object.variables[ELEVATION_GRID_X_MATRIX_KEY][:],
        ELEVATION_GRID_Y_MATRIX_KEY:
            dataset_object.variables[ELEVATION_GRID_Y_MATRIX_KEY][:],
        PATCH_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[PATCH_IDS_KEY][:])
        ]
    }

    for this_key in [
            MULTISPECTRAL_RADIANCE_MATRIX_KEY, PANCHROMATIC_RADIANCE_MATRIX_KEY,
            ELEVATION_MATRIX_KEY
    ]:
        patch_dict[this_key] = patch_dict[this_key].astype('float32')

    dataset_object.close()
    return patch_dict


def create_target_values(patch_dict, num_classes):
    """Creates target values.

    E = number of examples (patches)
    K = number of classes

    :param patch_dict: Dictionary in format returned by `read_image_patches`.
    :param num_classes: Number of classes.
    :return: target_matrix: E-by-K numpy array of zeros and ones.
    """

    patch_id_strings = patch_dict[PATCH_IDS_KEY]
    all_species_id_strings = [s.split('_')[0][:-4] for s in patch_id_strings]

    if num_classes == 6:
        unique_species_names = copy.deepcopy(UNIQUE_SPECIES_NAMES_6CLASSES)
    else:
        all_species_id_strings = [
            s.replace('POTR', 'Other') for s in all_species_id_strings
        ]
        all_species_id_strings = [
            s.replace('Salix', 'Other') for s in all_species_id_strings
        ]
        all_species_id_strings = [
            s.replace('BEGL', 'Other') for s in all_species_id_strings
        ]

        if num_classes == 2:
            unique_species_names = copy.deepcopy(UNIQUE_SPECIES_NAMES_2CLASSES)

            all_species_id_strings = [
                s.replace('PIEN', 'Other') for s in all_species_id_strings
            ]
            all_species_id_strings = [
                s.replace('ABLA', 'Other') for s in all_species_id_strings
            ]
        else:
            unique_species_names = copy.deepcopy(UNIQUE_SPECIES_NAMES_4CLASSES)

    all_species_enums = numpy.array(
        [unique_species_names.index(s) for s in all_species_id_strings],
        dtype=int
    )

    return to_categorical(
        all_species_enums, num_classes=num_classes
    )


def create_cnn(l2_weight, num_dense_layers, dropout_rate, num_classes,
               use_gerrity_loss, put_pifl_first_in_loss):
    """Creates CNN.

    :param l2_weight: See documentation at top of file.
    :param num_dense_layers: Same.
    :param dropout_rate: Same.
    :param num_classes: Same.
    :param use_gerrity_loss: Same.
    :param put_pifl_first_in_loss: Same.
    :return: model_object: Instance of `keras.models.Model`.
    """

    panchromatic_input_layer_object = Input(shape=(
        NUM_GRID_ROWS_PANCHROMATIC, NUM_GRID_COLUMNS_PANCHROMATIC, 1
    ))

    panchromatic_layer_object = _get_conv_layer(
        num_filters=2, l2_weight=l2_weight
    )(panchromatic_input_layer_object)

    panchromatic_layer_object = _get_conv_layer(
        num_filters=4, l2_weight=l2_weight
    )(panchromatic_layer_object)

    panchromatic_layer_object = LeakyReLU(alpha=0.2)(panchromatic_layer_object)
    panchromatic_layer_object = _get_batch_norm_layer()(
        panchromatic_layer_object
    )
    panchromatic_layer_object = _get_pooling_layer()(panchromatic_layer_object)

    panchromatic_layer_object = _get_conv_layer(
        num_filters=6, l2_weight=l2_weight
    )(panchromatic_layer_object)

    panchromatic_layer_object = _get_conv_layer(
        num_filters=8, l2_weight=l2_weight
    )(panchromatic_layer_object)

    panchromatic_layer_object = LeakyReLU(alpha=0.2)(panchromatic_layer_object)
    panchromatic_layer_object = _get_batch_norm_layer()(
        panchromatic_layer_object
    )
    panchromatic_layer_object = _get_pooling_layer()(panchromatic_layer_object)

    multispectral_input_layer_object = Input(shape=(
        NUM_GRID_ROWS_MULTISPECTRAL, NUM_GRID_COLUMNS_MULTISPECTRAL,
        NUM_CHANNELS_MULTISPECTRAL
    ))
    elevation_input_layer_object = Input(shape=(
        NUM_GRID_ROWS_MULTISPECTRAL, NUM_GRID_COLUMNS_MULTISPECTRAL, 1
    ))

    layer_object = Concatenate(axis=-1)([
        panchromatic_layer_object, multispectral_input_layer_object,
        elevation_input_layer_object
    ])

    layer_object = _get_conv_layer(
        num_filters=16, l2_weight=l2_weight
    )(layer_object)

    layer_object = _get_conv_layer(
        num_filters=16, l2_weight=l2_weight
    )(layer_object)

    layer_object = LeakyReLU(alpha=0.2)(layer_object)
    layer_object = _get_batch_norm_layer()(layer_object)
    layer_object = _get_pooling_layer()(layer_object)

    layer_object = _get_conv_layer(
        num_filters=24, l2_weight=l2_weight
    )(layer_object)

    layer_object = _get_conv_layer(
        num_filters=24, l2_weight=l2_weight
    )(layer_object)

    layer_object = LeakyReLU(alpha=0.2)(layer_object)
    layer_object = _get_batch_norm_layer()(layer_object)
    layer_object = _get_pooling_layer()(layer_object)

    layer_object = _get_conv_layer(
        num_filters=32, l2_weight=l2_weight
    )(layer_object)

    layer_object = _get_conv_layer(
        num_filters=32, l2_weight=l2_weight
    )(layer_object)

    layer_object = LeakyReLU(alpha=0.2)(layer_object)
    layer_object = _get_batch_norm_layer()(layer_object)
    layer_object = _get_pooling_layer()(layer_object)

    layer_object = _get_conv_layer(
        num_filters=48, l2_weight=l2_weight
    )(layer_object)

    layer_object = _get_conv_layer(
        num_filters=48, l2_weight=l2_weight
    )(layer_object)

    layer_object = LeakyReLU(alpha=0.2)(layer_object)
    layer_object = _get_batch_norm_layer()(layer_object)
    layer_object = _get_pooling_layer()(layer_object)

    layer_object = _get_conv_layer(
        num_filters=64, l2_weight=l2_weight
    )(layer_object)

    layer_object = _get_conv_layer(
        num_filters=64, l2_weight=l2_weight
    )(layer_object)

    layer_object = LeakyReLU(alpha=0.2)(layer_object)
    layer_object = _get_batch_norm_layer()(layer_object)
    layer_object = _get_pooling_layer()(layer_object)

    layer_object = _get_conv_layer(
        num_filters=128, l2_weight=l2_weight
    )(layer_object)

    layer_object = _get_conv_layer(
        num_filters=128, l2_weight=l2_weight
    )(layer_object)

    layer_object = LeakyReLU(alpha=0.2)(layer_object)
    layer_object = _get_batch_norm_layer()(layer_object)
    # layer_object = _get_pooling_layer()(layer_object)

    layer_object = Flatten()(layer_object)

    # The "3200" below is the number of flattened features.
    neuron_counts = numpy.logspace(
        numpy.log10(num_classes), numpy.log10(3200), num=num_dense_layers + 1
    )
    neuron_counts = neuron_counts[::-1][1:]
    neuron_counts = numpy.round(neuron_counts).astype(int)

    for k in range(num_dense_layers - 1):
        layer_object = _get_dense_layer(
            num_output_neurons=neuron_counts[k], l2_weight=l2_weight
        )(layer_object)

        layer_object = LeakyReLU(alpha=0.2)(layer_object)
        layer_object = Dropout(rate=dropout_rate)(layer_object)
        layer_object = _get_batch_norm_layer()(layer_object)

    layer_object = _get_dense_layer(
        num_output_neurons=neuron_counts[-1], l2_weight=l2_weight
    )(layer_object)

    layer_object = Softmax(axis=-1)(layer_object)

    model_object = keras.models.Model(
        inputs=[
            multispectral_input_layer_object, panchromatic_input_layer_object,
            elevation_input_layer_object
        ],
        outputs=layer_object
    )

    if use_gerrity_loss:
        model_object.compile(
            loss=gerrity_loss(
                num_classes=num_classes, put_pifl_first=put_pifl_first_in_loss
            ),
            optimizer=keras.optimizers.Adam(),
            metrics=METRIC_LIST[:num_classes]
        )
    else:
        model_object.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=METRIC_LIST[:num_classes]
        )

    model_object.summary()
    return model_object


def gerrity_loss(num_classes, put_pifl_first):
    """Returns function that computes Gerrity loss.

    :param num_classes: Number of classes.
    :param put_pifl_first: Boolean flag.  If True, will put PIFL first in the
        class list for the Gerrity score.  This will emphasize PIFL more than
        otherwise.
    :return: gerrity_loss_function: Function handle.
    """

    class_indices = numpy.linspace(
        0, num_classes - 1, num=num_classes, dtype=int
    )
    j_matrix, i_matrix = numpy.meshgrid(class_indices, class_indices)

    def loss(target_tensor, prediction_tensor):
        """Computes Gerrity loss (1 minus Gerrity score).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Gerrity loss.
        """

        if put_pifl_first:
            new_axis_order = numpy.linspace(
                0, target_tensor.shape[1] - 1,
                num=target_tensor.shape[1], dtype=int
            )

            if target_tensor.shape[1] == 6:
                pifl_index = UNIQUE_SPECIES_NAMES_6CLASSES.index('PIFL')
            elif target_tensor.shape[1] == 4:
                pifl_index = UNIQUE_SPECIES_NAMES_4CLASSES.index('PIFL')
            else:
                pifl_index = UNIQUE_SPECIES_NAMES_2CLASSES.index('PIFL')

            new_axis_order = numpy.delete(new_axis_order, pifl_index)
            new_axis_order = numpy.concatenate([
                numpy.array([pifl_index], dtype=int),
                new_axis_order
            ])

            new_axis_order = new_axis_order.tolist()
            print(new_axis_order)

            target_tensor = tensorflow.gather(
                target_tensor, new_axis_order, axis=1
            )
            prediction_tensor = tensorflow.gather(
                prediction_tensor, new_axis_order, axis=1
            )

        confusion_tensor = K.dot(K.transpose(prediction_tensor), target_tensor)
        # confusion_tensor = tensorflow.linalg.matmul(
        #     K.transpose(prediction_tensor), target_tensor
        # )
        print(confusion_tensor)

        s_tensor = _get_s_for_gerrity_loss(
            confusion_tensor=confusion_tensor, num_classes=num_classes,
            i_matrix=i_matrix, j_matrix=j_matrix
        )
        return -K.sum(confusion_tensor * s_tensor) / K.sum(confusion_tensor)

    return loss


def data_generator(
        predictor_matrices, target_matrix, noise_stdev, num_examples_per_batch,
        use_fast_data_aug, class_weight_dict):
    """Generator for training data.

    :param predictor_matrices: 1-D list of predictor matrices.
    :param target_matrix: Target matrix.
    :param noise_stdev: Standard deviation for Gaussian noise.
    :param num_examples_per_batch: Batch size.
    :param use_fast_data_aug: Boolean flag.  If True (False), will use fast
        (slow) data augmentation.
    :param class_weight_dict: Dictionary mapping target classes to weights.
        Each key in this dictionary is a class ID (non-negative integer), and
        the corresponding value is its weight in the loss function.  If you do
        not want to weight classes differently, make this argument None.
    :return: predictor_matrices_for_batch: 1-D list of predictor matrices.
    :return: target_matrix_for_batch: Target matrix.
    """

    predictor_matrices = [p.astype('float16') for p in predictor_matrices]
    target_matrix = target_matrix.astype('float32')
    num_matrices = len(predictor_matrices)

    perturbation_matrices = [
        numpy.random.normal(
            loc=0, scale=noise_stdev, size=predictor_matrices[k].shape
        ).astype('float16')
        for k in range(num_matrices - 1)
    ]

    for k in range(num_matrices - 1):
        print(numpy.mean(numpy.absolute(perturbation_matrices[k])))

    print('\n')
    for k in range(num_matrices):
        print(numpy.mean(predictor_matrices[k] > -4.))

    if use_fast_data_aug:
        aug_predictor_matrices = [
            predictor_matrices[k] + 0. if k == num_matrices - 1
            else predictor_matrices[k] + perturbation_matrices[k]
            for k in range(num_matrices)
        ]

    num_examples = target_matrix.shape[0]
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    num_batches = int(numpy.round(
        float(num_examples) / num_examples_per_batch
    ))
    num_batches_read = 0

    while True:
        if num_batches_read == 0:
            print('AUGMENTING AGAIN')
            exec_start_time_unix_sec = time.time()

            if not use_fast_data_aug:
                random_example_indices = numpy.random.choice(
                    example_indices, size=num_examples, replace=False
                )

                for k in range(num_matrices - 1):
                    perturbation_matrices[k] = perturbation_matrices[k][
                        random_example_indices, ...
                    ]

                aug_predictor_matrices = [
                    predictor_matrices[k] + 0. if k == num_matrices - 1
                    else predictor_matrices[k] + perturbation_matrices[k]
                    for k in range(num_matrices)
                ]

            shuffled_example_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=False
            )

            elapsed_time_sec = time.time() - exec_start_time_unix_sec
            print('ELAPSED TIME = {0:.1f} s'.format(elapsed_time_sec))

        first_index = num_batches_read * num_examples_per_batch
        last_index = first_index + num_examples_per_batch

        if last_index > num_examples:
            last_index = num_examples + 0
            first_index = last_index - num_examples_per_batch

        print((
            'Yielding {0:d}th to {1:d}th examples of randomized set...'
        ).format(
            first_index, last_index - 1
        ))

        these_indices = shuffled_example_indices[first_index:last_index]
        predictor_matrices_for_batch = [
            a[these_indices, ...] for a in aug_predictor_matrices
        ]
        target_matrix_for_batch = target_matrix[these_indices, ...]

        num_batches_read += 1
        if num_batches_read > num_batches:
            num_batches_read = 0

        if class_weight_dict is None:
            yield tuple(predictor_matrices_for_batch), target_matrix_for_batch
        else:
            target_classes = numpy.argmax(target_matrix_for_batch, axis=1)
            sample_weights = numpy.array(
                [class_weight_dict[c] for c in target_classes], dtype=float
            )

            yield (
                tuple(predictor_matrices_for_batch),
                target_matrix_for_batch,
                sample_weights
            )


def train_model(
        model_object, output_dir_name, noise_stdev, num_noisings,
        use_class_weights, training_predictor_matrices, training_target_matrix,
        validation_predictor_matrices, validation_target_matrix):
    """Trains CNN.

    :param model_object: Instance of `keras.models.Model`.
    :param output_dir_name: Path to output directory.
    :param noise_stdev: See documentation at top of file.
    :param num_noisings: Same.
    :param use_class_weights: Same.
    :param training_predictor_matrices: 1-D list of numpy arrays.
    :param training_target_matrix: numpy array.
    :param validation_predictor_matrices: 1-D list of numpy arrays.
    :param validation_target_matrix: numpy array.
    """

    mkdir_recursive_if_necessary(directory_name=output_dir_name)

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    mkdir_recursive_if_necessary(directory_name=backup_dir_name)

    model_file_name = '{0:s}/model.keras'.format(output_dir_name)
    history_file_name = '{0:s}/history.csv'.format(output_dir_name)

    try:
        history_table_pandas = pandas.read_csv(history_file_name)
        initial_epoch = history_table_pandas['epoch'].max() + 1
        best_validation_loss = history_table_pandas['val_loss'].min()
    except:
        initial_epoch = 0
        best_validation_loss = numpy.inf

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=True
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min',
        save_freq='epoch'
    )
    checkpoint_object.best = best_validation_loss

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0., patience=15, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.95,
        patience=5, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=False
    )

    list_of_callback_objects = [
        history_object, checkpoint_object,
        early_stopping_object, plateau_object,
        backup_object
    ]

    num_training_examples = training_target_matrix.shape[0]
    num_batches_per_epoch = int(numpy.round(
        num_noisings * float(num_training_examples) / NUM_EXAMPLES_PER_BATCH
    ))

    if use_class_weights:
        class_frequencies = numpy.mean(training_target_matrix, axis=0)
        class_weights = 1. / class_frequencies
        class_weights = numpy.minimum(class_weights, 50.)
        class_weights = numpy.log(class_weights)

        class_weight_dict = dict()
        for k in range(len(class_weights)):
            class_weight_dict[k] = class_weights[k]
    else:
        class_weight_dict = None

    training_generator = data_generator(
        predictor_matrices=training_predictor_matrices,
        target_matrix=training_target_matrix,
        noise_stdev=noise_stdev,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        use_fast_data_aug=False,
        class_weight_dict=class_weight_dict
    )

    for this_epoch in range(initial_epoch, 100):
        model_object.fit(
            x=training_generator,
            steps_per_epoch=num_batches_per_epoch,
            epochs=this_epoch + 1,
            initial_epoch=this_epoch,
            verbose=1,
            callbacks=list_of_callback_objects,
            validation_data=
            (validation_predictor_matrices, validation_target_matrix),
            validation_steps=None
        )


def evaluate_model(target_matrix, probability_matrix):
    """Evaluates model predictions.

    E = number of examples
    K = number of classes

    :param target_matrix: E-by-K numpy array of zeros and ones.
    :param probability_matrix: E-by-K numpy array of probabilities in range
        0...1.
    """

    class_frequencies = numpy.mean(target_matrix, axis=0)
    num_classes = len(class_frequencies)

    if num_classes == 6:
        unique_species_names = UNIQUE_SPECIES_NAMES_6CLASSES
    elif num_classes == 4:
        unique_species_names = UNIQUE_SPECIES_NAMES_4CLASSES
    else:
        unique_species_names = UNIQUE_SPECIES_NAMES_2CLASSES

    message_string = '; '.join([
        '{0:.3f} for {1:s}'.format(f, n)
        for f, n in zip(class_frequencies, unique_species_names)
    ])
    message_string = 'Class frequencies = {0:s}'.format(message_string)
    print(message_string)

    sort_indices = numpy.argsort(-1 * class_frequencies)
    loop_max = min([4, num_classes])

    for k in range(1, loop_max):
        print('\nTop-{0:d} accuracy of trivial model = {1:.3f}'.format(
            k, numpy.sum(class_frequencies[sort_indices[:k]])
        ))

        cnn_accuracy = numpy.mean(top_k_categorical_accuracy(
            y_true=target_matrix.astype(numpy.float32),
            y_pred=probability_matrix.astype(numpy.float32),
            k=k
        ))
        print('Top-{0:d} accuracy of CNN = {1:.3f}'.format(
            k, cnn_accuracy
        ))

    print('\n')

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=numpy.argmax(target_matrix, axis=1),
        y_pred=numpy.argmax(probability_matrix, axis=1),
        # labels=UNIQUE_SPECIES_NAMES
    )

    print('Confusion matrix:\n{0:s}'.format(str(confusion_matrix)))


def read_model(hdf5_file_name, num_classes, put_pifl_first_in_loss):
    """Reads trained model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :param num_classes: Number of classes.
    :param put_pifl_first_in_loss: Boolean flag.
    :return: model_object: Instance of `keras.models.Model`.
    """

    try:
        return tf_keras.models.load_model(hdf5_file_name)
    except:
        pass

    custom_object_dict = {
        'loss': gerrity_loss(
            num_classes=num_classes,
            put_pifl_first=put_pifl_first_in_loss
        )
    }
    model_object = tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict, compile=False
    )
    model_object.compile(
        loss=custom_object_dict['loss'], optimizer=keras.optimizers.Adam(),
        metrics=METRIC_LIST[:num_classes]
    )

    return model_object


def _run(image_patch_dir_name, noise_stdev, num_noisings, num_classes,
         use_gerrity_loss, use_class_weights, l2_weight, num_dense_layers,
         dropout_rate, put_pifl_first_in_loss, top_output_dir_name):
    """Trains new model.

    This is effectively the main method.

    :param image_patch_dir_name: See documentation at top of file.
    :param noise_stdev: Same.
    :param num_noisings: Same.
    :param num_classes: Same.
    :param use_gerrity_loss: Same.
    :param use_class_weights: Same.
    :param l2_weight: Same.
    :param num_dense_layers: Same.
    :param dropout_rate: Same.
    :param put_pifl_first_in_loss: Same.
    :param top_output_dir_name: Same.
    """

    assert num_classes in [2, 4, 6]

    image_patch_file_pattern = '{0:s}/image_patches[0-9][0-9].nc'.format(
        image_patch_dir_name
    )

    image_patch_file_names = glob.glob(image_patch_file_pattern)
    image_patch_file_names.sort()
    patch_id_strings = []

    for this_file_name in image_patch_file_names:
        print('Reading patch IDs from: "{0:s}"...'.format(this_file_name))
        this_patch_dict = read_image_patches(this_file_name)
        patch_id_strings += this_patch_dict[PATCH_IDS_KEY]

    num_examples = len(patch_id_strings)
    last_index = 0

    multispectral_radiance_matrix_w_m02 = None
    panchromatic_radiance_matrix_w_m02 = None
    elevation_matrix_m_agl = None

    for this_file_name in image_patch_file_names:
        print('Reading all data from: "{0:s}"...'.format(this_file_name))
        this_patch_dict = read_image_patches(this_file_name)

        if multispectral_radiance_matrix_w_m02 is None:
            dimensions = (
                (num_examples,) +
                this_patch_dict[MULTISPECTRAL_RADIANCE_MATRIX_KEY].shape[1:]
            )
            multispectral_radiance_matrix_w_m02 = numpy.full(
                dimensions, numpy.nan
            )

            dimensions = (
                (num_examples,) +
                this_patch_dict[PANCHROMATIC_RADIANCE_MATRIX_KEY].shape[1:]
            )
            panchromatic_radiance_matrix_w_m02 = numpy.full(
                dimensions, numpy.nan
            )

            dimensions = (
                (num_examples,) +
                this_patch_dict[ELEVATION_MATRIX_KEY].shape[1:]
            )
            elevation_matrix_m_agl = numpy.full(dimensions, numpy.nan)

        this_num_examples = len(this_patch_dict[PATCH_IDS_KEY])
        first_index = last_index + 0
        last_index = first_index + this_num_examples

        multispectral_radiance_matrix_w_m02[first_index:last_index, ...] = (
            this_patch_dict[MULTISPECTRAL_RADIANCE_MATRIX_KEY]
        )
        panchromatic_radiance_matrix_w_m02[first_index:last_index, ...] = (
            this_patch_dict[PANCHROMATIC_RADIANCE_MATRIX_KEY]
        )
        elevation_matrix_m_agl[first_index:last_index, ...] = (
            this_patch_dict[ELEVATION_MATRIX_KEY]
        )

    del this_patch_dict

    multispectral_radiance_matrix_w_m02 = (
        multispectral_radiance_matrix_w_m02.astype('float32')
    )
    panchromatic_radiance_matrix_w_m02 = (
        panchromatic_radiance_matrix_w_m02.astype('float32')
    )
    elevation_matrix_m_agl = elevation_matrix_m_agl.astype('float32')

    print(SEPARATOR_STRING)
    fold_indices = create_crossval_folds(
        patch_id_strings=patch_id_strings, num_folds=NUM_CROSSVAL_FOLDS,
        num_classes=num_classes
    )
    print(SEPARATOR_STRING)

    num_patches = len(fold_indices)
    oob_probability_matrix = numpy.full((num_patches, num_classes), numpy.nan)
    oob_target_matrix = numpy.full((num_patches, num_classes), -1, dtype=int)

    for k in range(NUM_CROSSVAL_FOLDS):
        training_indices = numpy.where(fold_indices != k)[0]
        validation_indices = numpy.where(fold_indices == k)[0]

        dummy_training_patch_dict = dict()
        dummy_validation_patch_dict = dict()

        for this_key in [PATCH_IDS_KEY]:
            dummy_training_patch_dict[this_key] = [
                patch_id_strings[i] for i in training_indices
            ]
            dummy_validation_patch_dict[this_key] = [
                patch_id_strings[i] for i in validation_indices
            ]

        training_target_matrix = create_target_values(
            patch_dict=dummy_training_patch_dict, num_classes=num_classes
        ).astype('float32')

        validation_target_matrix = create_target_values(
            patch_dict=dummy_validation_patch_dict, num_classes=num_classes
        ).astype('float32')

        multispectral_radiance_mean_w_m02 = numpy.nanmean(
            multispectral_radiance_matrix_w_m02[training_indices, ...],
            axis=(0, 1, 2), keepdims=True
        )
        multispectral_radiance_stdev_w_m02 = numpy.nanstd(
            multispectral_radiance_matrix_w_m02[training_indices, ...],
            axis=(0, 1, 2), ddof=1, keepdims=True
        )

        panchromatic_radiance_mean_w_m02 = numpy.nanmean(
            panchromatic_radiance_matrix_w_m02[training_indices, ...]
        )
        panchromatic_radiance_stdev_w_m02 = numpy.nanstd(
            panchromatic_radiance_matrix_w_m02[training_indices, ...], ddof=1
        )

        elevation_mean_m_asl = numpy.mean(
            elevation_matrix_m_agl[training_indices, ...]
        )
        elevation_stdev_m_asl = numpy.std(
            elevation_matrix_m_agl[training_indices, ...], ddof=1
        )

        training_predictor_matrices = [
            (multispectral_radiance_matrix_w_m02[training_indices, ...] -
             multispectral_radiance_mean_w_m02)
            / multispectral_radiance_stdev_w_m02,
            (panchromatic_radiance_matrix_w_m02[training_indices, ...] -
             panchromatic_radiance_mean_w_m02)
            / panchromatic_radiance_stdev_w_m02,
            (elevation_matrix_m_agl[training_indices, ...] -
             elevation_mean_m_asl)
            / elevation_stdev_m_asl
        ]

        validation_predictor_matrices = [
            (multispectral_radiance_matrix_w_m02[validation_indices, ...] -
             multispectral_radiance_mean_w_m02)
            / multispectral_radiance_stdev_w_m02,
            (panchromatic_radiance_matrix_w_m02[validation_indices, ...] -
             panchromatic_radiance_mean_w_m02)
            / panchromatic_radiance_stdev_w_m02,
            (elevation_matrix_m_agl[validation_indices, ...] -
             elevation_mean_m_asl)
            / elevation_stdev_m_asl
        ]

        print('NaN fractions:')

        for i in range(len(training_predictor_matrices)):
            print(numpy.mean(numpy.isnan(training_predictor_matrices[i])))
            print(numpy.mean(numpy.isnan(validation_predictor_matrices[i])))

            training_predictor_matrices[i][
                numpy.isnan(training_predictor_matrices[i])
            ] = -10.

            validation_predictor_matrices[i][
                numpy.isnan(validation_predictor_matrices[i])
            ] = -10.

        model_object = create_cnn(
            l2_weight=l2_weight, num_dense_layers=num_dense_layers,
            dropout_rate=dropout_rate, num_classes=num_classes,
            use_gerrity_loss=use_gerrity_loss,
            put_pifl_first_in_loss=put_pifl_first_in_loss
        )
        print(SEPARATOR_STRING)

        train_model(
            model_object=model_object,
            output_dir_name='{0:s}/fold{1:d}'.format(top_output_dir_name, k),
            noise_stdev=noise_stdev, num_noisings=num_noisings,
            use_class_weights=use_class_weights,
            training_predictor_matrices=training_predictor_matrices,
            training_target_matrix=training_target_matrix,
            validation_predictor_matrices=validation_predictor_matrices,
            validation_target_matrix=validation_target_matrix
        )

        del training_predictor_matrices

        model_object = read_model(
            hdf5_file_name=
            '{0:s}/fold{1:d}/model.keras'.format(top_output_dir_name, k),
            num_classes=num_classes,
            put_pifl_first_in_loss=put_pifl_first_in_loss
        )
        oob_target_matrix[validation_indices, :] = validation_target_matrix
        oob_probability_matrix[validation_indices, :] = model_object.predict(
            validation_predictor_matrices, batch_size=NUM_EXAMPLES_PER_BATCH
        )

        del validation_predictor_matrices

        print(SEPARATOR_STRING)
        evaluate_model(
            target_matrix=oob_target_matrix[validation_indices, :],
            probability_matrix=oob_probability_matrix[validation_indices, :]
        )
        print(SEPARATOR_STRING)

    evaluate_model(
        target_matrix=oob_target_matrix,
        probability_matrix=oob_probability_matrix
    )

    oob_prediction_file_name = '{0:s}/oob_predictions.p'.format(
        top_output_dir_name
    )
    print('Writing out-of-bag predictions to: "{0:s}"...'.format(
        oob_prediction_file_name
    ))

    pickle_file_handle = open(oob_prediction_file_name, 'wb')
    pickle.dump(oob_probability_matrix, pickle_file_handle)
    pickle.dump(oob_target_matrix, pickle_file_handle)
    pickle.dump(fold_indices, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        image_patch_dir_name=getattr(INPUT_ARG_OBJECT, PATCH_DIR_ARG_NAME),
        noise_stdev=getattr(INPUT_ARG_OBJECT, NOISE_STDEV_ARG_NAME),
        num_noisings=getattr(INPUT_ARG_OBJECT, NUM_NOISINGS_ARG_NAME),
        num_classes=getattr(INPUT_ARG_OBJECT, NUM_CLASSES_ARG_NAME),
        use_gerrity_loss=bool(
            getattr(INPUT_ARG_OBJECT, USE_GERRITY_LOSS_ARG_NAME)
        ),
        use_class_weights=bool(
            getattr(INPUT_ARG_OBJECT, USE_CLASS_WEIGHTS_ARG_NAME)
        ),
        l2_weight=getattr(INPUT_ARG_OBJECT, L2_WEIGHT_ARG_NAME),
        num_dense_layers=getattr(INPUT_ARG_OBJECT, NUM_DENSE_LAYERS_ARG_NAME),
        dropout_rate=getattr(INPUT_ARG_OBJECT, DROPOUT_RATE_ARG_NAME),
        put_pifl_first_in_loss=bool(
            getattr(INPUT_ARG_OBJECT, PUT_PIFL_FIRST_ARG_NAME)
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
