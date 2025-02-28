"""Runs permutation test on trained model suite (one for each CV fold)."""

import os
import copy
import glob
import errno
import pickle
import argparse
import numpy
import netCDF4
from sklearn.metrics import auc as sklearn_auc
import tensorflow.linalg
import tensorflow.keras as tf_keras
import keras
from keras import backend as K
from keras.metrics import TopKCategoricalAccuracy

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'
NUM_EXAMPLES_PER_BATCH = 64

FIRST_PREDICTOR_NAMES = [
    'Coastal blue (400-450 nm)', 'Blue (450-510 nm)', 'Green (510-580 nm)',
    'Yellow (585-625 nm)', 'Red (630-690 nm)', 'Red edge (705-745 nm)',
    'Near-IR1 (770-895 nm)', 'Near-IR2 (860-1040 nm)'
]
SECOND_PREDICTOR_NAMES = ['Panchromatic (450-800 nm)']
THIRD_PREDICTOR_NAMES = ['Elevation']
PREDICTOR_NAMES_BY_MATRIX = [
    FIRST_PREDICTOR_NAMES, SECOND_PREDICTOR_NAMES, THIRD_PREDICTOR_NAMES
]

UNIQUE_SPECIES_NAMES_6CLASSES = [
    'ABLA', 'BEGL', 'PIEN', 'PIFL', 'POTR', 'Salix'
]
UNIQUE_SPECIES_NAMES_4CLASSES = ['ABLA', 'PIEN', 'PIFL', 'Other']
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

METRIC_LIST = [
    TopKCategoricalAccuracy(k=1, name='top1_accuracy'),
    TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
    TopKCategoricalAccuracy(k=3, name='top3_accuracy')
]

PREDICTORS_KEY = 'predictor_matrices'
PERMUTED_FLAGS_KEY = 'permuted_flags_by_matrix'
PERMUTED_MATRICES_KEY = 'permuted_matrix_indices'
PERMUTED_CHANNELS_KEY = 'permuted_channel_indices'
PERMUTED_COSTS_KEY = 'permuted_cost_matrix'
DEPERMUTED_MATRICES_KEY = 'depermuted_matrix_indices'
DEPERMUTED_CHANNELS_KEY = 'depermuted_channel_indices'
DEPERMUTED_COSTS_KEY = 'depermuted_cost_matrix'

ORIGINAL_COST_KEY = 'orig_cost_estimates'
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COSTS_KEY = 'best_cost_matrix'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COSTS_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG_KEY = 'is_backwards_test'

MODEL_DIR_ARG_NAME = 'input_model_dir_name'
PATCH_DIR_ARG_NAME = 'input_patch_dir_name'
PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
POSITIVE_CLASS_SPECIES_ARG_NAME = 'species_name_for_positive_class'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
BOOTSTRAP_BY_SHUFFLING_ARG_NAME = 'bootstrap_by_shuffling_many_times'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_DIR_HELP_STRING = (
    'Name of directory with trained models (one per cross-validation fold).'
)
PATCH_DIR_HELP_STRING = 'Name of directory with image patches (predictors).'
PREDICTION_FILE_HELP_STRING = (
    'Name of file with predictions and target values, produced by '
    'train_model.py.'
)
POSITIVE_CLASS_SPECIES_HELP_STRING = (
    'Species considered positive class.  If you specify a species, the cost '
    'function will be AUC for one-vs.-all classification.  If you leave this '
    'arg alone, the cost function will be the multiclass Gerrity score.'
)
DO_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1, will run backwards test.  If 0, will run forward '
    'test.'
)
NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates (used to compute the cost function after '
    'each permutation).  If you do not want bootstrapping, make this <= 1.'
)
BOOTSTRAP_BY_SHUFFLING_HELP_STRING = (
    'Boolean flag.  If 1, will shuffle the variable B times at each step, '
    'where B = `{0:s}`.  If 0, will shuffle only once, then bootstrap the cost '
    'function over all the data samples.'
)
OUTPUT_FILE_HELP_STRING = (
    'Results will be written to a Pickle file with this path.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DIR_ARG_NAME, type=str, required=True,
    help=MODEL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PATCH_DIR_ARG_NAME, type=str, required=True,
    help=PATCH_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + POSITIVE_CLASS_SPECIES_ARG_NAME, type=str, required=False,
    default='', help=POSITIVE_CLASS_SPECIES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=False, default=0,
    help=DO_BACKWARDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BOOTSTRAP_BY_SHUFFLING_ARG_NAME, type=int, required=False, default=0,
    help=BOOTSTRAP_BY_SHUFFLING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _read_predictions_and_targets(pickle_file_name):
    """Reads predictions and targets from Pickle file.

    K = number of classes
    E = number of examples
    F = number of cross-validation folds

    :param pickle_file_name: Path to input file.
    :return: probability_matrix: E-by-K numpy array of predicted probabilities.
    :return: target_matrix: E-by-K numpy array of true labels (all 0 or 1).
    :return: fold_indices: length-E numpy array of indices in 0...(F - 1).
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    probability_matrix = pickle.load(pickle_file_handle)
    target_matrix = pickle.load(pickle_file_handle)
    fold_indices = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return probability_matrix, target_matrix, fold_indices


def _read_model(hdf5_file_name, num_classes):
    """Reads trained model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :param num_classes: Number of classes.
    :return: model_object: Instance of `keras.models.Model`.
    """

    try:
        return tf_keras.models.load_model(hdf5_file_name)
    except:
        pass

    custom_object_dict = {
        'loss': gerrity_loss(num_classes=num_classes)
    }
    model_object = tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict, compile=False
    )
    model_object.compile(
        loss=custom_object_dict['loss'], optimizer=keras.optimizers.Adam(),
        metrics=METRIC_LIST[:num_classes]
    )

    return model_object


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


def _get_a_for_gerrity_cost(confusion_matrix_column_is_obs):
    """Returns vector a for Gerrity-score cost function.

    The equation for a is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    K = number of classes

    :param confusion_matrix_column_is_obs: K-by-K numpy array, where entry
        [i, j] is the number of examples where the [i]th class is predicted and
        [j]th class is observed.
    :return: a_vector: See above.
    """

    num_examples = numpy.sum(confusion_matrix_column_is_obs)

    num_examples_by_class = numpy.sum(confusion_matrix_column_is_obs, axis=0)
    cumulative_freq_by_class = numpy.cumsum(
        num_examples_by_class.astype(float) / num_examples
    )

    return (1. - cumulative_freq_by_class) / cumulative_freq_by_class


def _get_s_for_gerrity_cost(confusion_matrix_column_is_obs):
    """Returns matrix S for Gerrity-score cost function.

    The equation for S is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    K = number of classes

    :param confusion_matrix_column_is_obs: K-by-K numpy array, where entry
        [i, j] is the number of examples where the [i]th class is predicted and
        [j]th class is observed.
    :return: s_matrix: See above.
    """

    a_vector = _get_a_for_gerrity_cost(confusion_matrix_column_is_obs)
    a_vector_reciprocal = 1. / a_vector

    num_classes = confusion_matrix_column_is_obs.shape[0]
    s_matrix = numpy.full((num_classes, num_classes), numpy.nan)

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                s_matrix[i, j] = (
                    numpy.sum(a_vector_reciprocal[:i]) +
                    numpy.sum(a_vector[i:-1])
                )
                continue

            if i > j:
                s_matrix[i, j] = s_matrix[j, i]
                continue

            s_matrix[i, j] = (
                numpy.sum(a_vector_reciprocal[:i]) - (j - i) +
                numpy.sum(a_vector[j:-1])
            )

    return s_matrix / (num_classes - 1)


def gerrity_loss(num_classes):
    """Returns function that computes Gerrity loss.

    :param num_classes: Number of classes.
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

        confusion_tensor = K.dot(K.transpose(prediction_tensor), target_tensor)

        s_tensor = _get_s_for_gerrity_loss(
            confusion_tensor=confusion_tensor, num_classes=num_classes,
            i_matrix=i_matrix, j_matrix=j_matrix
        )
        return -K.sum(confusion_tensor * s_tensor) / K.sum(confusion_tensor)

    return loss


def _read_image_patches(netcdf_file_name):
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


def make_prediction_function(model_objects, fold_indices):
    """Creates prediction function.

    F = number of cross-validation folds
    E = number of examples

    :param model_objects: length-F list of trained models.
    :param fold_indices: length-E numpy array of indices in 0...(F - 1).
    :return: prediction_function: Function (see below).
    """

    def prediction_function(predictor_matrices_norm):
        """Runs model in inference mode (to make predictions for new data).

        E = number of examples
        K = number of classes

        :param predictor_matrices_norm: length-3 list of normalized predictor
            matrices.  Each list item should be a numpy array where the first
            axis has length E.
        :return: class_probability_matrix: E-by-K numpy array of class
            probabilities.
        """

        num_folds = numpy.max(fold_indices) + 1
        num_examples = len(fold_indices)
        class_probability_matrix = numpy.array([])

        for k in range(num_folds):
            print('Applying model validated with {0:d}th fold...'.format(k + 1))
            these_example_indices = numpy.where(fold_indices == k)[0]

            this_prob_matrix = model_objects[k].predict(
                [p[these_example_indices, ...] for p in predictor_matrices_norm],
                batch_size=NUM_EXAMPLES_PER_BATCH
            )

            if class_probability_matrix.size == 0:
                num_classes = this_prob_matrix.shape[1]
                class_probability_matrix = numpy.full(
                    (num_examples, num_classes), numpy.nan
                )

            class_probability_matrix[these_example_indices, :] = (
                this_prob_matrix + 0.
            )

        return class_probability_matrix

    return prediction_function


def _permute_values(
        predictor_matrices, matrix_index, channel_index, permuted_values=None):
    """Permutes (shuffles) values for one predictor.

    E = number of examples

    :param predictor_matrices: length-3 list of numpy arrays.  The first axis of
        each numpy array must have length E.
    :param matrix_index: Will permute the [k]th channel in the [i]th matrix,
        where i = `matrix_index`.
    :param channel_index: Will permute the [k]th channel in the [i]th matrix,
        where k = `channel_index`.
    :param permuted_values: numpy array of permuted values with which to replace
        clean values.  If None, permuted values will be created randomly on the
        fly.
    :return: predictor_matrices: Same as input but after depermuting.
    :return: permuted_values: numpy array of permuted values with which clean
        values were replaced.
    """

    i = matrix_index
    k = channel_index

    if permuted_values is None:
        random_indices = numpy.random.permutation(
            predictor_matrices[i].shape[0]
        )
        predictor_matrices[i][..., k] = (
            predictor_matrices[i][random_indices, ..., k]
        )
    else:
        predictor_matrices[i][..., k] = permuted_values + 0.

    permuted_values = predictor_matrices[i][..., k] + 0.
    return predictor_matrices, permuted_values


def _depermute_values(
        predictor_matrices, clean_predictor_matrices, matrix_index,
        channel_index):
    """Depermutes (cleans up) values for one predictor.

    E = number of examples

    :param predictor_matrices: length-3 list of numpy arrays.  The first axis of
        each numpy array must have length E.
    :param clean_predictor_matrices: Clean version of `predictor_matrices`, with
        no variables permuted.
    :param matrix_index: Will depermute the [k]th channel in the [i]th matrix,
        where i = `matrix_index`.
    :param channel_index: Will depermute the [k]th channel in the [i]th matrix,
        where k = `channel_index`.
    :return: predictor_matrices: Same as input but after depermuting.
    """

    i = matrix_index
    k = channel_index

    predictor_matrices[i][..., k] = clean_predictor_matrices[i][..., k] + 0.
    return predictor_matrices


def _make_gerrity_cost_function():
    """Creates cost function.

    :return: cost_function: Function (see below).
    """

    def cost_function(target_matrix, probability_matrix):
        """Actual cost function.

        E = number of examples
        K = number of classes

        :param target_matrix: E-by-K numpy array of true labels in range 0...1.
        :param probability_matrix: E-by-K numpy array of probabilities.
        :return: gerrity_score: Gerrity score (scalar).
        """

        num_classes = target_matrix.shape[1]
        confusion_matrix_column_is_obs = numpy.full(
            (num_classes, num_classes), -1, dtype=int
        )

        observed_labels = numpy.argmax(target_matrix, axis=1)
        predicted_labels = numpy.argmax(probability_matrix, axis=1)

        for i in range(num_classes):
            for j in range(num_classes):
                confusion_matrix_column_is_obs[i, j] = numpy.sum(numpy.logical_and(
                    predicted_labels == i, observed_labels == j
                ))

        s_matrix = _get_s_for_gerrity_cost(confusion_matrix_column_is_obs)
        num_examples = numpy.sum(confusion_matrix_column_is_obs)

        return (
            numpy.sum(confusion_matrix_column_is_obs * s_matrix) / num_examples
        )

    return cost_function


def _get_points_in_roc_curve(observed_labels, forecast_probabilities):
    """Creates points for ROC curve.

    E = number of examples
    T = number of binarization thresholds

    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :return: pofd_by_threshold: length-T numpy array of POFD (probability of
        false detection) values.
    :return: pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :return: binarization_thresholds: length-T numpy array of thresholds
        themselves.
    """

    assert numpy.all(numpy.logical_or(
        observed_labels == 0, observed_labels == 1
    ))

    assert numpy.all(numpy.logical_and(
        forecast_probabilities >= 0, forecast_probabilities <= 1
    ))

    observed_labels = observed_labels.astype(int)
    binarization_thresholds = numpy.linspace(0, 1, num=1001, dtype=float)

    num_thresholds = len(binarization_thresholds)
    pofd_by_threshold = numpy.full(num_thresholds, numpy.nan)
    pod_by_threshold = numpy.full(num_thresholds, numpy.nan)

    for k in range(num_thresholds):
        these_forecast_labels = (
            forecast_probabilities >= binarization_thresholds[k]
        ).astype(int)

        this_num_hits = numpy.sum(numpy.logical_and(
            these_forecast_labels == 1, observed_labels == 1
        ))

        this_num_false_alarms = numpy.sum(numpy.logical_and(
            these_forecast_labels == 1, observed_labels == 0
        ))

        this_num_misses = numpy.sum(numpy.logical_and(
            these_forecast_labels == 0, observed_labels == 1
        ))

        this_num_correct_nulls = numpy.sum(numpy.logical_and(
            these_forecast_labels == 0, observed_labels == 0
        ))

        try:
            pofd_by_threshold[k] = (
                float(this_num_false_alarms) /
                (this_num_false_alarms + this_num_correct_nulls)
            )
        except ZeroDivisionError:
            pass

        try:
            pod_by_threshold[k] = (
                float(this_num_hits) / (this_num_hits + this_num_misses)
            )
        except ZeroDivisionError:
            pass

    pod_by_threshold = numpy.array([1.] + pod_by_threshold.tolist() + [0.])
    pofd_by_threshold = numpy.array([1.] + pofd_by_threshold.tolist() + [0.])

    return pofd_by_threshold, pod_by_threshold, binarization_thresholds


def _make_auc_cost_function(num_classes, species_name_for_positive_class):
    """Creates AUC (area under ROC curve) cost function.

    :param num_classes: Number of classes.
    :param species_name_for_positive_class: Species name considered the positive
        class.  Everything else will be considered the negative class.
    :return: cost_function: Function (see below).
    """

    if num_classes == 6:
        positive_class_index = UNIQUE_SPECIES_NAMES_6CLASSES.index(
            species_name_for_positive_class
        )
    elif num_classes == 4:
        positive_class_index = UNIQUE_SPECIES_NAMES_4CLASSES.index(
            species_name_for_positive_class
        )
    elif num_classes == 2:
        positive_class_index = UNIQUE_SPECIES_NAMES_2CLASSES.index(
            species_name_for_positive_class
        )

    def cost_function(target_matrix, probability_matrix):
        """Actual cost function.

        E = number of examples
        K = number of classes

        :param target_matrix: E-by-K numpy array of true labels in range 0...1.
        :param probability_matrix: E-by-K numpy array of probabilities.
        :return: area_under_roc_curve: Area under ROC curve (scalar).
        """

        pofd_by_threshold, pod_by_threshold, _ = _get_points_in_roc_curve(
            observed_labels=target_matrix[:, positive_class_index],
            forecast_probabilities=probability_matrix[:, positive_class_index]
        )

        return sklearn_auc(x=pofd_by_threshold, y=pod_by_threshold)

    return cost_function


def _bootstrap_cost(target_matrix, probability_matrix, cost_function,
                    num_replicates):
    """Uses bootstrapping to estimate cost.

    E = number of examples
    K = number of classes

    :param target_matrix: E-by-K numpy array of true labels in range 0...1.
    :param probability_matrix: E-by-K numpy array of probabilities.
    :param cost_function: Cost function.  Must be positively oriented (i.e.,
        higher is better), with the following inputs and outputs.
    Input: target_matrix: See above.
    Input: probability_matrix: See above.
    Output: cost: Scalar value.

    :param num_replicates: Number of bootstrap replicates (i.e., number of times
        to estimate cost).
    :return: cost_estimates: length-B numpy array of cost estimates, where B =
        number of bootstrap replicates.
    """

    cost_estimates = numpy.full(num_replicates, numpy.nan)

    if num_replicates == 1:
        cost_estimates[0] = cost_function(target_matrix, probability_matrix)
    else:
        num_examples = target_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )

        for k in range(num_replicates):
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

            cost_estimates[k] = cost_function(
                target_matrix[these_indices, ...],
                probability_matrix[these_indices, ...]
            )

    print('Average cost estimate over {0:d} replicates = {1:f}'.format(
        num_replicates, numpy.mean(cost_estimates)
    ))

    return cost_estimates


def _run_forward_test_one_step(
        predictor_matrices, target_matrix, prediction_function, cost_function,
        permuted_flags_by_matrix, num_bootstrap_reps,
        bootstrap_by_shuffling_many_times):
    """Runs one step of the forward permutation test.

    E = number of examples
    K = number of classes

    :param predictor_matrices: length-3 list of numpy arrays.  The first axis of
        each numpy array must have length E.
    :param target_matrix: E-by-K numpy array of true labels, all integers in
        0...1.
    :param prediction_function: Function with the following inputs and outputs.
    Input: predictor_matrices: See above.
    Output: probability_matrix: E-by-K numpy array of class probabilities.

    :param cost_function: See doc for `_bootstrap_cost`.
    :param permuted_flags_by_matrix: length-3 list of numpy arrays.  The [i]th
        list item is a Boolean array with length C_i -- C_i being the number of
        channels in the [i]th predictor matrix -- indicating which channels are
        already permuted.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param bootstrap_by_shuffling_many_times: Boolean flag.  If True, will
        shuffle the variable B times, where B = `num_bootstrap_reps`.  If False,
        will shuffle only once, then bootstrap the cost function over all the
        data samples.

    :return: result_dict: Dictionary with the following keys, where P = number
        of permutations done in this step and B = number of bootstrap
        replicates.
    result_dict['predictor_matrices']: Same as input but with more values
        permuted.
    result_dict['permuted_flags_by_matrix']: Same as input but with more `True`
        flags.
    result_dict['permuted_matrix_indices']: length-P numpy array with matrix
        indices for predictors permuted.
    result_dict['permuted_channel_indices']: length-P numpy array with channel
        indices for predictors permuted.
    result_dict['permuted_cost_matrix']: P-by-B numpy array of costs after
        permutation.
    """

    if all([numpy.all(f) for f in permuted_flags_by_matrix]):
        return None

    # Housekeeping.
    permuted_matrix_indices = []
    permuted_channel_indices = []
    permuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    best_cost = numpy.inf
    best_matrix_index = -1
    best_channel_index = -1
    best_permuted_values = None

    for i in range(len(predictor_matrices)):
        this_num_channels = predictor_matrices[i].shape[-1]

        for k in range(this_num_channels):
            if permuted_flags_by_matrix[i][k]:
                continue

            permuted_matrix_indices.append(i)
            permuted_channel_indices.append(k)

            if bootstrap_by_shuffling_many_times:
                these_predictor_matrices = copy.deepcopy(predictor_matrices)
                this_cost_matrix = numpy.full(num_bootstrap_reps, numpy.nan)

                for j in range(num_bootstrap_reps):
                    print((
                        'Permuting {0:d}th of {1:d} channels in {2:d}th of '
                        '{3:d} predictor matrices with {4:d}th of {5:d} random '
                        'seeds...'
                    ).format(
                        k + 1, this_num_channels,
                        i + 1, len(predictor_matrices),
                        j + 1, num_bootstrap_reps
                    ))

                    these_predictor_matrices, _ = _permute_values(
                        predictor_matrices=these_predictor_matrices,
                        matrix_index=i, channel_index=k
                    )
                    this_prob_matrix = prediction_function(
                        these_predictor_matrices
                    )
                    this_cost_matrix[j] = cost_function(
                        target_matrix, this_prob_matrix
                    )
            else:
                print((
                    'Permuting {0:d}th of {1:d} channels in {2:d}th of {3:d} '
                    'predictor matrices...'
                ).format(
                    k + 1, this_num_channels, i + 1, len(predictor_matrices)
                ))

                these_predictor_matrices, these_permuted_values = (
                    _permute_values(
                        predictor_matrices=copy.deepcopy(predictor_matrices),
                        matrix_index=i, channel_index=k
                    )
                )
                this_prob_matrix = prediction_function(these_predictor_matrices)
                this_cost_matrix = _bootstrap_cost(
                    target_matrix=target_matrix,
                    probability_matrix=this_prob_matrix,
                    cost_function=cost_function,
                    num_replicates=num_bootstrap_reps
                )

            this_cost_matrix = numpy.reshape(
                this_cost_matrix, (1, this_cost_matrix.size)
            )
            permuted_cost_matrix = numpy.concatenate(
                (permuted_cost_matrix, this_cost_matrix), axis=0
            )
            this_average_cost = numpy.mean(permuted_cost_matrix[-1, :])
            if this_average_cost > best_cost:
                continue

            best_cost = this_average_cost + 0.
            best_matrix_index = i
            best_channel_index = k

            if bootstrap_by_shuffling_many_times:
                _, best_permuted_values = _permute_values(
                    predictor_matrices=copy.deepcopy(predictor_matrices),
                    matrix_index=best_matrix_index,
                    channel_index=best_channel_index
                )
            else:
                best_permuted_values = these_permuted_values

    predictor_matrices = _permute_values(
        predictor_matrices=predictor_matrices,
        matrix_index=best_matrix_index, channel_index=best_channel_index,
        permuted_values=best_permuted_values
    )[0]

    print((
        'Best predictor = {0:d}th channel in {1:d}th matrix (cost = {2:.4f})'
    ).format(
        best_channel_index + 1, best_matrix_index + 1, best_cost
    ))

    permuted_flags_by_matrix[best_matrix_index][best_channel_index] = True
    permuted_matrix_indices = numpy.array(permuted_matrix_indices, dtype=int)
    permuted_channel_indices = numpy.array(permuted_channel_indices, dtype=int)

    return {
        PREDICTORS_KEY: predictor_matrices,
        PERMUTED_FLAGS_KEY: permuted_flags_by_matrix,
        PERMUTED_MATRICES_KEY: permuted_matrix_indices,
        PERMUTED_CHANNELS_KEY: permuted_channel_indices,
        PERMUTED_COSTS_KEY: permuted_cost_matrix,
    }


def _run_backwards_test_one_step(
        predictor_matrices, clean_predictor_matrices, target_matrix,
        prediction_function, cost_function, permuted_flags_by_matrix,
        num_bootstrap_reps, bootstrap_by_shuffling_many_times):
    """Runs one step of the backwards permutation test.

    :param predictor_matrices: See doc for `_run_forward_test_one_step`.
    :param clean_predictor_matrices: Clean version of `predictor_matrices`, with
        no variables permuted.
    :param target_matrix: See doc for `_run_forward_test_one_step`.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param permuted_flags_by_matrix: Same.
    :param num_bootstrap_reps: Same.
    :param bootstrap_by_shuffling_many_times: Same.
    :return: result_dict: Dictionary with the following keys, where P = number
        of permutations done in this step and B = number of bootstrap
        replicates.
    result_dict['predictor_matrices']: Same as input but with fewer values
        permuted.
    result_dict['permuted_flags_by_matrix']: Same as input but with more `False`
        flags.
    result_dict['depermuted_matrix_indices']: length-P numpy array with matrix
        indices for predictors depermuted.
    result_dict['depermuted_channel_indices']: length-P numpy array with channel
        indices for predictors depermuted.
    result_dict['depermuted_cost_matrix']: P-by-B numpy array of costs after
        depermutation.
    """

    if all([not numpy.any(f) for f in permuted_flags_by_matrix]):
        return None

    # Housekeeping.
    depermuted_matrix_indices = []
    depermuted_channel_indices = []
    depermuted_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    best_cost = -numpy.inf
    best_matrix_index = -1
    best_channel_index = -1

    for i in range(len(predictor_matrices)):
        this_num_channels = predictor_matrices[i].shape[-1]

        for k in range(this_num_channels):
            if not permuted_flags_by_matrix[i][k]:
                continue

            depermuted_matrix_indices.append(i)
            depermuted_channel_indices.append(k)

            if bootstrap_by_shuffling_many_times:
                these_predictor_matrices = _depermute_values(
                    predictor_matrices=copy.deepcopy(predictor_matrices),
                    clean_predictor_matrices=clean_predictor_matrices,
                    matrix_index=i, channel_index=k
                )

                this_cost_matrix = numpy.full(num_bootstrap_reps, numpy.nan)

                for j in range(num_bootstrap_reps):
                    print((
                        'Depermuting {0:d}th of {1:d} channels in {2:d}th of '
                        '{3:d} predictor matrices with {4:d}th of {5:d} random '
                        'seeds...'
                    ).format(
                        k + 1, this_num_channels,
                        i + 1, len(predictor_matrices),
                        j + 1, num_bootstrap_reps
                    ))

                    for i_minor in range(len(predictor_matrices)):
                        for k_minor in range(
                                predictor_matrices[i_minor].shape[-1]
                        ):
                            if not permuted_flags_by_matrix[i_minor][k_minor]:
                                continue

                            these_predictor_matrices, _ = _permute_values(
                                predictor_matrices=these_predictor_matrices,
                                matrix_index=i_minor, channel_index=k_minor
                            )

                    this_prob_matrix = prediction_function(
                        these_predictor_matrices
                    )
                    this_cost_matrix[j] = cost_function(
                        target_matrix, this_prob_matrix
                    )
            else:
                print((
                    'Depermuting {0:d}th of {1:d} channels in {2:d}th of {3:d} '
                    'predictor matrices...'
                ).format(
                    k + 1, this_num_channels, i + 1, len(predictor_matrices)
                ))

                these_predictor_matrices = _depermute_values(
                    predictor_matrices=copy.deepcopy(predictor_matrices),
                    clean_predictor_matrices=clean_predictor_matrices,
                    matrix_index=i, channel_index=k
                )
                this_prob_matrix = prediction_function(these_predictor_matrices)
                this_cost_matrix = _bootstrap_cost(
                    target_matrix=target_matrix,
                    probability_matrix=this_prob_matrix,
                    cost_function=cost_function,
                    num_replicates=num_bootstrap_reps
                )

            this_cost_matrix = numpy.reshape(
                this_cost_matrix, (1, this_cost_matrix.size)
            )
            depermuted_cost_matrix = numpy.concatenate(
                (depermuted_cost_matrix, this_cost_matrix), axis=0
            )
            this_average_cost = numpy.mean(depermuted_cost_matrix[-1, :])
            if this_average_cost < best_cost:
                continue

            best_cost = this_average_cost + 0.
            best_matrix_index = i
            best_channel_index = k

    predictor_matrices = _depermute_values(
        predictor_matrices=predictor_matrices,
        clean_predictor_matrices=clean_predictor_matrices,
        matrix_index=best_matrix_index, channel_index=best_channel_index
    )

    print((
        'Best predictor = {0:d}th channel in {1:d}th matrix (cost = {2:.4f})'
    ).format(
        best_channel_index + 1, best_matrix_index + 1, best_cost
    ))

    permuted_flags_by_matrix[best_matrix_index][best_channel_index] = False
    depermuted_matrix_indices = numpy.array(
        depermuted_matrix_indices, dtype=int
    )
    depermuted_channel_indices = numpy.array(
        depermuted_channel_indices, dtype=int
    )

    return {
        PREDICTORS_KEY: predictor_matrices,
        PERMUTED_FLAGS_KEY: permuted_flags_by_matrix,
        DEPERMUTED_MATRICES_KEY: depermuted_matrix_indices,
        DEPERMUTED_CHANNELS_KEY: depermuted_channel_indices,
        DEPERMUTED_COSTS_KEY: depermuted_cost_matrix,
    }


def _run_forward_test(
        predictor_matrices, target_matrix, prediction_function, cost_function,
        num_bootstrap_reps, bootstrap_by_shuffling_many_times):
    """Runs forward version of permutation test (both single- and multi-pass).

    B = number of bootstrap replicates
    N = number of predictor variables available to permute

    :param predictor_matrices: See doc for `_run_forward_test_one_step`.
    :param target_matrix: Same.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.
    :param bootstrap_by_shuffling_many_times: Same.
    :return: result_dict: Dictionary with the following keys.
    result_dict['orig_cost_estimates']: length-B numpy array with estimates of
        original cost (before permutation).
    result_dict['best_predictor_names']: length-N list with best predictor at
        each step.
    result_dict['best_cost_matrix']: N-by-B numpy array of costs after
        permutation at each step.
    result_dict['step1_predictor_names']: length-N list with predictors in order
        that they were permuted in step 1.
    result_dict['step1_cost_matrix']: N-by-B numpy array of costs after
        permutation in step 1.
    result_dict['is_backwards_test']: Boolean flag (always False for this
        method).
    """

    # Find original cost (before permutation).
    print('Finding original cost (before permutation)...')

    if bootstrap_by_shuffling_many_times:
        orig_cost_estimate = cost_function(
            target_matrix,
            prediction_function(predictor_matrices)
        )
        orig_cost_estimates = numpy.full(num_bootstrap_reps, orig_cost_estimate)
    else:
        orig_cost_estimates = _bootstrap_cost(
            target_matrix=target_matrix,
            probability_matrix=prediction_function(predictor_matrices),
            cost_function=cost_function, num_replicates=num_bootstrap_reps
        )

    # Do actual stuff.
    permuted_flags_by_matrix = [
        numpy.full(p.shape[-1], False, dtype=bool) for p in predictor_matrices
    ]

    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    step1_predictor_names = None
    step1_cost_matrix = None
    step_num = 0

    while True:
        print(MINOR_SEPARATOR_STRING)
        step_num += 1

        this_result_dict = _run_forward_test_one_step(
            predictor_matrices=predictor_matrices,
            target_matrix=target_matrix,
            prediction_function=prediction_function,
            cost_function=cost_function,
            permuted_flags_by_matrix=permuted_flags_by_matrix,
            num_bootstrap_reps=num_bootstrap_reps,
            bootstrap_by_shuffling_many_times=bootstrap_by_shuffling_many_times
        )

        if this_result_dict is None:
            break

        predictor_matrices = this_result_dict[PREDICTORS_KEY]
        permuted_flags_by_matrix = this_result_dict[PERMUTED_FLAGS_KEY]

        these_predictor_names = [
            PREDICTOR_NAMES_BY_MATRIX[i][k] for i, k in zip(
                this_result_dict[PERMUTED_MATRICES_KEY],
                this_result_dict[PERMUTED_CHANNELS_KEY]
            )
        ]

        this_best_index = numpy.argmin(
            numpy.mean(this_result_dict[PERMUTED_COSTS_KEY], axis=1)
        )
        best_predictor_names.append(these_predictor_names[this_best_index])
        best_cost_matrix = numpy.concatenate((
            best_cost_matrix,
            this_result_dict[PERMUTED_COSTS_KEY][[this_best_index], :]
        ), axis=0)

        print((
            'Best predictor at {0:d}th step = {1:s} (cost = {2:.4f})'
        ).format(
            step_num,
            best_predictor_names[-1],
            numpy.mean(best_cost_matrix[-1, :])
        ))

        if step_num != 1:
            continue

        step1_predictor_names = copy.deepcopy(these_predictor_names)
        step1_cost_matrix = this_result_dict[PERMUTED_COSTS_KEY] + 0.

    return {
        ORIGINAL_COST_KEY: orig_cost_estimates,
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COSTS_KEY: best_cost_matrix,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COSTS_KEY: step1_cost_matrix,
        BACKWARDS_FLAG_KEY: False
    }


def _run_backwards_test(
        predictor_matrices, target_matrix, prediction_function, cost_function,
        num_bootstrap_reps, bootstrap_by_shuffling_many_times):
    """Runs backwards version of permutation test (both single- and multi-pass).

    B = number of bootstrap replicates
    N = number of predictor variables available to permute

    :param predictor_matrices: See doc for `_run_forward_test`.
    :param target_matrix: Same.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.
    :param bootstrap_by_shuffling_many_times: Same.
    :return: result_dict: Dictionary with the following keys.
    result_dict['orig_cost_estimates']: length-B numpy array with estimates of
        original cost (before *de*permutation).
    result_dict['best_predictor_names']: length-N list with best predictor at
        each step.
    result_dict['best_cost_matrix']: N-by-B numpy array of costs after
        *de*permutation at each step.
    result_dict['step1_predictor_names']: length-N list with predictors in order
        that they were *de*permuted in step 1.
    result_dict['step1_cost_matrix']: N-by-B numpy array of costs after
        *de*permutation in step 1.
    result_dict['is_backwards_test']: Boolean flag (always True for this
        method).
    """

    # Find original cost (before *de*permutation).
    clean_predictor_matrices = copy.deepcopy(predictor_matrices)

    if bootstrap_by_shuffling_many_times:
        orig_cost_estimates = numpy.full(num_bootstrap_reps, numpy.nan)

        for j in range(num_bootstrap_reps):
            print((
                'Finding original cost (before *de*permutation) with {0:d}th '
                'of {1:d} random seeds...'
            ).format(
                j + 1, num_bootstrap_reps
            ))

            for i in range(len(predictor_matrices)):
                for k in range(predictor_matrices[i].shape[-1]):
                    predictor_matrices = _permute_values(
                        predictor_matrices=predictor_matrices,
                        matrix_index=i, channel_index=k
                    )[0]

            orig_cost_estimates[j] = cost_function(
                target_matrix, prediction_function(predictor_matrices)
            )
    else:
        print('Finding original cost (before *de*permutation)...')

        for i in range(len(predictor_matrices)):
            for k in range(predictor_matrices[i].shape[-1]):
                predictor_matrices = _permute_values(
                    predictor_matrices=predictor_matrices,
                    matrix_index=i, channel_index=k
                )[0]

        orig_cost_estimates = _bootstrap_cost(
            target_matrix=target_matrix,
            probability_matrix=prediction_function(predictor_matrices),
            cost_function=cost_function, num_replicates=num_bootstrap_reps
        )

    # Do actual stuff.
    permuted_flags_by_matrix = [
        numpy.full(p.shape[-1], True, dtype=bool) for p in predictor_matrices
    ]

    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    step1_predictor_names = None
    step1_cost_matrix = None
    step_num = 0

    while True:
        print(MINOR_SEPARATOR_STRING)
        step_num += 1

        this_result_dict = _run_backwards_test_one_step(
            predictor_matrices=predictor_matrices,
            clean_predictor_matrices=clean_predictor_matrices,
            target_matrix=target_matrix,
            prediction_function=prediction_function,
            cost_function=cost_function,
            permuted_flags_by_matrix=permuted_flags_by_matrix,
            num_bootstrap_reps=num_bootstrap_reps,
            bootstrap_by_shuffling_many_times=bootstrap_by_shuffling_many_times
        )

        if this_result_dict is None:
            break

        predictor_matrices = this_result_dict[PREDICTORS_KEY]
        permuted_flags_by_matrix = this_result_dict[PERMUTED_FLAGS_KEY]

        these_predictor_names = [
            PREDICTOR_NAMES_BY_MATRIX[i][k] for i, k in zip(
                this_result_dict[DEPERMUTED_MATRICES_KEY],
                this_result_dict[DEPERMUTED_CHANNELS_KEY]
            )
        ]

        this_best_index = numpy.argmax(
            numpy.mean(this_result_dict[DEPERMUTED_COSTS_KEY], axis=1)
        )
        best_predictor_names.append(these_predictor_names[this_best_index])
        best_cost_matrix = numpy.concatenate((
            best_cost_matrix,
            this_result_dict[DEPERMUTED_COSTS_KEY][[this_best_index], :]
        ), axis=0)

        print((
            'Best predictor at {0:d}th step = {1:s} (cost = {2:.4f})'
        ).format(
            step_num,
            best_predictor_names[-1],
            numpy.mean(best_cost_matrix[-1, :])
        ))

        if step_num != 1:
            continue

        step1_predictor_names = copy.deepcopy(these_predictor_names)
        step1_cost_matrix = this_result_dict[DEPERMUTED_COSTS_KEY] + 0.

    return {
        ORIGINAL_COST_KEY: orig_cost_estimates,
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COSTS_KEY: best_cost_matrix,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COSTS_KEY: step1_cost_matrix,
        BACKWARDS_FLAG_KEY: True
    }


def _mkdir_recursive_if_necessary(directory_name=None, file_name=None):
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


def _run(model_dir_name, image_patch_dir_name, prediction_file_name,
         species_name_for_positive_class, do_backwards_test, num_bootstrap_reps,
         bootstrap_by_shuffling_many_times, output_file_name):
    """Runs permutation test on trained model suite (one for each CV fold).

    This is effectively the main method.

    :param model_dir_name: See documentation at top of file.
    :param image_patch_dir_name: Same.
    :param prediction_file_name: Same.
    :param species_name_for_positive_class: Same.
    :param do_backwards_test: Same.
    :param num_bootstrap_reps: Same.
    :param bootstrap_by_shuffling_many_times: Same.
    :param output_file_name: Same.
    """

    num_bootstrap_reps = max([num_bootstrap_reps, 1])

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    probability_matrix, target_matrix, fold_indices = (
        _read_predictions_and_targets(prediction_file_name)
    )

    num_folds = numpy.max(fold_indices) + 1
    num_classes = probability_matrix.shape[1]

    model_file_names = [
        '{0:s}/fold{1:d}/model.keras'.format(model_dir_name, i)
        for i in range(num_folds)
    ]
    model_objects = [None] * num_folds

    for i in range(num_folds):
        print('Reading model from: "{0:s}"...'.format(model_file_names[i]))
        model_objects[i] = _read_model(
            hdf5_file_name=model_file_names[i], num_classes=num_classes
        )

    image_patch_file_pattern = '{0:s}/image_patches[0-9][0-9].nc'.format(
        image_patch_dir_name
    )

    image_patch_file_names = glob.glob(image_patch_file_pattern)
    image_patch_file_names.sort()
    patch_id_strings = []

    for this_file_name in image_patch_file_names:
        print('Reading patch IDs from: "{0:s}"...'.format(this_file_name))
        this_patch_dict = _read_image_patches(this_file_name)
        patch_id_strings += this_patch_dict[PATCH_IDS_KEY]

    num_examples = len(patch_id_strings)
    assert num_examples == probability_matrix.shape[0]
    last_index = 0

    multispectral_radiance_matrix_w_m02 = None
    panchromatic_radiance_matrix_w_m02 = None
    elevation_matrix_m_agl = None

    for this_file_name in image_patch_file_names:
        print('Reading all data from: "{0:s}"...'.format(this_file_name))
        this_patch_dict = _read_image_patches(this_file_name)

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

    multispectral_radiance_matrix_norm = numpy.full(
        multispectral_radiance_matrix_w_m02.shape, numpy.nan
    )
    panchromatic_radiance_matrix_norm = numpy.full(
        panchromatic_radiance_matrix_w_m02.shape, numpy.nan
    )
    elevation_matrix_norm = numpy.full(
        elevation_matrix_m_agl.shape, numpy.nan
    )

    for k in range(num_folds):
        training_indices = numpy.where(fold_indices != k)[0]
        validation_indices = numpy.where(fold_indices == k)[0]

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

        multispectral_radiance_matrix_norm[validation_indices, ...] = (
            (multispectral_radiance_matrix_w_m02[validation_indices, ...] -
             multispectral_radiance_mean_w_m02) /
            multispectral_radiance_stdev_w_m02
        )

        panchromatic_radiance_matrix_norm[validation_indices, ...] = (
            (panchromatic_radiance_matrix_w_m02[validation_indices, ...] -
             panchromatic_radiance_mean_w_m02) /
            panchromatic_radiance_stdev_w_m02
        )

        elevation_matrix_norm[validation_indices, ...] = (
            (elevation_matrix_m_agl[validation_indices, ...] -
             elevation_mean_m_asl) /
            elevation_stdev_m_asl
        )

    predictor_matrices_norm = [
        multispectral_radiance_matrix_norm.astype('float32'),
        numpy.expand_dims(
            panchromatic_radiance_matrix_norm, axis=-1
        ).astype('float32'),
        numpy.expand_dims(elevation_matrix_norm, axis=-1).astype('float32')
    ]

    del multispectral_radiance_matrix_w_m02
    del panchromatic_radiance_matrix_w_m02
    del elevation_matrix_m_agl

    print('NaN fractions:')

    for i in range(len(predictor_matrices_norm)):
        print(numpy.mean(numpy.isnan(predictor_matrices_norm[i])))

        predictor_matrices_norm[i][
            numpy.isnan(predictor_matrices_norm[i])
        ] = -10.

    prediction_function = make_prediction_function(
        model_objects=model_objects, fold_indices=fold_indices
    )

    if species_name_for_positive_class == '':
        cost_function = _make_gerrity_cost_function()
    else:
        cost_function = _make_auc_cost_function(
            num_classes=num_classes,
            species_name_for_positive_class=species_name_for_positive_class
        )

    this_prob_matrix = prediction_function(predictor_matrices_norm)
    print((
        'Max absolute difference between original and new predicted probs = '
        '{0:.2g}'
    ).format(
        numpy.max(numpy.absolute(this_prob_matrix - probability_matrix))
    ))

    print(SEPARATOR_STRING)

    if do_backwards_test:
        result_dict = _run_backwards_test(
            predictor_matrices=predictor_matrices_norm,
            target_matrix=target_matrix,
            prediction_function=prediction_function,
            cost_function=cost_function,
            num_bootstrap_reps=num_bootstrap_reps,
            bootstrap_by_shuffling_many_times=bootstrap_by_shuffling_many_times
        )
    else:
        result_dict = _run_forward_test(
            predictor_matrices=predictor_matrices_norm,
            target_matrix=target_matrix,
            prediction_function=prediction_function,
            cost_function=cost_function,
            num_bootstrap_reps=num_bootstrap_reps,
            bootstrap_by_shuffling_many_times=bootstrap_by_shuffling_many_times
        )

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    _mkdir_recursive_if_necessary(file_name=output_file_name)
    pickle_file_handle = open(output_file_name, 'wb')
    pickle.dump(result_dict, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_dir_name=getattr(INPUT_ARG_OBJECT, MODEL_DIR_ARG_NAME),
        image_patch_dir_name=getattr(INPUT_ARG_OBJECT, PATCH_DIR_ARG_NAME),
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        species_name_for_positive_class=getattr(
            INPUT_ARG_OBJECT, POSITIVE_CLASS_SPECIES_ARG_NAME
        ),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        bootstrap_by_shuffling_many_times=getattr(
            INPUT_ARG_OBJECT, BOOTSTRAP_BY_SHUFFLING_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
