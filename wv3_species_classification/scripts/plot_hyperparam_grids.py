"""Plots hyperparameter grids for main experiment (shown in paper)."""

import os
import copy
import errno
import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from scipy.stats import rankdata
from keras.metrics import top_k_categorical_accuracy

MIN_PROB_FOR_XENTROPY = numpy.finfo(float).eps
MAX_PROB_FOR_XENTROPY = 1. - numpy.finfo(float).eps

UNIQUE_SPECIES_NAMES_6CLASSES = [
    'POTR', 'Salix', 'PIEN', 'BEGL', 'PIFL', 'ABLA'
]
UNIQUE_SPECIES_NAMES_4CLASSES = ['PIEN', 'PIFL', 'ABLA', 'Other']
UNIQUE_SPECIES_NAMES_2CLASSES = ['PIFL', 'Other']

DENSE_LAYER_COUNTS_AXIS1_6CLASSES = numpy.array([1, 2, 3, 4], dtype=int)
DROPOUT_RATES_AXIS2_6CLASSES = numpy.array([0.575, 0.575, 0.650, 0.650])
L2_WEIGHTS_AXIS2_6CLASSES = numpy.array([10 ** -6.5, 10 ** -6, 10 ** -6.5, 10 ** -6])
USE_GERRITY_LOSS_FLAGS_AXIS3_6CLASSES = numpy.array([0, 0, 1, 1, 1, 1], dtype=int)
USE_CLASS_WEIGHTS_FLAGS_AXIS3_6CLASSES = numpy.array([0, 1, 0, 1, 0, 1], dtype=int)
PUT_PIFL_FIRST_FLAGS_AXIS3_6CLASSES = numpy.array([0, 0, 0, 0, 1, 1], dtype=int)

DENSE_LAYER_COUNTS_AXIS1_4CLASSES = numpy.array([1, 2, 3, 4], dtype=int)
DROPOUT_RATES_AXIS2_4CLASSES = numpy.array([0.650])
L2_WEIGHTS_AXIS2_4CLASSES = numpy.array([10 ** -6.5])
USE_GERRITY_LOSS_FLAGS_AXIS3_4CLASSES = numpy.array([0, 0, 1, 1, 1, 1], dtype=int)
USE_CLASS_WEIGHTS_FLAGS_AXIS3_4CLASSES = numpy.array([0, 1, 0, 1, 0, 1], dtype=int)
PUT_PIFL_FIRST_FLAGS_AXIS3_4CLASSES = numpy.array([0, 0, 0, 0, 1, 1], dtype=int)

DENSE_LAYER_COUNTS_AXIS1_2CLASSES = numpy.array([1, 2, 3, 4], dtype=int)
DROPOUT_RATES_AXIS2_2CLASSES = numpy.array([0.650])
L2_WEIGHTS_AXIS2_2CLASSES = numpy.array([10 ** -6.5])
USE_GERRITY_LOSS_FLAGS_AXIS3_2CLASSES = numpy.array([0, 0, 1, 1, 1, 1], dtype=int)
USE_CLASS_WEIGHTS_FLAGS_AXIS3_2CLASSES = numpy.array([0, 1, 0, 1, 0, 1], dtype=int)
PUT_PIFL_FIRST_FLAGS_AXIS3_2CLASSES = numpy.array([0, 0, 0, 0, 1, 1], dtype=int)

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.375
SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.25
MARKER_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_INDICES_6CLASSES = numpy.array([0, 2, 3], dtype=int)
SELECTED_MARKER_INDICES_4CLASSES = numpy.array([0, 0, 2], dtype=int)
SELECTED_MARKER_INDICES_2CLASSES = numpy.array([0, 0, 4], dtype=int)

FONT_SIZE = 26
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

# DEFAULT_CONVERT_EXE_NAME = '/usr/bin/convert'
# DEFAULT_MONTAGE_EXE_NAME = '/usr/bin/montage'
DEFAULT_CONVERT_EXE_NAME = 'convert'
DEFAULT_MONTAGE_EXE_NAME = 'montage'
IMAGEMAGICK_ERROR_STRING = (
    '\nUnix command failed (log messages shown above should explain why).'
)

INPUT_DIR_ARG_NAME = 'input_experiment_dir_name'
NUM_CLASSES_ARG_NAME = 'num_classes'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing all models for the experiment.'
)
NUM_CLASSES_HELP_STRING = 'Number of classes.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CLASSES_ARG_NAME, type=int, required=True,
    help=NUM_CLASSES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_predictions_and_targets(pickle_file_name):
    """Reads predictions and targets from Pickle file.

    K = number of classes
    E = number of examples

    :param pickle_file_name: Path to input file.
    :return: probability_matrix: E-by-K numpy array of predicted probabilities.
    :return: target_matrix: E-by-K numpy array of true labels (all 0 or 1).
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    probability_matrix = pickle.load(pickle_file_handle)
    target_matrix = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return probability_matrix, target_matrix


def _get_contingency_table(predicted_labels, observed_labels, num_classes):
    """Creates contingency table.

    E = number of examples
    K = number of classes

    :param observed_labels: length-E numpy array of observed labels (classes in
        0...[K - 1]).
    :param predicted_labels: length-E numpy array of predicted labels (classes
        in 0...[K - 1]).
    :return: contingency_matrix: See doc for `_check_contingency_table`.
    """

    contingency_matrix = numpy.full(
        (num_classes, num_classes), -1, dtype=int
    )

    for i in range(num_classes):
        for j in range(num_classes):
            contingency_matrix[i, j] = numpy.sum(numpy.logical_and(
                predicted_labels == i, observed_labels == j
            ))

    return contingency_matrix


def _non_zero(input_value):
    """Makes input non-zero.

    :param input_value: Input.
    :return: output_value: Closest number to input that is outside of
        [-epsilon, epsilon], where epsilon is the machine limit for
        floating-point numbers.
    """

    epsilon = numpy.finfo(float).eps
    if input_value >= 0:
        return max([input_value, epsilon])

    return min([input_value, -epsilon])


def _num_examples_with_observed_class(contingency_matrix, class_index):
    """Returns number of examples where a given class is observed.

    This method returns number of examples where [k]th class is observed, k
    being `class_index`.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :param class_index: See above.
    :return: num_examples: See above.
    """

    return numpy.sum(contingency_matrix[:, class_index])


def _num_examples_with_predicted_class(contingency_matrix, class_index):
    """Returns number of examples where a given class is predicted.

    This method returns number of examples where [k]th class is predicted, k
    being `class_index`.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :param class_index: See above.
    :return: num_examples: See above.
    """

    return numpy.sum(contingency_matrix[class_index, :])


def _get_a_for_gerrity_score(contingency_matrix):
    """Returns vector a for Gerrity score.

    The equation for a is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: a_vector: See above.
    """

    num_classes = contingency_matrix.shape[0]
    num_examples = numpy.sum(contingency_matrix)

    num_examples_by_class = numpy.array([
        _num_examples_with_observed_class(contingency_matrix, i)
        for i in range(num_classes)
    ])
    cumulative_freq_by_class = numpy.cumsum(
        num_examples_by_class.astype(float) / num_examples
    )

    return (1. - cumulative_freq_by_class) / cumulative_freq_by_class


def _get_s_for_gerrity_score(contingency_matrix):
    """Returns matrix S for Gerrity score.

    The equation for S is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: s_matrix: See above.
    """

    a_vector = _get_a_for_gerrity_score(contingency_matrix)
    a_vector_reciprocal = 1. / a_vector

    num_classes = contingency_matrix.shape[0]
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


def _get_gerrity_score(contingency_matrix, put_pifl_first):
    """Computes Gerrity score.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :param put_pifl_first: Boolean flag.
    :return: gerrity_score: Gerrity score (range -1...1).
    """

    if put_pifl_first:
        if contingency_matrix.shape[1] == 6:
            pifl_index = UNIQUE_SPECIES_NAMES_6CLASSES.index('PIFL')
        elif contingency_matrix.shape[1] == 4:
            pifl_index = UNIQUE_SPECIES_NAMES_4CLASSES.index('PIFL')
        else:
            pifl_index = UNIQUE_SPECIES_NAMES_2CLASSES.index('PIFL')

        print('ORIGINAL CONFUSION MATRIX:\n{0:s}'.format(
            str(contingency_matrix)
        ))

        pifl_column = contingency_matrix[:, pifl_index]
        contingency_matrix = numpy.delete(
            contingency_matrix, pifl_index, axis=1
        )
        contingency_matrix = numpy.insert(
            contingency_matrix, 0, pifl_column, axis=1
        )

        pifl_row = contingency_matrix[pifl_index, :]
        contingency_matrix = numpy.delete(
            contingency_matrix, pifl_index, axis=0
        )
        contingency_matrix = numpy.insert(
            contingency_matrix, 0, pifl_row, axis=0
        )

        print('NEW CONFUSION MATRIX:\n{0:s}'.format(
            str(contingency_matrix)
        ))

    s_matrix = _get_s_for_gerrity_score(contingency_matrix)
    num_examples = numpy.sum(contingency_matrix)

    return numpy.sum(contingency_matrix * s_matrix) / num_examples


def _get_peirce_score(contingency_matrix):
    """Computes Peirce score.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: peirce_score: Peirce score (range -1...1).
    """

    first_numerator_term = 0
    second_numerator_term = 0
    denominator_term = 0
    num_classes = contingency_matrix.shape[0]

    for i in range(num_classes):
        first_numerator_term += contingency_matrix[i, i]

        second_numerator_term += (
            _num_examples_with_predicted_class(contingency_matrix, i) *
            _num_examples_with_observed_class(contingency_matrix, i)
        )

        denominator_term += (
            _num_examples_with_observed_class(contingency_matrix, i) ** 2
        )

    num_examples = numpy.sum(contingency_matrix)

    first_numerator_term = float(first_numerator_term) / num_examples
    second_numerator_term = float(second_numerator_term) / num_examples ** 2
    denominator = _non_zero(1. - float(denominator_term) / num_examples ** 2)

    return (first_numerator_term - second_numerator_term) / denominator


def _get_heidke_score(contingency_matrix):
    """Computes Heidke score.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: heidke_score: Heidke score (range -inf...1).
    """

    first_numerator_term = 0
    second_numerator_term = 0
    num_classes = contingency_matrix.shape[0]

    for i in range(num_classes):
        first_numerator_term += contingency_matrix[i, i]

        second_numerator_term += (
            _num_examples_with_predicted_class(contingency_matrix, i) *
            _num_examples_with_observed_class(contingency_matrix, i)
        )

    num_examples = numpy.sum(contingency_matrix)

    first_numerator_term = float(first_numerator_term) / num_examples
    second_numerator_term = (float(second_numerator_term) / num_examples**2)
    denominator = _non_zero(1. - second_numerator_term)

    return (first_numerator_term - second_numerator_term) / denominator


def _get_cross_entropy(probability_matrix, target_matrix):
    """Computes cross-entropy.

    E = number of examples
    K = number of classes

    :param probability_matrix: E-by-K numpy array of probabilities.
    :param target_matrix: E-by-K numpy array of true labels (all 0 or 1).
    :return: cross_entropy: Cross-entropy.
    """

    this_prob_matrix = copy.deepcopy(probability_matrix)
    this_prob_matrix = numpy.maximum(this_prob_matrix, MIN_PROB_FOR_XENTROPY)
    this_prob_matrix = numpy.minimum(this_prob_matrix, MAX_PROB_FOR_XENTROPY)

    cross_entropy = 0.
    num_classes = probability_matrix.shape[1]

    for k in range(num_classes):
        cross_entropy -= numpy.sum(
            target_matrix[:, k] * numpy.log2(this_prob_matrix[:, k])
        )

    cross_entropy = cross_entropy / probability_matrix.size
    return cross_entropy


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        score_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(x_tick_values, x_tick_labels, rotation=90)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )

    colour_bar_object = _plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE, fraction_of_axis_length=0.8
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


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


def _plot_colour_bar(
        axes_object_or_matrix, data_matrix, colour_map_object,
        colour_norm_object, orientation_string,
        padding=None, extend_min=True, extend_max=True,
        fraction_of_axis_length=1., font_size=FONT_SIZE):
    """Plots colour bar.

    :param axes_object_or_matrix: Either one axis handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`) or a numpy array thereof.
    :param data_matrix: numpy array of values to which the colour map applies.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm` or
        similar).
    :param colour_norm_object: Colour normalization (maps from data space to
        colour-bar space, which goes from 0...1).  This should be an instance of
        `matplotlib.colors.Normalize`.
    :param orientation_string: Orientation ("vertical" or "horizontal").
    :param padding: Padding between colour bar and main plot (in range 0...1).
        To use the default (there are different defaults for vertical and horiz
        colour bars), leave this alone.
    :param extend_min: Boolean flag.  If True, values below the minimum
        specified by `colour_norm_object` are possible, so the colour bar will
        be plotted with an arrow at the bottom.
    :param extend_max: Boolean flag.  If True, values above the max specified by
        `colour_norm_object` are possible, so the colour bar will be plotted
        with an arrow at the top.
    :param fraction_of_axis_length: The colour bar will take up this fraction of
        the axis length (x-axis if orientation_string = "horizontal", y-axis if
        orientation_string = "vertical").
    :param font_size: Font size for tick marks on colour bar.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).
    """

    extend_min = bool(extend_min)
    extend_max = bool(extend_max)
    fraction_of_axis_length = max([fraction_of_axis_length, 1e-6])

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(data_matrix)

    if extend_min and extend_max:
        extend_arg = 'both'
    elif extend_min:
        extend_arg = 'min'
    elif extend_max:
        extend_arg = 'max'
    else:
        extend_arg = 'neither'

    if padding is None:
        if orientation_string == 'horizontal':
            padding = 0.075
        else:
            padding = 0.05

    padding = max([padding, 0.])

    if isinstance(axes_object_or_matrix, numpy.ndarray):
        axes_arg = axes_object_or_matrix.ravel().tolist()
    else:
        axes_arg = axes_object_or_matrix

    colour_bar_object = pyplot.colorbar(
        ax=axes_arg, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_arg,
        shrink=fraction_of_axis_length
    )

    colour_bar_object.ax.tick_params(labelsize=font_size)

    if orientation_string == 'horizontal':
        colour_bar_object.ax.set_xticklabels(
            colour_bar_object.ax.get_xticklabels(), rotation=90
        )

    return colour_bar_object


def _concatenate_images(
        input_file_names, output_file_name, num_panel_rows, num_panel_columns,
        border_width_pixels=50, montage_exe_name=DEFAULT_MONTAGE_EXE_NAME,
        extra_args_string=None):
    """Concatenates many images into one paneled image.

    :param input_file_names: 1-D list of paths to input files (may be in any
        format handled by ImageMagick).
    :param output_file_name: Path to output file.
    :param num_panel_rows: Number of rows in paneled image.
    :param num_panel_columns: Number of columns in paneled image.
    :param border_width_pixels: Border width (whitespace) around each pixel.
    :param montage_exe_name: Path to executable file for ImageMagick's `montage`
        function.  If you installed ImageMagick with root access, this should be
        the default.  Regardless, the pathless file name should be just
        "montage".
    :param extra_args_string: String with extra args for ImageMagick's `montage`
        function.  This string will be inserted into the command after
        "montage -mode concatenate".  An example is "-gravity south", in which
        case the beginning of the command is
        "montage -mode concatenate -gravity south".
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    if extra_args_string is None:
        extra_args_string = ''

    _mkdir_recursive_if_necessary(file_name=output_file_name)
    num_panel_rows = int(numpy.round(num_panel_rows))
    num_panel_columns = int(numpy.round(num_panel_columns))
    border_width_pixels = int(numpy.round(border_width_pixels))
    border_width_pixels = max([border_width_pixels, 0])

    num_panels = num_panel_rows * num_panel_columns
    assert num_panels >= len(input_file_names)

    command_string = '"{0:s}" -mode concatenate {1:s} -tile {2:d}x{3:d}'.format(
        montage_exe_name, extra_args_string, num_panel_columns, num_panel_rows
    )

    for this_file_name in input_file_names:
        command_string += ' "{0:s}"'.format(this_file_name)

    command_string += ' -trim -bordercolor White -border {0:d} "{1:s}"'.format(
        border_width_pixels, output_file_name
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return
    raise ValueError(IMAGEMAGICK_ERROR_STRING)


def _resize_image(input_file_name, output_file_name, output_size_pixels,
                  convert_exe_name=DEFAULT_CONVERT_EXE_NAME):
    """Resizes image.

    :param input_file_name: Path to input file (may be in any format handled by
        ImageMagick).
    :param output_file_name: Path to output file.
    :param output_size_pixels: Output size.
    :param convert_exe_name: See doc for `trim_whitespace`.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    _mkdir_recursive_if_necessary(file_name=output_file_name)
    output_size_pixels = int(numpy.round(output_size_pixels))
    output_size_pixels = max([output_size_pixels, 1])

    command_string = '"{0:s}" "{1:s}" -resize {2:d}@ "{3:s}"'.format(
        convert_exe_name, input_file_name, output_size_pixels, output_file_name
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(IMAGEMAGICK_ERROR_STRING)


def _run(experiment_dir_name, num_classes, output_dir_name):
    """Plots hyperparameter grids for main experiment (shown in paper).

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param num_classes: Same.
    :param output_dir_name: Same.
    """

    # TODO(thunderhoser): Implement new Gerrity score (PIFL first).

    assert num_classes in [2, 4, 6]

    if num_classes == 6:
        dense_layer_counts_axis1 = DENSE_LAYER_COUNTS_AXIS1_6CLASSES
        dropout_rates_axis2 = DROPOUT_RATES_AXIS2_6CLASSES
        l2_weights_axis2 = L2_WEIGHTS_AXIS2_6CLASSES
        use_gerrity_loss_flags_axis3 = USE_GERRITY_LOSS_FLAGS_AXIS3_6CLASSES
        use_class_weights_flags_axis3 = USE_CLASS_WEIGHTS_FLAGS_AXIS3_6CLASSES
        put_pifl_first_flags_axis3 = PUT_PIFL_FIRST_FLAGS_AXIS3_6CLASSES
        selected_marker_indices = SELECTED_MARKER_INDICES_6CLASSES
    elif num_classes == 4:
        dense_layer_counts_axis1 = DENSE_LAYER_COUNTS_AXIS1_4CLASSES
        dropout_rates_axis2 = DROPOUT_RATES_AXIS2_4CLASSES
        l2_weights_axis2 = L2_WEIGHTS_AXIS2_4CLASSES
        use_gerrity_loss_flags_axis3 = USE_GERRITY_LOSS_FLAGS_AXIS3_4CLASSES
        use_class_weights_flags_axis3 = USE_CLASS_WEIGHTS_FLAGS_AXIS3_4CLASSES
        put_pifl_first_flags_axis3 = PUT_PIFL_FIRST_FLAGS_AXIS3_4CLASSES
        selected_marker_indices = SELECTED_MARKER_INDICES_4CLASSES
    else:
        dense_layer_counts_axis1 = DENSE_LAYER_COUNTS_AXIS1_2CLASSES
        dropout_rates_axis2 = DROPOUT_RATES_AXIS2_2CLASSES
        l2_weights_axis2 = L2_WEIGHTS_AXIS2_2CLASSES
        use_gerrity_loss_flags_axis3 = USE_GERRITY_LOSS_FLAGS_AXIS3_2CLASSES
        use_class_weights_flags_axis3 = USE_CLASS_WEIGHTS_FLAGS_AXIS3_2CLASSES
        put_pifl_first_flags_axis3 = PUT_PIFL_FIRST_FLAGS_AXIS3_2CLASSES
        selected_marker_indices = SELECTED_MARKER_INDICES_2CLASSES

    axis1_length = len(dense_layer_counts_axis1)
    axis2_length = len(l2_weights_axis2)
    axis3_length = len(use_gerrity_loss_flags_axis3)

    dimensions = (axis1_length, axis2_length, axis3_length)
    top1_accuracy_matrix = numpy.full(dimensions, numpy.nan)
    top2_accuracy_matrix = numpy.full(dimensions, numpy.nan)
    top3_accuracy_matrix = numpy.full(dimensions, numpy.nan)
    gerrity_score_matrix = numpy.full(dimensions, numpy.nan)
    pifl_gerrity_score_matrix = numpy.full(dimensions, numpy.nan)
    peirce_score_matrix = numpy.full(dimensions, numpy.nan)
    heidke_score_matrix = numpy.full(dimensions, numpy.nan)
    cross_entropy_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(axis1_length):
        for j in range(axis2_length):
            for k in range(axis3_length):
                this_prediction_file_name = (
                    '{0:s}/use-gerrity-loss={1:d}_use-class-weights={2:d}_'
                    'put-pifl-first={3:d}_'
                    'num-dense-layers={4:012d}_dropout-rate={5:.10f}_'
                    'l2-weight={6:.10f}/oob_predictions.p'
                ).format(
                    experiment_dir_name,
                    use_gerrity_loss_flags_axis3[k],
                    use_class_weights_flags_axis3[k],
                    put_pifl_first_flags_axis3[k],
                    dense_layer_counts_axis1[i],
                    dropout_rates_axis2[j],
                    l2_weights_axis2[j]
                )

                print(this_prediction_file_name)

                if not os.path.isfile(this_prediction_file_name):
                    continue

                print('Reading data from: "{0:s}"...'.format(
                    this_prediction_file_name
                ))
                this_probability_matrix, this_target_matrix = (
                    _read_predictions_and_targets(this_prediction_file_name)
                )

                num_classes = this_target_matrix.shape[1]

                top1_accuracy_matrix[i, j, k] = numpy.mean(
                    top_k_categorical_accuracy(
                        y_true=this_target_matrix.astype(numpy.float32),
                        y_pred=this_probability_matrix.astype(numpy.float32),
                        k=1
                    )
                )

                if num_classes > 2:
                    top2_accuracy_matrix[i, j, k] = numpy.mean(
                        top_k_categorical_accuracy(
                            y_true=this_target_matrix.astype(numpy.float32),
                            y_pred=this_probability_matrix.astype(numpy.float32),
                            k=2
                        )
                    )

                    top3_accuracy_matrix[i, j, k] = numpy.mean(
                        top_k_categorical_accuracy(
                            y_true=this_target_matrix.astype(numpy.float32),
                            y_pred=this_probability_matrix.astype(numpy.float32),
                            k=3
                        )
                    )

                cross_entropy_matrix[i, j, k] = _get_cross_entropy(
                    probability_matrix=this_probability_matrix,
                    target_matrix=this_target_matrix
                )

                this_confusion_matrix = _get_contingency_table(
                    predicted_labels=
                    numpy.argmax(this_probability_matrix, axis=1),
                    observed_labels=numpy.argmax(this_target_matrix, axis=1),
                    num_classes=num_classes
                )

                gerrity_score_matrix[i, j, k] = _get_gerrity_score(
                    this_confusion_matrix, put_pifl_first=False
                )
                pifl_gerrity_score_matrix[i, j, k] = _get_gerrity_score(
                    this_confusion_matrix, put_pifl_first=True
                )
                peirce_score_matrix[i, j, k] = _get_peirce_score(
                    this_confusion_matrix
                )
                heidke_score_matrix[i, j, k] = _get_heidke_score(
                    this_confusion_matrix
                )

    i = selected_marker_indices[0]
    j = selected_marker_indices[1]
    k = selected_marker_indices[2]

    top1_accuracy_rank_matrix = numpy.reshape(
        rankdata(
            a=-1 * numpy.nan_to_num(top1_accuracy_matrix, nan=-numpy.inf),
            method='average'
        ),
        top1_accuracy_matrix.shape
    )
    top2_accuracy_rank_matrix = numpy.reshape(
        rankdata(
            a=-1 * numpy.nan_to_num(top2_accuracy_matrix, nan=-numpy.inf),
            method='average'
        ),
        top2_accuracy_matrix.shape
    )
    top3_accuracy_rank_matrix = numpy.reshape(
        rankdata(
            a=-1 * numpy.nan_to_num(top3_accuracy_matrix, nan=-numpy.inf),
            method='average'
        ),
        top3_accuracy_matrix.shape
    )
    gerrity_rank_matrix = numpy.reshape(
        rankdata(
            a=-1 * numpy.nan_to_num(gerrity_score_matrix, nan=-numpy.inf),
            method='average'
        ),
        gerrity_score_matrix.shape
    )
    pifl_gerrity_rank_matrix = numpy.reshape(
        rankdata(
            a=-1 * numpy.nan_to_num(pifl_gerrity_score_matrix, nan=-numpy.inf),
            method='average'
        ),
        pifl_gerrity_score_matrix.shape
    )
    peirce_rank_matrix = numpy.reshape(
        rankdata(
            a=-1 * numpy.nan_to_num(peirce_score_matrix, nan=-numpy.inf),
            method='average'
        ),
        peirce_score_matrix.shape
    )
    heidke_rank_matrix = numpy.reshape(
        rankdata(
            a=-1 * numpy.nan_to_num(heidke_score_matrix, nan=-numpy.inf),
            method='average'
        ),
        heidke_score_matrix.shape
    )
    cross_entropy_rank_matrix = numpy.reshape(
        rankdata(
            a=numpy.nan_to_num(cross_entropy_matrix, nan=numpy.inf),
            method='average'
        ),
        cross_entropy_matrix.shape
    )

    overall_rank_matrix = numpy.stack([
        top1_accuracy_rank_matrix,
        top2_accuracy_rank_matrix,
        top3_accuracy_rank_matrix,
        gerrity_rank_matrix,
        pifl_gerrity_rank_matrix,
        peirce_rank_matrix,
        heidke_rank_matrix,
        cross_entropy_rank_matrix
    ], axis=-1)

    overall_rank_matrix = numpy.mean(overall_rank_matrix, axis=-1)
    overall_rank_matrix[numpy.isnan(overall_rank_matrix)] = numpy.inf
    best_indices_linear = numpy.argsort(numpy.ravel(overall_rank_matrix))
    best_indices_linear = best_indices_linear[
        :numpy.sum(numpy.isfinite(gerrity_score_matrix))
    ]

    for p in range(len(best_indices_linear)):
        i, j, k = numpy.unravel_index(
            best_indices_linear[p], overall_rank_matrix.shape
        )

        print((
            '{0:05d}th-best overall rank ... '
            'number of dense layers = {1:d} ... '
            'L2 weight and dropout rate = {2:.10f}, {3:.1f} ... '
            'Gerrity loss? {4:s}  Class weights? {5:s}  '
            'PIFL first in loss?  {6:s} ... '
            'Accuracies = {7:.4f}, {8:.4f}, {9:.4f} ... '
            'Accuracy rankings = {10:.1f}, {11:.1f}, {12:.1f} ... '
            'Gerrity/PIFL-Gerrity/Heidke/Peirce = {13:.4f}, {14:.4f}, {15:.4f}, {16:.4f} ... '
            'G/PG/H/P rankings = {17:.1f}, {18:.1f}, {19:.1f}, {20:.1f} ... '
            'X-entropy = {21:.4f} ... '
            'X-entropy ranking = {22:.1f}'
        ).format(
            p + 1,
            dense_layer_counts_axis1[i],
            l2_weights_axis2[j],
            dropout_rates_axis2[j],
            'YES' if use_gerrity_loss_flags_axis3[k] else 'NO',
            'YES' if use_class_weights_flags_axis3[k] else 'NO',
            'YES' if put_pifl_first_flags_axis3[k] else 'NO',
            top1_accuracy_matrix[i, j, k],
            top2_accuracy_matrix[i, j, k],
            top3_accuracy_matrix[i, j, k],
            top1_accuracy_rank_matrix[i, j, k],
            top2_accuracy_rank_matrix[i, j, k],
            top3_accuracy_rank_matrix[i, j, k],
            gerrity_score_matrix[i, j, k],
            pifl_gerrity_score_matrix[i, j, k],
            heidke_score_matrix[i, j, k],
            peirce_score_matrix[i, j, k],
            gerrity_rank_matrix[i, j, k],
            pifl_gerrity_rank_matrix[i, j, k],
            heidke_rank_matrix[i, j, k],
            peirce_rank_matrix[i, j, k],
            cross_entropy_matrix[i, j, k],
            cross_entropy_rank_matrix[i, j, k]
        ))

    y_tick_labels = ['{0:d}'.format(d) for d in dense_layer_counts_axis1]
    
    dropout_rate_strings = ['{0:.3f}'.format(d) for d in dropout_rates_axis2]
    l2_weight_strings = [
        r'10$^{' + '{0:.1f}'.format(numpy.log10(w)) + r'}$'
        for w in l2_weights_axis2
    ]
    x_tick_labels = [
        '{0:s}, {1:s}'.format(a, b) for a, b in
        zip(dropout_rate_strings, l2_weight_strings)
    ]

    y_axis_label = 'Number of dense layers'
    x_axis_label = r'Dropout rate and L$_2$ weight'

    top1_accuracy_panel_file_names = [''] * axis3_length
    top2_accuracy_panel_file_names = [''] * axis3_length
    top3_accuracy_panel_file_names = [''] * axis3_length
    gerrity_panel_file_names = [''] * axis3_length
    pifl_gerrity_panel_file_names = [''] * axis3_length
    peirce_panel_file_names = [''] * axis3_length
    heidke_panel_file_names = [''] * axis3_length
    cross_entropy_panel_file_names = [''] * axis3_length

    _mkdir_recursive_if_necessary(directory_name=output_dir_name)

    if num_classes < 6:
        BEST_MARKER_SIZE_GRID_CELLS = 0.375 / 4
        SELECTED_MARKER_SIZE_GRID_CELLS = 0.25 / 4

    for k in range(axis3_length):
        title_string = (
            'Loss function = {0:s} with{1:s} class weights{2:s}'
        ).format(
            'Gerrity score' if use_gerrity_loss_flags_axis3[k]
            else 'cross-entropy',
            '' if use_class_weights_flags_axis3[k] else 'out',
            '\nwith PIFL first' if put_pifl_first_flags_axis3[k] else ''
        )

        # Plot top-1 accuracy for all CNNs with the [k]th loss function.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=top1_accuracy_matrix[..., k],
            min_colour_value=numpy.nanpercentile(top1_accuracy_matrix, 5.),
            max_colour_value=numpy.nanpercentile(top1_accuracy_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        best_indices = numpy.unravel_index(
            numpy.nanargmax(numpy.ravel(top1_accuracy_matrix)),
            top1_accuracy_matrix.shape
        )
        figure_width_px = (
            figure_object.get_size_inches()[0] * figure_object.dpi
        )
        best_marker_size_px = figure_width_px * (
            BEST_MARKER_SIZE_GRID_CELLS / top1_accuracy_matrix.shape[1]
        )
        selected_marker_size_px = figure_width_px * (
            SELECTED_MARKER_SIZE_GRID_CELLS / top1_accuracy_matrix.shape[1]
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=best_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )
        if selected_marker_indices[2] == k:
            axes_object.plot(
                selected_marker_indices[1], selected_marker_indices[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=selected_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        top1_accuracy_panel_file_names[k] = (
            '{0:s}/top1_accuracy_use-gerrity={1:d}_use-class-weights={2:d}_'
            'put-pifl-first-in-loss={3:d}.jpg'
        ).format(
            output_dir_name,
            int(use_gerrity_loss_flags_axis3[k]),
            int(use_class_weights_flags_axis3[k]),
            int(put_pifl_first_flags_axis3[k])
        )

        print('Saving figure to: "{0:s}"...'.format(
            top1_accuracy_panel_file_names[k]
        ))
        figure_object.savefig(
            top1_accuracy_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot top-2 accuracy for all CNNs with the [k]th loss function.
        if num_classes > 2:
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=top2_accuracy_matrix[..., k],
                min_colour_value=numpy.nanpercentile(top2_accuracy_matrix, 5.),
                max_colour_value=numpy.nanpercentile(top2_accuracy_matrix, 100.),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            axes_object.set_xlabel(x_axis_label)
            axes_object.set_ylabel(y_axis_label)
            axes_object.set_title(title_string)

            best_indices = numpy.unravel_index(
                numpy.nanargmax(numpy.ravel(top2_accuracy_matrix)),
                top2_accuracy_matrix.shape
            )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=best_marker_size_px, markeredgewidth=0,
                    markerfacecolor=MARKER_COLOUR,
                    markeredgecolor=MARKER_COLOUR
                )
            if selected_marker_indices[2] == k:
                axes_object.plot(
                    selected_marker_indices[1], selected_marker_indices[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=selected_marker_size_px, markeredgewidth=0,
                    markerfacecolor=MARKER_COLOUR,
                    markeredgecolor=MARKER_COLOUR
                )

            top2_accuracy_panel_file_names[k] = (
                '{0:s}/top2_accuracy_use-gerrity={1:d}_use-class-weights={2:d}_'
                'put-pifl-first-in-loss={3:d}.jpg'
            ).format(
                output_dir_name,
                int(use_gerrity_loss_flags_axis3[k]),
                int(use_class_weights_flags_axis3[k]),
                int(put_pifl_first_flags_axis3[k])
            )

            print('Saving figure to: "{0:s}"...'.format(
                top2_accuracy_panel_file_names[k]
            ))
            figure_object.savefig(
                top2_accuracy_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

        # Plot top-3 accuracy for all CNNs with the [k]th loss function.
        if num_classes > 3:
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=top3_accuracy_matrix[..., k],
                min_colour_value=numpy.nanpercentile(top3_accuracy_matrix, 5.),
                max_colour_value=numpy.nanpercentile(top3_accuracy_matrix, 100.),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            axes_object.set_xlabel(x_axis_label)
            axes_object.set_ylabel(y_axis_label)
            axes_object.set_title(title_string)

            if num_classes == 2:
                best_indices = numpy.full(3, -1, dtype=int)
            else:
                best_indices = numpy.unravel_index(
                    numpy.nanargmax(numpy.ravel(top3_accuracy_matrix)),
                    top3_accuracy_matrix.shape
                )

            if best_indices[2] == k:
                axes_object.plot(
                    best_indices[1], best_indices[0],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=best_marker_size_px, markeredgewidth=0,
                    markerfacecolor=MARKER_COLOUR,
                    markeredgecolor=MARKER_COLOUR
                )
            if selected_marker_indices[2] == k:
                axes_object.plot(
                    selected_marker_indices[1], selected_marker_indices[0],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=selected_marker_size_px, markeredgewidth=0,
                    markerfacecolor=MARKER_COLOUR,
                    markeredgecolor=MARKER_COLOUR
                )

            top3_accuracy_panel_file_names[k] = (
                '{0:s}/top3_accuracy_use-gerrity={1:d}_use-class-weights={2:d}_'
                'put-pifl-first-in-loss={3:d}.jpg'
            ).format(
                output_dir_name,
                int(use_gerrity_loss_flags_axis3[k]),
                int(use_class_weights_flags_axis3[k]),
                int(put_pifl_first_flags_axis3[k])
            )

            print('Saving figure to: "{0:s}"...'.format(
                top3_accuracy_panel_file_names[k]
            ))
            figure_object.savefig(
                top3_accuracy_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

        # Plot Gerrity score for all CNNs with the [k]th loss function.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=gerrity_score_matrix[..., k],
            min_colour_value=numpy.nanpercentile(gerrity_score_matrix, 5.),
            max_colour_value=numpy.nanpercentile(gerrity_score_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        best_indices = numpy.unravel_index(
            numpy.nanargmax(numpy.ravel(gerrity_score_matrix)),
            gerrity_score_matrix.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=best_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )
        if selected_marker_indices[2] == k:
            axes_object.plot(
                selected_marker_indices[1], selected_marker_indices[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=selected_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        gerrity_panel_file_names[k] = (
            '{0:s}/gerrity_use-gerrity={1:d}_use-class-weights={2:d}_'
            'put-pifl-first-in-loss={3:d}.jpg'
        ).format(
            output_dir_name,
            int(use_gerrity_loss_flags_axis3[k]),
            int(use_class_weights_flags_axis3[k]),
            int(put_pifl_first_flags_axis3[k])
        )

        print('Saving figure to: "{0:s}"...'.format(
            gerrity_panel_file_names[k]
        ))
        figure_object.savefig(
            gerrity_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot PIFL-first Gerrity score for all CNNs with the [k]th loss
        # function.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=pifl_gerrity_score_matrix[..., k],
            min_colour_value=numpy.nanpercentile(pifl_gerrity_score_matrix, 5.),
            max_colour_value=numpy.nanpercentile(pifl_gerrity_score_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        best_indices = numpy.unravel_index(
            numpy.nanargmax(numpy.ravel(pifl_gerrity_score_matrix)),
            pifl_gerrity_score_matrix.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=best_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )
        if selected_marker_indices[2] == k:
            axes_object.plot(
                selected_marker_indices[1], selected_marker_indices[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=selected_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        pifl_gerrity_panel_file_names[k] = (
            '{0:s}/pifl_first_gerrity_use-gerrity={1:d}_use-class-weights={2:d}_'
            'put-pifl-first-in-loss={3:d}.jpg'
        ).format(
            output_dir_name,
            int(use_gerrity_loss_flags_axis3[k]),
            int(use_class_weights_flags_axis3[k]),
            int(put_pifl_first_flags_axis3[k])
        )

        print('Saving figure to: "{0:s}"...'.format(
            pifl_gerrity_panel_file_names[k]
        ))
        figure_object.savefig(
            pifl_gerrity_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot Peirce score for all CNNs with the [k]th loss function.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=peirce_score_matrix[..., k],
            min_colour_value=numpy.nanpercentile(peirce_score_matrix, 5.),
            max_colour_value=numpy.nanpercentile(peirce_score_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        best_indices = numpy.unravel_index(
            numpy.nanargmax(numpy.ravel(peirce_score_matrix)),
            peirce_score_matrix.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=best_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )
        if selected_marker_indices[2] == k:
            axes_object.plot(
                selected_marker_indices[1], selected_marker_indices[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=selected_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        peirce_panel_file_names[k] = (
            '{0:s}/peirce_use-gerrity={1:d}_use-class-weights={2:d}_'
            'put-pifl-first-in-loss={3:d}.jpg'
        ).format(
            output_dir_name,
            int(use_gerrity_loss_flags_axis3[k]),
            int(use_class_weights_flags_axis3[k]),
            int(put_pifl_first_flags_axis3[k])
        )

        print('Saving figure to: "{0:s}"...'.format(
            peirce_panel_file_names[k]
        ))
        figure_object.savefig(
            peirce_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot Heidke score for all CNNs with the [k]th loss function.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=heidke_score_matrix[..., k],
            min_colour_value=numpy.nanpercentile(heidke_score_matrix, 5.),
            max_colour_value=numpy.nanpercentile(heidke_score_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        best_indices = numpy.unravel_index(
            numpy.nanargmax(numpy.ravel(heidke_score_matrix)),
            heidke_score_matrix.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=best_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )
        if selected_marker_indices[2] == k:
            axes_object.plot(
                selected_marker_indices[1], selected_marker_indices[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=selected_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        heidke_panel_file_names[k] = (
            '{0:s}/heidke_use-gerrity={1:d}_use-class-weights={2:d}_'
            'put-pifl-first-in-loss={3:d}.jpg'
        ).format(
            output_dir_name,
            int(use_gerrity_loss_flags_axis3[k]),
            int(use_class_weights_flags_axis3[k]),
            int(put_pifl_first_flags_axis3[k])
        )

        print('Saving figure to: "{0:s}"...'.format(
            heidke_panel_file_names[k]
        ))
        figure_object.savefig(
            heidke_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot cross-entropy for all CNNs with the [k]th loss function.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=cross_entropy_matrix[..., k],
            min_colour_value=numpy.nanpercentile(cross_entropy_matrix, 0.),
            max_colour_value=numpy.nanpercentile(cross_entropy_matrix, 95.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)

        best_indices = numpy.unravel_index(
            numpy.nanargmin(numpy.ravel(cross_entropy_matrix)),
            cross_entropy_matrix.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=best_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )
        if selected_marker_indices[2] == k:
            axes_object.plot(
                selected_marker_indices[1], selected_marker_indices[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=selected_marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        cross_entropy_panel_file_names[k] = (
            '{0:s}/cross_entropy_use-gerrity={1:d}_use-class-weights={2:d}_'
            'put-pifl-first-in-loss={3:d}.jpg'
        ).format(
            output_dir_name,
            int(use_gerrity_loss_flags_axis3[k]),
            int(use_class_weights_flags_axis3[k]),
            int(put_pifl_first_flags_axis3[k])
        )

        print('Saving figure to: "{0:s}"...'.format(
            cross_entropy_panel_file_names[k]
        ))
        figure_object.savefig(
            cross_entropy_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(axis3_length)
    ))
    num_panel_columns = int(numpy.ceil(
        float(axis3_length) / num_panel_rows
    ))

    top1_accuracy_concat_file_name = '{0:s}/top1_accuracy.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        top1_accuracy_concat_file_name
    ))
    _concatenate_images(
        input_file_names=top1_accuracy_panel_file_names,
        output_file_name=top1_accuracy_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    _resize_image(
        input_file_name=top1_accuracy_concat_file_name,
        output_file_name=top1_accuracy_concat_file_name,
        output_size_pixels=int(1e7)
    )

    if num_classes > 2:
        top2_accuracy_concat_file_name = '{0:s}/top2_accuracy.jpg'.format(
            output_dir_name
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            top2_accuracy_concat_file_name
        ))
        _concatenate_images(
            input_file_names=top2_accuracy_panel_file_names,
            output_file_name=top2_accuracy_concat_file_name,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
        )
        _resize_image(
            input_file_name=top2_accuracy_concat_file_name,
            output_file_name=top2_accuracy_concat_file_name,
            output_size_pixels=int(1e7)
        )

    if num_classes > 3:
        top3_accuracy_concat_file_name = '{0:s}/top3_accuracy.jpg'.format(
            output_dir_name
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            top3_accuracy_concat_file_name
        ))
        _concatenate_images(
            input_file_names=top3_accuracy_panel_file_names,
            output_file_name=top3_accuracy_concat_file_name,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
        )
        _resize_image(
            input_file_name=top3_accuracy_concat_file_name,
            output_file_name=top3_accuracy_concat_file_name,
            output_size_pixels=int(1e7)
        )

    gerrity_concat_file_name = '{0:s}/gerrity_score.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(
        gerrity_concat_file_name
    ))
    _concatenate_images(
        input_file_names=gerrity_panel_file_names,
        output_file_name=gerrity_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    _resize_image(
        input_file_name=gerrity_concat_file_name,
        output_file_name=gerrity_concat_file_name,
        output_size_pixels=int(1e7)
    )

    pifl_gerrity_concat_file_name = '{0:s}/pifl_first_gerrity_score.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        pifl_gerrity_concat_file_name
    ))
    _concatenate_images(
        input_file_names=pifl_gerrity_panel_file_names,
        output_file_name=pifl_gerrity_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    _resize_image(
        input_file_name=pifl_gerrity_concat_file_name,
        output_file_name=pifl_gerrity_concat_file_name,
        output_size_pixels=int(1e7)
    )

    peirce_concat_file_name = '{0:s}/peirce_score.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(
        peirce_concat_file_name
    ))
    _concatenate_images(
        input_file_names=peirce_panel_file_names,
        output_file_name=peirce_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    _resize_image(
        input_file_name=peirce_concat_file_name,
        output_file_name=peirce_concat_file_name,
        output_size_pixels=int(1e7)
    )

    heidke_concat_file_name = '{0:s}/heidke_score.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(
        heidke_concat_file_name
    ))
    _concatenate_images(
        input_file_names=heidke_panel_file_names,
        output_file_name=heidke_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    _resize_image(
        input_file_name=heidke_concat_file_name,
        output_file_name=heidke_concat_file_name,
        output_size_pixels=int(1e7)
    )

    cross_entropy_concat_file_name = '{0:s}/cross_entropy.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        cross_entropy_concat_file_name
    ))
    _concatenate_images(
        input_file_names=cross_entropy_panel_file_names,
        output_file_name=cross_entropy_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    _resize_image(
        input_file_name=cross_entropy_concat_file_name,
        output_file_name=cross_entropy_concat_file_name,
        output_size_pixels=int(1e7)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_classes=getattr(INPUT_ARG_OBJECT, NUM_CLASSES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
