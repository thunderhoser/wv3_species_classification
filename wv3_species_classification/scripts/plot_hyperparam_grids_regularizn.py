"""Plots hyperparameter grids for regularization experiment."""

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
from keras.metrics import top_k_categorical_accuracy

MIN_PROB_FOR_XENTROPY = numpy.finfo(float).eps
MAX_PROB_FOR_XENTROPY = 1. - numpy.finfo(float).eps

L2_WEIGHTS = numpy.logspace(-7, -5, num=5, dtype=float)
DROPOUT_RATES = numpy.linspace(0.5, 0.8, num=5, dtype=float)
DENSE_LAYER_COUNTS = numpy.linspace(3, 7, num=5, dtype=int)

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
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing all models for the experiment.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
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


def _get_gerrity_score(contingency_matrix):
    """Computes Gerrity score.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: gerrity_score: Gerrity score (range -1...1).
    """

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

    pyplot.xticks(x_tick_values, x_tick_labels)
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
        font_size=FONT_SIZE
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


def _run(experiment_dir_name, output_dir_name):
    """Plots hyperparameter grids for regularization experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)
    num_dropout_rates = len(DROPOUT_RATES)
    num_l2_weights = len(L2_WEIGHTS)
    num_classes = -1

    dimensions = (num_dense_layer_counts, num_dropout_rates, num_l2_weights)
    top1_accuracy_matrix = numpy.full(dimensions, numpy.nan)
    top2_accuracy_matrix = numpy.full(dimensions, numpy.nan)
    top3_accuracy_matrix = numpy.full(dimensions, numpy.nan)
    gerrity_score_matrix = numpy.full(dimensions, numpy.nan)
    peirce_score_matrix = numpy.full(dimensions, numpy.nan)
    heidke_score_matrix = numpy.full(dimensions, numpy.nan)
    cross_entropy_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(num_dense_layer_counts):
        for j in range(num_dropout_rates):
            for k in range(num_l2_weights):
                this_prediction_file_name = (
                    '{0:s}/l2-weight={1:.10f}_dropout-rate={2:.10f}_'
                    'num-dense-layers={3:012d}/oob_predictions.p'
                ).format(
                    experiment_dir_name, L2_WEIGHTS[k], DROPOUT_RATES[j],
                    DENSE_LAYER_COUNTS[i]
                )

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
                        y_true=this_target_matrix,
                        y_pred=this_probability_matrix, k=1
                    )
                )
                top2_accuracy_matrix[i, j, k] = numpy.mean(
                    top_k_categorical_accuracy(
                        y_true=this_target_matrix,
                        y_pred=this_probability_matrix, k=2
                    )
                )

                if num_classes > 2:
                    top3_accuracy_matrix[i, j, k] = numpy.mean(
                        top_k_categorical_accuracy(
                            y_true=this_target_matrix,
                            y_pred=this_probability_matrix, k=3
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
                    this_confusion_matrix
                )
                peirce_score_matrix[i, j, k] = _get_peirce_score(
                    this_confusion_matrix
                )
                heidke_score_matrix[i, j, k] = _get_heidke_score(
                    this_confusion_matrix
                )

    y_tick_labels = ['{0:.3f}'.format(d) for d in DROPOUT_RATES]
    x_tick_labels = [
        r'10$^{' + '{0:.1f}'.format(numpy.log10(w)) + r'}$' for w in L2_WEIGHTS
    ]

    y_axis_label = 'Dropout rate'
    x_axis_label = r'L$_2$ weight'

    top1_accuracy_panel_file_names = [''] * num_dense_layer_counts
    top2_accuracy_panel_file_names = [''] * num_dense_layer_counts
    top3_accuracy_panel_file_names = [''] * num_dense_layer_counts
    gerrity_panel_file_names = [''] * num_dense_layer_counts
    peirce_panel_file_names = [''] * num_dense_layer_counts
    heidke_panel_file_names = [''] * num_dense_layer_counts
    cross_entropy_panel_file_names = [''] * num_dense_layer_counts

    _mkdir_recursive_if_necessary(directory_name=output_dir_name)

    for i in range(num_dense_layer_counts):

        # Plot top-1 accuracy for all CNNs with the [i]th dense-layer count.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=top1_accuracy_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(top1_accuracy_matrix, 5.),
            max_colour_value=numpy.nanpercentile(top1_accuracy_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(
            'Top-1 accuracy for CNNs with {0:d} dense layers'.format(
                DENSE_LAYER_COUNTS[i]
            )
        )

        top1_accuracy_panel_file_names[i] = (
            '{0:s}/top1_accuracy_num-dense-layers={1:d}.jpg'
        ).format(output_dir_name, DENSE_LAYER_COUNTS[i])

        print('Saving figure to: "{0:s}"...'.format(
            top1_accuracy_panel_file_names[i]
        ))
        figure_object.savefig(
            top1_accuracy_panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot top-2 accuracy for all CNNs with the [i]th dense-layer count.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=top2_accuracy_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(top2_accuracy_matrix, 5.),
            max_colour_value=numpy.nanpercentile(top2_accuracy_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(
            'Top-2 accuracy for CNNs with {0:d} dense layers'.format(
                DENSE_LAYER_COUNTS[i]
            )
        )

        top2_accuracy_panel_file_names[i] = (
            '{0:s}/top2_accuracy_num-dense-layers={1:d}.jpg'
        ).format(output_dir_name, DENSE_LAYER_COUNTS[i])

        print('Saving figure to: "{0:s}"...'.format(
            top2_accuracy_panel_file_names[i]
        ))
        figure_object.savefig(
            top2_accuracy_panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot top-3 accuracy for all CNNs with the [i]th dense-layer count.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=top3_accuracy_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(top3_accuracy_matrix, 5.),
            max_colour_value=numpy.nanpercentile(top3_accuracy_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(
            'Top-3 accuracy for CNNs with {0:d} dense layers'.format(
                DENSE_LAYER_COUNTS[i]
            )
        )

        top3_accuracy_panel_file_names[i] = (
            '{0:s}/top3_accuracy_num-dense-layers={1:d}.jpg'
        ).format(output_dir_name, DENSE_LAYER_COUNTS[i])

        print('Saving figure to: "{0:s}"...'.format(
            top3_accuracy_panel_file_names[i]
        ))
        figure_object.savefig(
            top3_accuracy_panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot Gerrity score for all CNNs with the [i]th dense-layer count.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=gerrity_score_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(gerrity_score_matrix, 5.),
            max_colour_value=numpy.nanpercentile(gerrity_score_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(
            'Gerrity score for CNNs with {0:d} dense layers'.format(
                DENSE_LAYER_COUNTS[i]
            )
        )

        gerrity_panel_file_names[i] = (
            '{0:s}/gerrity_num-dense-layers={1:d}.jpg'
        ).format(output_dir_name, DENSE_LAYER_COUNTS[i])

        print('Saving figure to: "{0:s}"...'.format(
            gerrity_panel_file_names[i]
        ))
        figure_object.savefig(
            gerrity_panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot Peirce score for all CNNs with the [i]th dense-layer count.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=peirce_score_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(peirce_score_matrix, 5.),
            max_colour_value=numpy.nanpercentile(peirce_score_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(
            'Peirce score for CNNs with {0:d} dense layers'.format(
                DENSE_LAYER_COUNTS[i]
            )
        )

        peirce_panel_file_names[i] = (
            '{0:s}/peirce_num-dense-layers={1:d}.jpg'
        ).format(output_dir_name, DENSE_LAYER_COUNTS[i])

        print('Saving figure to: "{0:s}"...'.format(
            peirce_panel_file_names[i]
        ))
        figure_object.savefig(
            peirce_panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot Heidke score for all CNNs with the [i]th dense-layer count.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=heidke_score_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(heidke_score_matrix, 5.),
            max_colour_value=numpy.nanpercentile(heidke_score_matrix, 100.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(
            'Heidke score for CNNs with {0:d} dense layers'.format(
                DENSE_LAYER_COUNTS[i]
            )
        )

        heidke_panel_file_names[i] = (
            '{0:s}/heidke_num-dense-layers={1:d}.jpg'
        ).format(output_dir_name, DENSE_LAYER_COUNTS[i])

        print('Saving figure to: "{0:s}"...'.format(
            heidke_panel_file_names[i]
        ))
        figure_object.savefig(
            heidke_panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot cross-entropy for all CNNs with the [i]th dense-layer count.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=cross_entropy_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(cross_entropy_matrix, 0.),
            max_colour_value=numpy.nanpercentile(cross_entropy_matrix, 95.),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(
            'Cross-entropy for CNNs with {0:d} dense layers'.format(
                DENSE_LAYER_COUNTS[i]
            )
        )

        cross_entropy_panel_file_names[i] = (
            '{0:s}/cross_entropy_num-dense-layers={1:d}.jpg'
        ).format(output_dir_name, DENSE_LAYER_COUNTS[i])

        print('Saving figure to: "{0:s}"...'.format(
            cross_entropy_panel_file_names[i]
        ))
        figure_object.savefig(
            cross_entropy_panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_dense_layer_counts)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_dense_layer_counts) / num_panel_rows
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
