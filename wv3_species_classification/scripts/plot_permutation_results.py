"""Plots results of permutation-based importance test."""

import os
import errno
import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import percentileofscore

ORIGINAL_COST_KEY = 'orig_cost_estimates'
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COSTS_KEY = 'best_cost_matrix'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COSTS_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG_KEY = 'is_backwards_test'

BAR_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

BAR_EDGE_WIDTH = 2
BAR_EDGE_COLOUR = numpy.full(3, 0.)

REFERENCE_LINE_WIDTH = 4
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

ERROR_BAR_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
ERROR_BAR_CAP_SIZE = 8
ERROR_BAR_DICT = {'alpha': 1., 'linewidth': 4, 'capthick': 4}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

BAR_TEXT_COLOUR = numpy.full(3, 0.)
BAR_FONT_SIZE = 22
DEFAULT_FONT_SIZE = 30

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

DEFAULT_CONVERT_EXE_NAME = 'convert'
DEFAULT_MONTAGE_EXE_NAME = 'montage'
IMAGEMAGICK_ERROR_STRING = (
    '\nUnix command failed (log messages shown above should explain why).'
)

FORWARD_TEST_FILE_ARG_NAME = 'input_forward_test_file_name'
BACKWARDS_TEST_FILE_ARG_NAME = 'input_backwards_test_file_name'
NUM_PREDICTORS_ARG_NAME = 'num_predictors_to_plot'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

FORWARD_TEST_FILE_HELP_STRING = (
    'Path to Pickle file with results of forward test, created by '
    'run_permutation.py.'
)
BACKWARDS_TEST_FILE_HELP_STRING = (
    'Path to Pickle file with results of backwards test, created by '
    'run_permutation.py.'
)
NUM_PREDICTORS_HELP_STRING = (
    'Will plot the M most important predictors in each bar graph, where M is '
    'this input arg.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for error bars (in range 0...1).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FORWARD_TEST_FILE_ARG_NAME, type=str, required=True,
    help=FORWARD_TEST_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BACKWARDS_TEST_FILE_ARG_NAME, type=str, required=True,
    help=BACKWARDS_TEST_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PREDICTORS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


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


def _get_error_matrix(cost_matrix, confidence_level, backwards_flag,
                      multipass_flag):
    """Creates error matrix (used to plot error bars).

    S = number of steps in permutation test
    B = number of bootstrap replicates

    :param cost_matrix: S-by-B numpy array of costs.
    :param confidence_level: Confidence level (in range 0...1).
    :param backwards_flag: Boolean flag, indicating whether the test is forward
        or backwards.
    :param multipass_flag: Boolean flag, indicating whether the test is
        single-pass or multi-pass.
    :return: error_matrix: 2-by-S numpy array, where the first row contains
        negative errors and second row contains positive errors.
    :return: significant_flags: length-S numpy array of Boolean flags.  If
        significant_flags[i] = True, the [i]th step has a significantly
        different cost than the [i + 1]th step.
    """

    num_steps = cost_matrix.shape[0]
    significant_flags = numpy.full(num_steps, False, dtype=bool)

    for i in range(num_steps - 1):
        if backwards_flag:
            these_diffs = cost_matrix[i + 1, :] - cost_matrix[i, :]
        else:
            these_diffs = cost_matrix[i, :] - cost_matrix[i + 1, :]

        # if not multipass_flag:
        #     these_diffs *= -1

        print(numpy.mean(these_diffs))

        this_percentile = percentileofscore(
            a=these_diffs, score=0., kind='mean'
        )

        if multipass_flag:
            significant_flags[i] = this_percentile <= 5.
        else:
            significant_flags[i + 1] = this_percentile <= 5.

        print((
            'Percentile of 0 in (cost at step {0:d}) - (cost at step {1:d}) = '
            '{2:.4f}'
        ).format(
            i + 1, i, this_percentile
        ))

    print(significant_flags)
    print('\n')

    median_costs = numpy.median(cost_matrix, axis=-1)
    min_costs = numpy.percentile(
        cost_matrix, 50 * (1. - confidence_level), axis=-1
    )
    max_costs = numpy.percentile(
        cost_matrix, 50 * (1. + confidence_level), axis=-1
    )

    negative_errors = median_costs - min_costs
    positive_errors = max_costs - median_costs

    negative_errors = numpy.reshape(negative_errors, (1, negative_errors.size))
    positive_errors = numpy.reshape(positive_errors, (1, positive_errors.size))
    error_matrix = numpy.vstack((negative_errors, positive_errors))

    return error_matrix, significant_flags


def _label_bars(axes_object, y_tick_coords, y_tick_strings, significant_flags):
    """Labels bars in graph.

    J = number of bars

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param y_tick_coords: length-J numpy array with y-coordinates of bars.
    :param y_tick_strings: length-J list of labels.
    :param significant_flags: length-J numpy array of Boolean flags.  If
        significant_flags[i] = True, the [i]th step has a significantly
        different cost than the [i + 1]th step.
    """

    for j in range(len(y_tick_coords)):
        axes_object.text(
            0., y_tick_coords[j], '      ' + y_tick_strings[j],
            color=BAR_TEXT_COLOUR, horizontalalignment='left',
            verticalalignment='center',
            fontweight='bold' if significant_flags[j] else 'normal',
            fontsize=BAR_FONT_SIZE
        )


def _plot_bars(
        cost_matrix, clean_cost_array, predictor_names, backwards_flag,
        multipass_flag, confidence_level, axes_object):
    """Plots bar graph for either single-pass or multi-pass test.

    P = number of predictors permuted or depermuted
    B = number of bootstrap replicates

    :param cost_matrix: (P + 1)-by-B numpy array of costs.  The first row
        contains costs at the beginning of the test -- before (un)permuting any
        variables -- and the [i]th row contains costs after (un)permuting the
        variable represented by predictor_names[i - 1].
    :param clean_cost_array: length-B numpy array of costs with clean
        (unpermuted) predictors.
    :param predictor_names: length-P list of predictor names (used to label
        bars).
    :param backwards_flag: Boolean flag.  If True, will plot backwards version
        of permutation, where each step involves *un*permuting a variable.  If
        False, will plot forward version, where each step involves permuting a
        variable.
    :param multipass_flag: Boolean flag.  If True, plotting multi-pass version
        of test.  If False, plotting single-pass version.
    :param confidence_level: Confidence level for error bars (in range 0...1).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    """

    median_clean_cost = numpy.median(clean_cost_array)

    if backwards_flag:
        y_tick_strings = ['All permuted'] + predictor_names
    else:
        y_tick_strings = ['None permuted'] + predictor_names

    y_tick_coords = numpy.linspace(
        0, len(y_tick_strings) - 1, num=len(y_tick_strings), dtype=float
    )

    if multipass_flag:
        y_tick_coords = y_tick_coords[::-1]

    median_costs = numpy.median(cost_matrix, axis=-1)
    num_steps = cost_matrix.shape[0]
    num_bootstrap_reps = cost_matrix.shape[1]

    if num_bootstrap_reps > 1:
        error_matrix, significant_flags = _get_error_matrix(
            cost_matrix=cost_matrix, confidence_level=confidence_level,
            backwards_flag=backwards_flag, multipass_flag=multipass_flag
        )

        x_min = numpy.min(median_costs - error_matrix[0, :])
        x_max = numpy.max(median_costs + error_matrix[1, :])

        axes_object.barh(
            y_tick_coords, median_costs, color=BAR_FACE_COLOUR,
            edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH,
            xerr=error_matrix,
            ecolor=ERROR_BAR_COLOUR, capsize=ERROR_BAR_CAP_SIZE,
            error_kw=ERROR_BAR_DICT
        )
    else:
        significant_flags = numpy.full(num_steps, False, dtype=bool)
        x_min = numpy.min(median_costs)
        x_max = numpy.max(median_costs)

        axes_object.barh(
            y_tick_coords, median_costs, color=BAR_FACE_COLOUR,
            edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH
        )

    reference_x_coords = numpy.full(2, median_clean_cost)
    reference_y_tick_coords = numpy.array([
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    ])

    axes_object.plot(
        reference_x_coords, reference_y_tick_coords,
        color=REFERENCE_LINE_COLOUR, linestyle='--',
        linewidth=REFERENCE_LINE_WIDTH
    )

    axes_object.set_yticks([], [])

    if backwards_flag:
        axes_object.set_ylabel('Variable cleaned')
    else:
        axes_object.set_ylabel('Variable permuted')

    axes_object.set_xlim(
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    )

    x_max *= 1.01
    if x_min <= 0:
        x_min *= 1.01
    else:
        x_min = 0.

    axes_object.set_xlim(x_min, x_max)

    _label_bars(
        axes_object=axes_object, y_tick_coords=y_tick_coords,
        y_tick_strings=y_tick_strings, significant_flags=significant_flags
    )

    axes_object.set_ylim(
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    )


def _plot_single_pass_test(
        permutation_dict, axes_object, confidence_level,
        num_predictors_to_plot):
    """Plots results of single-pass (Breiman) permutation test.

    :param permutation_dict: Dictionary created by
        `permutation.run_forward_test` or `permutation.run_backwards_test`.
    :param axes_object: See doc for `_plot_bars`.
    :param confidence_level: See documentation at top of file.
    :param num_predictors_to_plot: Same.
    """

    # Check input args.
    predictor_names = permutation_dict[STEP1_PREDICTORS_KEY]
    backwards_flag = permutation_dict[BACKWARDS_FLAG_KEY]
    perturbed_cost_matrix = permutation_dict[STEP1_COSTS_KEY]
    median_perturbed_costs = numpy.median(perturbed_cost_matrix, axis=-1)

    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    if backwards_flag:
        sort_indices = numpy.argsort(
            median_perturbed_costs
        )[-num_predictors_to_plot:]
    else:
        sort_indices = numpy.argsort(
            median_perturbed_costs
        )[:num_predictors_to_plot][::-1]

    perturbed_cost_matrix = perturbed_cost_matrix[sort_indices, :]
    predictor_names = [predictor_names[k] for k in sort_indices]

    original_cost_array = permutation_dict[ORIGINAL_COST_KEY]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    # Do plotting.
    if backwards_flag:
        clean_cost_array = permutation_dict[BEST_COSTS_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    _plot_bars(
        cost_matrix=cost_matrix, clean_cost_array=clean_cost_array,
        predictor_names=predictor_names,
        backwards_flag=backwards_flag, multipass_flag=False,
        confidence_level=confidence_level, axes_object=axes_object
    )


def _plot_multipass_test(permutation_dict, axes_object, confidence_level,
                         num_predictors_to_plot):
    """Plots results of multi-pass (Lakshmanan) permutation test.

    :param permutation_dict: See doc for `_plot_single_pass_test`.
    :param axes_object: Same.
    :param confidence_level: Same.
    :param num_predictors_to_plot: Same.
    """

    predictor_names = permutation_dict[BEST_PREDICTORS_KEY]
    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    backwards_flag = permutation_dict[BACKWARDS_FLAG_KEY]
    perturbed_cost_matrix = (
        permutation_dict[BEST_COSTS_KEY][:num_predictors_to_plot, :]
    )
    predictor_names = predictor_names[:num_predictors_to_plot]

    original_cost_array = permutation_dict[ORIGINAL_COST_KEY]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    if backwards_flag:
        clean_cost_array = permutation_dict[BEST_COSTS_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    _plot_bars(
        cost_matrix=cost_matrix, clean_cost_array=clean_cost_array,
        predictor_names=predictor_names,
        backwards_flag=backwards_flag, multipass_flag=True,
        confidence_level=confidence_level, axes_object=axes_object
    )


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


def _run(forward_test_file_name, backwards_test_file_name,
         num_predictors_to_plot, confidence_level, output_dir_name):
    """Plots results of permutation-based importance test.

    This is effectively the main method.

    :param forward_test_file_name: See documentation at top of file.
    :param backwards_test_file_name: Same.
    :param num_predictors_to_plot: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    if num_predictors_to_plot < 1:
        num_predictors_to_plot = None

    print('Reading data from: "{0:s}"...'.format(forward_test_file_name))
    pickle_file_handle = open(forward_test_file_name, 'rb')
    forward_test_result_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    print(len(set(forward_test_result_dict[BEST_PREDICTORS_KEY])))
    print(len(set(forward_test_result_dict[STEP1_PREDICTORS_KEY])))

    print('Reading data from: "{0:s}"...'.format(backwards_test_file_name))
    pickle_file_handle = open(backwards_test_file_name, 'rb')
    backwards_test_result_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    print(len(set(backwards_test_result_dict[BEST_PREDICTORS_KEY])))
    print(len(set(backwards_test_result_dict[STEP1_PREDICTORS_KEY])))

    pathless_file_name = os.path.split(forward_test_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    species_name_for_positive_class = extensionless_file_name.split('_')[-1]

    pathless_file_name = os.path.split(backwards_test_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    this_species_name = extensionless_file_name.split('_')[-1]

    assert species_name_for_positive_class == this_species_name

    if species_name_for_positive_class == 'all':
        species_name_for_positive_class = None

    if species_name_for_positive_class is None:
        loss_function_name = 'Gerrity score'
    else:
        loss_function_name = (
            'AUC for binary classification:\n{0:s} vs. everything else'
        ).format(species_name_for_positive_class)

    assert not forward_test_result_dict[BACKWARDS_FLAG_KEY]
    assert backwards_test_result_dict[BACKWARDS_FLAG_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    _plot_single_pass_test(
        permutation_dict=forward_test_result_dict, axes_object=axes_object,
        confidence_level=confidence_level,
        num_predictors_to_plot=num_predictors_to_plot
    )
    axes_object.set_xlabel(loss_function_name)
    axes_object.set_title('(a) Single-pass forward test')

    panel_file_names = [
        '{0:s}/single_pass_forward_test.jpg'.format(output_dir_name)
    ]
    _mkdir_recursive_if_necessary(directory_name=output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    _plot_multipass_test(
        permutation_dict=forward_test_result_dict, axes_object=axes_object,
        confidence_level=confidence_level,
        num_predictors_to_plot=num_predictors_to_plot
    )
    axes_object.set_xlabel(loss_function_name)
    axes_object.set_title('(b) Multi-pass forward test')

    panel_file_names.append(
        '{0:s}/multi_pass_forward_test.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    _plot_single_pass_test(
        permutation_dict=backwards_test_result_dict, axes_object=axes_object,
        confidence_level=confidence_level,
        num_predictors_to_plot=num_predictors_to_plot
    )
    axes_object.set_xlabel(loss_function_name)
    axes_object.set_title('(c) Single-pass backwards test')

    panel_file_names.append(
        '{0:s}/single_pass_backwards_test.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    _plot_multipass_test(
        permutation_dict=backwards_test_result_dict, axes_object=axes_object,
        confidence_level=confidence_level,
        num_predictors_to_plot=num_predictors_to_plot
    )
    axes_object.set_xlabel(loss_function_name)
    axes_object.set_title('(d) Multi-pass backwards test')

    panel_file_names.append(
        '{0:s}/multi_pass_backwards_test.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}/permutation_test_all_versions.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    _concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    _resize_image(
        input_file_name=concat_file_name,
        output_file_name=concat_file_name,
        output_size_pixels=int(1e7)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        forward_test_file_name=getattr(
            INPUT_ARG_OBJECT, FORWARD_TEST_FILE_ARG_NAME
        ),
        backwards_test_file_name=getattr(
            INPUT_ARG_OBJECT, BACKWARDS_TEST_FILE_ARG_NAME
        ),
        num_predictors_to_plot=getattr(
            INPUT_ARG_OBJECT, NUM_PREDICTORS_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
