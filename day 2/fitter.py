import matplotlib.pyplot as plt
import string
from math import *
import numpy as np

FILE_NAME = 'data.csv'  # Remember to include the extension
SKIP_FIRST_LINE = False
DELIMITER = ','  # Set to space, ' ', if working with .txt file without commas

# Plotting details
PLOT_TITLE = 'Zero Offset Data'
X_LABEL = r'$\theta$ (rad)'
Y_LABEL = r"Intensity (Counts per Second)$"
AUTO_X_LIMITS = True
X_LIMITS = [0., 10.]  # Not used unless AUTO_X_LIMITS = False
AUTO_Y_LIMITS = True
Y_LIMITS = [0., 10.]  # Not used unless AUTO_Y_LIMITS = False
LINE_COLOUR = 'red'  # See documentation for options:
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
LINE_STYLE = '-'   # See documentation for options:
# https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
MARKER_STYLE = 'x'  # See documentation for options:
# https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
MARKER_COLOUR = 'black'
GRID_LINES = True
SAVE_FIGURE = True
FIGURE_NAME = 'offset-error.png'
FIGURE_RESOLUTION = 600  # in dpi



def linear(U,x):
    return  U[0]*x**3 + U[1]*x**2 + U[2]*x + U[3]

def fitter(parameters, xdata, ydata):
    fit = np.polyfit(xdata, ydata, parameters - 1, cov = True)
    
    fit_parameters = fit[0]
    fit_parameter_error = np.sqrt(np.diag(fit[1]))
    
    return fit_parameters, fit_parameter_error

def check_numeric(entry):
    """Checks if entry is numeric
    Args:
        entry: string
    Returns:
        bool
    Raises:
        ValueError: if entry cannot be cast to float type
    """
    try:
        float(entry)
        return True
    except ValueError:
        return False

def validate_line(line):
    """Validates line. Outputs error messages accordingly.
    Args:
        line: string
    Returns:
        bool, if validation has been succesful
        line_floats, numpy array of floats
    """
    line_split = line.split(DELIMITER)

    for entry in line_split:
        if check_numeric(entry) is False:
            print('Line omitted: {0:s}.'.format(line.strip('\n')))
            print('{0:s} is nonnumerical.'.format(entry))
            return False, line_split
    line_floats = np.array([float(line_split[0]), float(line_split[1])])
   
    return True, line_floats


def open_file(file_name=FILE_NAME, skip_first_line=SKIP_FIRST_LINE):
    """Opens file, reads data and outputs data in numpy arrays.
    Args:
        file_name: string, default given by FILE_NAME
    Returns:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
    Raises:
        FileNotFoundError
    """
    # Create empty arrays ready to store the data
    x_data = np.array([])
    y_data = np.array([])
    try:
        raw_file_data = open(file_name, 'r')
    except FileNotFoundError:
        print("File '{0:s}' cannot be found.".format(file_name))
        print('Check it is in the correct directory.')
        return x_data, y_data
    for line in raw_file_data:
        if skip_first_line:
            skip_first_line = False
        else:
            line_valid, line_data = validate_line(line)
            if line_valid:
                x_data = np.append(x_data, line_data[0])
                y_data = np.append(y_data, line_data[1])

    raw_file_data.close()
    return x_data, y_data

def chi_squared_function(x_data, y_data, parameters):
    """Calculates the chi squared for the data given, assuming a linear
    relationship.
    Args:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
        parameters: numpy array of floats, [slope, offset]
    Returns:
        chi_squared: float
    """
    return np.sum(((y_data - linear(parameters, x_data))**2)/linear(parameters, x_data))


def create_plot(x_data, y_data, parameters,
                parameter_uncertainties):
    """Produces graphic of resulting fit
    Args:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
        parameters: numpy array of floats, [slope, offset]
        parameter_uncertainties: numpy array of floats, [slope_uncertainty,
                                 offset_uncertainty]
    Returns:
        None
    """
    x = np.linspace(x_data[0],x_data[-1], num=100)
    
    # Main plot
    figure = plt.figure(figsize=(8, 6))

    axes_main_plot = figure.add_subplot(211)

    axes_main_plot.plot(x_data, y_data,
                            MARKER_STYLE, color=MARKER_COLOUR)
    axes_main_plot.plot(x, linear(parameters, x),
                        color=LINE_COLOUR)
    axes_main_plot.grid(GRID_LINES)
    axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
    axes_main_plot.set_xlabel(X_LABEL)
    axes_main_plot.set_ylabel(Y_LABEL)
    # Fitting details
    chi_squared = chi_squared_function(x_data, y_data,
                                       parameters)
    degrees_of_freedom = len(x_data) - 2
    reduced_chi_squared = chi_squared / degrees_of_freedom

    axes_main_plot.annotate((r'$\chi^2$ = {0:4.2f}'.
                             format(chi_squared)), (1, 0), (-60, -35),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('Degrees of freedom = {0:d}'.
                             format(degrees_of_freedom)), (1, 0), (-147, -55),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate((r'Reduced $\chi^2$ = {0:4.2f}'.
                             format(reduced_chi_squared)), (1, 0), (-104, -70),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate('Fit: $y=mx+c$', (0, 0), (0, -35),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points')
    axes_main_plot.annotate(('m = {0:6.4e}'.format(parameters[0])), (0, 0),
                            (0, -55), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('± {0:6.4e}'.format(parameter_uncertainties[0])),
                            (0, 0), (100, -55), xycoords='axes fraction',
                            va='top', fontsize='10',
                            textcoords='offset points')
    axes_main_plot.annotate(('c = {0:6.4e}'.format(parameters[1])), (0, 0),
                            (0, -70), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('± {0:6.4e}'.format(parameter_uncertainties[1])),
                            (0, 0), (100, -70), xycoords='axes fraction',
                            textcoords='offset points', va='top',
                            fontsize='10')
    # Residuals plot

    if not AUTO_X_LIMITS:
        axes_main_plot.set_xlim(X_LIMITS)

    if not AUTO_Y_LIMITS:
        axes_main_plot.set_ylim(Y_LIMITS)

    if SAVE_FIGURE:
        plt.savefig(FIGURE_NAME, dpi=FIGURE_RESOLUTION, transparent=True)
    plt.show()
    return None

def main():
    x, y = open_file()
    parameter, error = fitter(4, x, y)
    
    create_plot(x, y, parameter, error)
     
    x_lin = np.linspace(x[0],x[-1], num=200)

    maximum = np.max(linear(parameter, x_lin))
    maximum_index = np.where(linear(parameter, x_lin) == maximum)
    
    print(x_lin[maximum_index])
    
main()