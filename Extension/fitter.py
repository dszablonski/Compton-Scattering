import matplotlib.pyplot as plt
import string
from scipy.optimize import curve_fit
from math import *
import numpy as np

FILE_NAME = 'data.csv'  # Remember to include the extension
SKIP_FIRST_LINE = True
DELIMITER = ','  # Set to space, ' ', if working with .txt file without commas

# Plotting details
PLOT_TITLE = 'Extension'
X_LABEL = r'$1 - \cos\theta$'
Y_LABEL = r"$1/E'$ $(\text{keV}^{-1})$"
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
FIGURE_NAME = 'data1-with-offset-error.png'
FIGURE_RESOLUTION = 600  # in dpi



def linear(U,x):
    return U[0]*x + U[1]

def linear2(x, a, b):
    return a*x + b

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

    line_floats = np.array([float(line_split[0]), float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])])
   
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
    theta = np.array([])
    iron = np.array([])
    yew = np.array([])
    lead = np.array([])
    aluminium = np.array([])
    try:
        raw_file_data = open(file_name, 'r')
    except FileNotFoundError:
        print("File '{0:s}' cannot be found.".format(file_name))
        print('Check it is in the correct directory.')
        return theta, iron, yew, lead, aluminium
    for line in raw_file_data:
        if skip_first_line:
            skip_first_line = False
        else:
            line_valid, line_data = validate_line(line)
            if line_valid:
                print(line_data)
                theta = np.append(theta, line_data[0])
                aluminium = np.append(aluminium, line_data[1])
                iron = np.append(iron, line_data[2])
                yew = np.append(yew, line_data[3])
                lead = np.append(lead, line_data[4])

    raw_file_data.close()
    return theta, iron, yew, lead, aluminium

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


def create_plot(theta, a, b, c, d, parameters,
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
    # Main plot
    figure = plt.figure(figsize=(8, 6))
    
    fig, axes_main_plot = plt.subplots(subplot_kw={'projection': 'polar'})
    
    axes_main_plot.plot((theta), (a),".", color="r", label="Z = 13")
    axes_main_plot.plot((theta), (b),".", color="b", label="Z = 26")
    axes_main_plot.plot((theta), (c),".", color="g", label="Z = 6.87")
    axes_main_plot.plot((theta), (d),".", color="purple", label="Z = 82")
    
    """
    axes_main_plot.plot(x_data, linear(parameters, x_data),
                        color=LINE_COLOUR)
    """
    axes_main_plot.grid(GRID_LINES)
    axes_main_plot.set_rticks([])
    axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
    # Fitting details
    chi_squared = chi_squared_function(x_data, y_data,
                                       parameters)
    degrees_of_freedom = len(x_data) - 2
    reduced_chi_squared = chi_squared / degrees_of_freedom
    

    """
    axes_main_plot.annotate((r'$\chi^2$ = {0:4.10f}'.
                             format(chi_squared)), (1, 0), (-60, -35),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('Degrees of freedom = {0:d}'.
                             format(degrees_of_freedom)), (1, 0), (-147, -55),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate((r'Reduced $\chi^2$ = {0:4.10f}'.
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
    """
    
    if not AUTO_X_LIMITS:
        axes_main_plot.set_xlim(X_LIMITS)

    if not AUTO_Y_LIMITS:
        axes_main_plot.set_ylim(Y_LIMITS)

    if SAVE_FIGURE:
        plt.savefig(FIGURE_NAME, dpi=FIGURE_RESOLUTION, transparent=True)
    plt.show()
    plt.legend()
    return None

def main():
    theta, iron, yew, lead, aluminium = open_file()
    parameter, error = fitter(2, theta, iron)
    
    popt, pcov = curve_fit(linear2, theta, iron)
    errors = np.sqrt(np.diag(pcov))
    
    print(popt)
    print(errors)
    
    print("{:.4e}, {:.4e}".format(parameter[0], parameter[1]))
    print("{:.4e}, {:.4e}".format(error[0], error[1]))
    
    create_plot(theta, iron, yew, lead, aluminium, parameter, error)
    
main()