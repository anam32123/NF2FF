## import statements:
import numpy as np
from matplotlib.pyplot import *
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from numpy.fft import ifftshift
from scipy.interpolate import interp2d

def exclude_7sigma_outliers(data):
    '''This function excludes outliers >7sigma from the mean value.
    This may seem rare, but with this type of transformation, occurs more commonly than one might think.'''
    sigma_data=np.std(data)
    preliminary_mean=np.mean(data)
    data[np.abs((data-preliminary_mean) > (7*sigma_data))]=float('nan')

    return data

def electric_field_errors (original_field,perturbed_field,x,y,label_1,label_2,outputs=0):
    '''Returns a bunch of metrics in a Python dictionary called error_metrics that quantify the
    error between original_field and perturbed_field. Can also produce output plots if outputs argument
    is set to 1. Meant to be used in conjunction with NF2FF.
    Takes in as inputs two electric fields, and the labels for axes on plots
    x and y are values along the two axes (theta and phi for a far-field pattern)
    set outputs to 1 if want text and plot outputs. 0 (default) supresses outputs
    '''

    [x_grid,y_grid]=np.meshgrid(x,y)

    error_metrics={} # making a dictionary to store the error metrics. Eventually, hopefully this will be
                     # its own class or contained within an electric field class set
    error_metrics['error']=perturbed_field-original_field
    error_real=error_metrics['error'].real
    error_im=error_metrics['error'].imag
    original_field_real=original_field.real
    original_field_im=original_field.imag
    perturbed_field_real=perturbed_field.real
    perturbed_field_im=perturbed_field.imag
    error_metrics['error_amplitude']=np.abs(error_metrics['error'])
    error_metrics['error_phase']=np.angle(error_metrics['error'])
    original_field_amplitude=np.abs(original_field)
    perturbed_field_amplitude=np.abs(perturbed_field)
    original_field_phase=np.angle(original_field)
    perturbed_field_phase=np.angle(perturbed_field)

    error_real=exclude_7sigma_outliers(error_real)
    error_im=exclude_7sigma_outliers(error_im)

    # calculating weighted percent difference
    # As a vector difference of real and imaginary components
    weighted_percent_difference_real=error_real/((original_field_real+perturbed_field_real)/2)
    weighted_percent_difference_im=error_im/((original_field_im+perturbed_field_im)/2)
    error_metrics['weighted_percent_difference']= weighted_percent_difference_real + 1j*weighted_percent_difference_im
    error_metrics['weighted_percent_difference_amplitude']=np.abs(error_metrics['weighted_percent_difference'])
    error_metrics['weighted_percent_difference_phase']=np.angle(error_metrics['weighted_percent_difference'])
    # Method 1 for mean: taking mean of amplitude and phase errors
    error_metrics['weighted_percent_difference_amplitude_mean']=np.nanmean(error_metrics['weighted_percent_difference_amplitude'])
    error_metrics['weighted_percent_difference_phase_mean']=np.nanmean(error_metrics['weighted_percent_difference_phase'])
    # Method 2: finding mean weighted percent difference by averaging real and imaginary components separately
    weighted_percent_difference_real_mean=np.nanmean(weighted_percent_difference_real)
    weighted_percent_difference_im_mean=np.nanmean(weighted_percent_difference_im)
    error_metrics['weighted_percent_difference_mean']=weighted_percent_difference_real_mean + 1j*weighted_percent_difference_im_mean
    error_metrics['weighted_percent_difference_mean_amplitude']=np.abs(error_metrics['weighted_percent_difference_mean'])
    error_metrics['weighted_percent_difference_mean_phase']=np.angle(error_metrics['weighted_percent_difference_mean'])

    # range of the error--calculated as max - min of the error (complex subtraction between the two fields)
    error_range_real=np.nanmax(error_real)-np.nanmin(error_real)
    error_range_im=np.nanmax(error_im)-np.nanmin(error_im)
    error_metrics['error_range']=error_range_real + 1j*error_range_im
    error_metrics['error_range_amplitude']=np.abs(error_metrics['error_range'])
    error_metrics['error_range_phase']=np.angle(error_metrics['error_range'])
    # doing the same for the weighted percent difference
    pct_difference_range_real=np.nanmax(weighted_percent_difference_real) - np.nanmin(weighted_percent_difference_real)
    pct_difference_range_im = np.nanmax(weighted_percent_difference_im) - np.nanmin(weighted_percent_difference_im)
    error_metrics['pct_difference_range'] = pct_difference_range_real + 1j*pct_difference_range_im
    error_metrics['pct_difference_range_amplitude'] = np.abs(error_metrics['pct_difference_range'])
    error_metrics['pct_difference_range_phase'] = np.angle(error_metrics['pct_difference_range'])

    # standard deviation of the error
    std_error_real=np.nanstd(error_real)
    std_error_im=np.nanstd(error_im)
    error_metrics['std_error'] = std_error_real + 1j*std_error_im
    error_metrics['std_error_amplitude'] = np.abs(error_metrics['std_error'])
    error_metrics['std_error_phase'] = np.angle(error_metrics['std_error'])
    # standard deviation of the percent error
    pct_diff_std_error_real = np.nanstd(weighted_percent_difference_real)
    pct_diff_std_error_im = np.nanstd(weighted_percent_difference_im)
    error_metrics['pct_diff_std_error'] = pct_diff_std_error_real + 1j*pct_diff_std_error_im
    error_metrics['pct_diff_std_error_amplitude'] = np.abs(error_metrics['pct_diff_std_error'])
    error_metrics['pct_diff_std_error_phase'] = np.angle(error_metrics['pct_diff_std_error'])

    # overall magnitude of the error: abs(original) - abs(perturbed)
    error_metrics['error_magnitudes'] = np.abs(original_field) - np.abs(perturbed_field)
    error_metrics['error_magnitudes_mean'] = np.nanmean(error_metrics['error_magnitudes'])
    error_metrics['pct_error_magnitudes'] = np.divide(error_metrics['error_magnitudes'],(np.abs(original_field) + np.abs(perturbed_field))/2)
    error_metrics['pct_error_magnitudes_mean'] = np.nanmean(error_metrics['pct_error_magnitudes'])

    # excluding data beyond the main beam to calculate the fractional solid angle error
    dB_cutoff = -145 # this is the cutoff for the sample data provided with the MATLAB script
    power_cutoff = 10**(dB_cutoff/20)
    original_excluded_indices = original_field_amplitude < power_cutoff
    original_field_amplitude[original_excluded_indices] = float('nan')
    original_field_real[original_excluded_indices] = float('nan')
    original_field_im[original_excluded_indices] = float('nan')
    perturbed_excluded_indices = perturbed_field_amplitude < power_cutoff
    perturbed_field_amplitude[perturbed_excluded_indices] = float('nan')
    perturbed_field_real[perturbed_excluded_indices] = float('nan')
    perturbed_field_im[perturbed_excluded_indices] = float('nan')

    # calculating fractional change in solid angle for the excluded data set
    solid_angle_original = np.nansum(original_field_amplitude)
    solid_angle_perturbed = np.nansum(perturbed_field_amplitude)
    error_metrics['solid_angle_error_fraction'] = np.abs((solid_angle_original - solid_angle_perturbed)/solid_angle_original)

    # can possibly add in the vector version, but I don't think this is right

    if outputs==1:
        fig,ax=subplots(1,2,constrained_layout=True,figsize=(17,10))
        pcm=ax[0].pcolormesh(x,y,error_metrics['error_amplitude'])
        ax[0].set_aspect('equal')
        ax[0].set_title('Difference between perturbed and original fields: amplitude')
        ax[0].set_xlabel(label_1)
        ax[0].set_ylabel(label_2)
        colorbar(mappable=pcm,ax=ax[0],shrink=0.7)
        pcm=ax[1].pcolormesh(x,y,error_metrics['error_phase'])
        ax[1].set_aspect('equal')
        ax[1].set_title('Difference between perturbed and original fields: phase')
        ax[1].set_xlabel(label_1)
        ax[1].set_ylabel(label_2)
        colorbar(mappable=pcm,ax=ax[1],shrink=0.7)

        # plotting percent difference over the entire 2D grid
        fig,ax=subplots()
        pcolor(x,y,error_metrics['weighted_percent_difference_amplitude'],vmax=1)
        ax.set_aspect('equal')
        ax.set_xlabel(label_1)
        ax.set_ylabel(label_2)
        ax.set_title('Weighted percent difference: amplitude')
        colorbar()

        # print statements
        print('Amplitude and phase of mean weighted percent difference (vector form): {}, {}'.format(error_metrics['weighted_percent_difference_mean_amplitude'],error_metrics['weighted_percent_difference_mean_phase']))
        print('Mean weighted percent difference in amplitude and phase: {}, {}'.format(error_metrics['weighted_percent_difference_amplitude_mean'],error_metrics['weighted_percent_difference_phase_mean']))
        print('Amplitude and phase of error range: {}, {}'.format(error_metrics['error_range_amplitude'],error_metrics['error_range_phase']))
        print('Amplitude and phase of percent difference range: {}, {}'.format(error_metrics['pct_difference_range_amplitude'],error_metrics['pct_difference_range_phase']))
        print('Amplitude and phase of standard deviation of the error: {}, {}'.format(error_metrics['std_error_amplitude'],error_metrics['std_error_phase']))
        print('Amplitude and phase of standard deviation of percent error: {}, {}'.format(error_metrics['pct_diff_std_error_amplitude'],error_metrics['pct_diff_std_error_phase']))
        print('Mean of amplitude(b1) - amplitude(b2): {}'.format(error_metrics['error_magnitudes_mean']))
        print('Mean of pct difference amplitude metric: {}'.format(error_metrics['pct_error_magnitudes_mean']))
        print('Solid angle error fraction (excluded): {}'.format(error_metrics['solid_angle_error_fraction']))

    return error_metrics
