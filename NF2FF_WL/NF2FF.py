## CLONING the NF2FF Script from MATLAB: https://www.mathworks.com/matlabcentral/fileexchange/23385-nf2ff

## import statements:
import numpy as np
from matplotlib.pyplot import *
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from numpy.fft import ifftshift
import pandas
from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline

def RegularGridding(x,y,dtheta,dphi):
    '''
    Produces k-space grid and parameters for given x, y vectors (this may only work for the traditional
    NF2FF case, not offsets).'''

    M = len(x)
    N = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # setting up k-space, theta, phi grids
    # See equations (16-13a) and (16-13b) in Balanis
    # Zero padding is used to increase the resolution of the plane wave spectral domain.
    MI=4*M;#2^(ceil(log2(M))+1);
    NI=4*N;#2^(ceil(log2(N))+1);
    m=np.linspace(-MI/2,MI/2-1,MI);
    n=np.linspace(-NI/2,NI/2-1,NI)
    k_X_Rectangular=(2*np.pi*m)/(MI*dx);
    k_Y_Rectangular=(2*np.pi*n)/(NI*dy);
    theta=np.arange(-np.pi/2+dtheta,np.pi/2-dtheta,dtheta);
    phi=np.arange(0+dphi,np.pi-dphi,dphi);
    [theta,phi]=np.meshgrid(theta,phi);

    return [m,n,k_X_Rectangular,k_Y_Rectangular,theta,phi]

def NyquistSampling(broadcast_freq):
    '''
    Returns the Nyquist spacing in m for broadcasting at a given frequency (input as broad_freq).
    '''

    c=299792458.0 # Speed of light in vacuum [m/s]

    lambda_m=c/broadcast_freq
    nyquist_sampling_m=lambda_m/2; # necessary grid spacing in m

    return nyquist_sampling_m

def Calculate_k0(freq):
    '''
    Returns the k0 wave number for measurements at a certain frequency (input as freq).
    '''

    c=299792458.0 # Speed of light in vacuum [m/s]

    # calculating k0
    lambda0=c/freq;
    k0=2*np.pi/lambda0;

    return k0

def NFtoFourier (NF_X,NF_Y,kx,ky,k0,plotFourier=False):

    '''Function to bring NF data into k-space (plane wave spectrum). Returns a list that returns
    plane wave spectrum in x, y, z polarizations, along with their respective magnitudes.
    Input parameters:
    NF_X and NF_Y: NF data in x and y polarizations on a rectangular grid
    kx and ky: spectral frequency vector components
    k0: wave number
    plotFourier: if True, plots the plane wave spectrum over the k-space grid'''

    [kx_grid,ky_grid]=np.meshgrid(kx,ky)

    grid_size = np.shape(kx_grid)
    MI = grid_size[0]
    NI = grid_size[1]

    kz_grid=np.emath.sqrt((k0**2.0)-(kx_grid**2.0)-(ky_grid**2.0));
    fx=ifftshift(ifft2(NF_X,[MI,NI],axes=[0,1]))
    fy=ifftshift(ifft2(NF_Y,[MI,NI],axes=[0,1]))
    fz=(-1.0*((fx*kx_grid)+(fy*ky_grid)))/kz_grid

    # this magnitude is not in dB, can be plotted in dB by taking log and multiplying by 20 or simply using lognorm plotting
    fx_magnitude=(np.abs(fx));
    fy_magnitude=(np.abs(fy));
    fz_magnitude=(np.abs(fz));

    if plotFourier == True:
        fig,[ax1,ax2,ax3]=subplots(nrows=1,ncols=3,figsize=(16,5.5))
        ax1.pcolormesh(kx_grid,ky_grid,fx_magnitude,cmap='gnuplot2',norm=LogNorm())
        ax2.pcolormesh(kx_grid,ky_grid,fy_magnitude,cmap='gnuplot2',norm=LogNorm())
        ax3.pcolormesh(kx_grid,ky_grid,fz_magnitude,cmap='gnuplot2',norm=LogNorm())
        ax1.set_title('f_x')
        ax2.set_title('f_y')
        ax3.set_title('f_z')
        for ax in [ax1,ax2,ax3]:
            ax.set_aspect('equal')
            ax.set_xlabel('k_x')
            ax.set_ylabel('k_y')
        tight_layout()

    return [fx, fy, fz, fx_magnitude, fy_magnitude, fz_magnitude]

def FouriertoFF(f_x,f_y,f_z,theta,phi,k_x,k_y,k0,FFOutputs=False):

    '''Brings data from plave wave spectrum to the far-field pattern, by first inteprolating the data into
    spherical, then applying transformation.
    Returns list with [Etheta,Ephi]
    Input parameters:
    f_x, f_y, f_z: plane wave spectrum in x, y, z polarizations
    theta: vector of theta values for spherical coordinates
    phi: vector of phi values for spherical coordinates
    k_x and k_y: spectral frequency vector components
    k0: wave number (2pi/lambda0)
    plotFourier: if True, plots final FF pattern'''

    xx = k0*np.sin(theta)*np.cos(phi)
    yy = k0*np.sin(theta)*np.sin(phi)

    f_x_interp = RectBivariateSpline(k_x,k_y,np.abs(f_x.T))
    f_y_interp = RectBivariateSpline(k_x,k_y,np.abs(f_y.T))
    f_z_interp = RectBivariateSpline(k_x,k_y,np.abs(f_z.T))

    f_X_Spherical = f_x_interp.ev(xx,yy)
    f_Y_Spherical = f_y_interp.ev(xx,yy)
    f_Z_Spherical = f_z_interp.ev(xx,yy)

    r=10000;
    C=1.0j*(k0*np.exp(-1.0j*k0*r))/(2*np.pi*r)
    Etheta=C*(f_X_Spherical*np.cos(phi)+f_Y_Spherical*np.sin(phi));
    Ephi=C*np.cos(theta)*(-f_X_Spherical*np.sin(phi)+f_Y_Spherical*np.cos(phi));

    if FFOutputs==True:
        fig=figure(figsize=(9,9))
        ax=fig.add_subplot(2,2,1)
        pcm=ax.pcolormesh(theta,phi,20*np.log10(np.abs(Etheta)),cmap='gnuplot2')
        ax.set_aspect('equal')
        ax.set_title('Etheta')
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        colorbar(mappable=pcm,ax=ax)
        ax=fig.add_subplot(2,2,2)
        pcm=ax.pcolormesh(theta,phi,20*np.log10(np.abs(Ephi)),cmap='gnuplot2')
        ax.set_aspect('equal')
        ax.set_title('Ephi')
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        colorbar(mappable=pcm,ax=ax)
        ax=fig.add_subplot(2,2,3,projection='3d')
        ax.plot_surface(theta,phi,20.0*np.log10(np.abs(Etheta))-np.nanmax(20.0*np.log10(np.abs(Etheta.T.conj()))),cmap='gnuplot2')
        ax.view_init(30,-37.5)
        ax.set_title('Etheta')
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        ax=fig.add_subplot(2,2,4,projection='3d')
        ax.plot_surface(theta,phi,20.0*np.log10(np.abs(Ephi))-np.nanmax(20.0*np.log10(np.abs(Ephi.T.conj()))),cmap='gnuplot2')
        ax.view_init(30,-37.5)
        ax.set_title('Ephi')
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        fig.tight_layout()

    return [Etheta,Ephi]
