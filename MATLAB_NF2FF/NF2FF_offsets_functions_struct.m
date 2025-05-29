%% In this script: consolidating the data for each offset and non-offset case into a struct to facilitate writing functions
% 
% Near-Field to Far-Field Transformation for Antenna Measurements (NF2FF)
% J. Logan (john_logan@mail.uri.edu)
% A. P. Mynster (andersmynster@hotmail.com)
% M. J. Pelk (m.j.pelk@tudelft.nl)
% C. Ponder (chris.ponder@sli-institute.ac.uk)
% K. Van Caekenberghe (vcaeken@umich.edu) 
% 
% Bug fixes:
% 1. Aspect ratio maintained during zero padding
%
% New extensions include:
% 1. Holographic back projection. Hologram analysis can also be used to study near field communication (NFC) antennas. 
% 2. Cross polarization (Ludwig 1 and 3)
% 3. Probe compensation (EXPERIMENTAL)
%
% The script assumes:
% 1. Rectangular coordinate system with z axis normal to planar aperture
% 2. exp(j*omega*t) time dependence convention. Please, substitute i with -j
%    whenever implementing exp(-i*omega*t) time dependence convention based
%    algorithms.
%
% The script uses:
% 1. Near field datasets of a 94 GHz slotted waveguide (U.S. Patent No.: 7,994,969)
%    can be downloaded from: http://www-personal.umich.edu/~vcaeken/DATASETS.zip.
% 2. GoldsteinUnwrap2D.m, posted on the fileexchangewebsite by B. Spottiswoode on 
%    22 December 2008
% 3. Sphere3D.m, posted on the fileexchange website by J. M. De Freitas on 
%    15 September 2005
% 
% Sought-after extensions are listed below. Please post them on the 
% website, if you are willing to contribute:
% 1. Cylindrical and spherical near field to far field transformation
% 2. Graphical user interface
%
% References:
% 1. C. A. Balanis, "Antenna Theory, Analysis and Design, 2nd Ed.", Wiley, 1997. [exp(j*omega*t) time dependence convention]
% 2. D. Paris, W. Leach, Jr., E. Joy, "Basic Theory of Probe-Compensated Near-Field Measurements", IEEE Transactions on Antennas and Propagation, Vol. 26, No. 3, May 1978. [exp(-i*omega*t) time dependence convention]
% 3. A. D. Yaghjian, "Approximate Formulas for the Far Field and Gain of Open-Ended Rectangular Waveguide", IEEE Transactions on Antennas and Propagation, Vol. 32, No. 4, April 1984. [exp(-i*omega*t) time dependence convention]
% 4. A. D. Yaghjian, "An Overview of Near-Field Antenna Measurements", IEEE Transactions on Antennas and Propagation, Vol. 34, No. 1, June 1986. [exp(-i*omega*t) time dependence convention]
% 5. G. F. Masters, "Probe-Correction Coefficients Derived From Near-Field Measurements", AMTA Conference, October 7-11, 1991.
% 6. J. J. Lee, E. M. Ferren, D. P. Woollen, and K. M. Lee, "Near-Field Probe Used as a Diagnostic Tool to Locate Defective Elements in an Array Antenna", IEEE Transactions on Antennas and Propagation, Vol. 36, No. 6, June 1988. [exp(-i*omega*t) time dependence convention]
% 7. http://www.fftw.org/
% 8. R. M. Goldstein, H. A. Zebken, and C. L. Werner, "Satellite Radar Interferometry: Two-Dimensional Phase Unwrapping", Radio Sci., Vol. 23, No. 4, pp. 713-720, 1988.
% 9. D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping: Theory, Algorithms and Software". Wiley-Interscience, 1998.
% 10. J. M. De Freitas. "SPHERE3D: A Matlab Function to Plot 3-Dimensional Data on a Spherical Surface".
%    QinetiQ Ltd, Winfrith Technology Centre, Winfrith,
%    Dorchester DT2 8XJ. UK. 15 September 2005.

%%
clc;
close all;
clear all;

%% loading in data and relevant variables
% loads in two different measurement data structures
load('scanarray_pol2_h6mm-10_2_2009.mat'); % Measurement data
sdata2=sdata;
load('scanarray_pol1_h6mm-10_2_2009.mat'); % Measurement data
freq=sdata.freq; % Measured frequency points [Hz]
sdata_offset=sdata;

% my interested variables
c=299792458; % Speed of light in vacuum [m/s]
broadcast_freq=94*10^9; % broadcast frequency in hertz

%% Plotting initial data

% plot_frequency_spectra_time_series(freq,sdata);

%% creating our gridding

% See equations (16-10a) and (16-10b) in Balanis
M=sdata.xpoints; % Amount of samples in the x direction (along table, left to right)
N=sdata.ypoints; % Amount of samples in the y direction (across table, front to back)
dx=sdata.x_step/1000; % Sample spacing in the x direction [m]--I assume spacing is then in mm in data structure
dy=sdata.y_step/1000; % Sample spacing in the y direction [m]

% See equations (16-10a) and (16-10b) in Balanis
a=dx*(M-1); % The length of the scanned area in the x direction [m]
b=dy*(N-1); % The length of the scanned area in the y direction [m]
sdata.x=[-a/2:a/(M-1):a/2]; % arrays of relevant x- and y-values created by spacing by dx/dy
sdata.y=[-b/2:b/(N-1):b/2]; % between -a/2 and a/2, -b/2 and b/2
sdata_offset.y=sdata.y;
z0=0.006;

% nyquist sampling and offsets from that
lambda_m=c/broadcast_freq;
nyquist_sampling_m=lambda_m/2; % necessary grid spacing in m
pct_offsets=[0.001,0.01,0.1];
offsets=nyquist_sampling_m*pct_offsets;
index=3;
sdata_offset.current_offset=offsets(index);
sdata_offset.x=sdata.x+sdata_offset.current_offset;

% See equations (16-13a) and (16-13b) in Balanis
% Zero padding is used to increase the resolution of the plane wave spectral domain.
MI=4*M;%2^(ceil(log2(M))+1);
NI=4*N;%2^(ceil(log2(N))+1);
sdata.m=[-MI/2:1:MI/2-1];
sdata.n=[-NI/2:1:NI/2-1];
sdata_offset.n=sdata.n;
% defining k-vectors and their components, and making that into a grid for
% kX and kY
sdata.k_X_Rectangular=2*pi*sdata.m/(MI*dx);
sdata.k_Y_Rectangular=2*pi*sdata.n/(NI*dy);
[sdata.k_Y_Rectangular_Grid,sdata.k_X_Rectangular_Grid] = meshgrid(sdata.k_Y_Rectangular,sdata.k_X_Rectangular);

% creating offset m, kx, ky, etc.
sdata_offset.m=sdata_offset.x/dx;
sdata_offset.m_spaced=linspace(sdata_offset.m(1)*4,sdata_offset.m(M)*4,MI);
% defining offset k grid
sdata_offset.k_X_Rectangular=2*pi*sdata_offset.m_spaced/(MI*dx);
sdata_offset.k_Y_Rectangular=2*pi*sdata_offset.n/(NI*dy);
[sdata_offset.k_Y_Rectangular_Grid,sdata_offset.k_X_Rectangular_Grid] = meshgrid(sdata_offset.k_Y_Rectangular,sdata_offset.k_X_Rectangular);

% for test purposes
% k_X_Rectangular_Grid=k_X_Rectangular_Offset_Grid;

% creating theta and phi grids from -pi/2 to pi/2 and 0 to pi
% (respectively)
dtheta=0.05;
dphi=0.05;
theta=[-pi/2+dtheta:dtheta:pi/2-dtheta];
phi=[0+dphi:dphi:pi-dphi];
[theta,phi]=meshgrid(theta,phi);

%% calculations

Index = 1;
for f_Index = 201:1:201 %1:1:N Only does this at the 201st frequency?? why--maybe only interested in one frequency and you can adjust it
    
%     close all;
    f=freq(f_Index);
    lambda0=c/f;
    k0=2*pi/lambda0;
    sdata.k_Z_Rectangular_Grid = sqrt(k0^2-sdata.k_X_Rectangular_Grid.^2-sdata.k_Y_Rectangular_Grid.^2);

    % offset version of k_z
    sdata_offset.k_Z_Rectangular_Grid=sqrt(k0^2-sdata_offset.k_X_Rectangular_Grid.^2-sdata_offset.k_Y_Rectangular_Grid.^2);

    %% loading in and plotting all NF data at given frequency

    for iy=1:1:N
       for ix=1:1:M 
           sdata.NF_X_Complex(ix,iy)=sdata.s21{iy,ix}(f_Index);
           sdata.NF_Y_Complex(ix,iy)=sdata2.s21{iy,ix}(f_Index);
       end
    end

    sdata_offset.NF_X_Complex=sdata.NF_X_Complex;
    sdata_offset.NF_Y_Complex=sdata.NF_Y_Complex;

    sdata.NF_X_Magnitude = 20*log10(abs(sdata.NF_X_Complex));
    sdata.NF_Y_Magnitude = 20*log10(abs(sdata.NF_Y_Complex));
    sdata_offset.NF_X_Magnitude=sdata.NF_X_Magnitude;
    sdata_offset.NF_Y_Magnitude=sdata.NF_Y_Magnitude;

    figure;
    subplot(2,2,1)
    surf(sdata.x*1000,sdata.y*1000,sdata.NF_X_Magnitude');
    title(sprintf('f = %f GHz (z = %i mm)--without offset',f/1000000000,z0*1000));
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{x}| (dB)');
    set(gca,'XLim',[min(sdata.x)*1000 max(sdata.x)*1000]);
    set(gca,'YLim',[min(sdata.y)*1000 max(sdata.y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(2,2,3);
    surf(sdata.x*1000,sdata.y*1000,sdata.NF_Y_Magnitude');
    title('Without offset')
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{y}| (dB)');
    set(gca,'XLim',[min(sdata.x)*1000 max(sdata.x)*1000]);
    set(gca,'YLim',[min(sdata.y)*1000 max(sdata.y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    %print(gcf,'-dtiff',['NF_dB_' num2str(f) '_GHz']);
    
    % adding offset plot
    subplot(2,2,2)
    surf(sdata_offset.x*1000,sdata_offset.y*1000,sdata_offset.NF_X_Magnitude');
    title(sprintf('f = %f GHz (z = %i mm)--with x offset',f/1000000000,z0*1000));
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{x}| (dB)');
    set(gca,'XLim',[min(sdata_offset.x)*1000 max(sdata_offset.x)*1000]);
    set(gca,'YLim',[min(sdata_offset.y)*1000 max(sdata_offset.y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(2,2,4);
    surf(sdata_offset.x*1000,sdata_offset.y*1000,sdata_offset.NF_Y_Magnitude');
    title('With x offset')
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{y}| (dB)');
    set(gca,'XLim',[min(sdata_offset.x)*1000 max(sdata_offset.x)*1000]);
    set(gca,'YLim',[min(sdata_offset.y)*1000 max(sdata_offset.y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;

    clear sdata.NF_X_Magnitude sdata.NF_Y_Magnitude;

%% plotting NF phase data

%     try
%         
%         [NF_X_Phase] = GoldsteinUnwrap2D(NF_X_Complex);
%         [NF_Y_Phase] = GoldsteinUnwrap2D(NF_Y_Complex);
%         NF_Slots_X_Phase(Index,:) = interp2(x,y,NF_X_Phase',0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],[0 0 0 0 0 0 0 0],'spline');
%         NF_Slots_Y_Phase(Index,:) = interp2(x,y,NF_Y_Phase',0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],[0 0 0 0 0 0 0 0],'spline');
%         NF_Slots_X_Offset_Phase(Index,:) = interp2(x_offset,y,NF_X_Phase',0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],[0 0 0 0 0 0 0 0],'spline');
%         NF_Slots_Y_Offset_Phase(Index,:) = interp2(x_offset,y,NF_Y_Phase',0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],[0 0 0 0 0 0 0 0],'spline');
% 
%         figure;
%         subplot(2,2,1)
%         surf(x*1000,y*1000,NF_X_Phase');
%         title(sprintf('f = %f GHz (z = %i mm)--no offsets in this plot for now',f/1000000000,z0*1000));
%         xlabel('x (mm)');
%         ylabel('y (mm)');
%         zlabel('\angle E_{x} (rad)');
%         set(gca,'XLim',[min(x)*1000 max(x)*1000]);
%         set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%         view(-37.5,30);
%         shading flat;
%         colorbar;
%         subplot(2,2,2);
%         imagesc(NF_X_Phase');
%         colorbar;
%         subplot(2,2,3);
%         surf(x*1000,y*1000,NF_Y_Phase');
%         xlabel('x (mm)');
%         ylabel('y (mm)');
%         zlabel('\angle E_{y} (rad)');
%         set(gca,'XLim',[min(x)*1000 max(x)*1000]);
%         set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%         view(-37.5,30);
%         shading flat;
%         colorbar;
%         subplot(2,2,4);
%         imagesc(NF_Y_Phase');
%         colorbar;
%         %print(gcf,'-dtiff',['NF_rad_' num2str(f) '_GHz']);
%         clear NF_X_Phase NF_Y_Phase;
%         
%     end

    %% transforming to Fourier space and plotting

    % achieved by taking into account kx, ky, fx, fy
    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,f_X_Rectangular_Magnitude,f_Y_Rectangular_Magnitude,f_Z_Rectangular_Magnitude]=NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Grid,k_Y_Rectangular_Grid,k_Z_Rectangular_Grid,MI,NI);

    [f_X_Rectangular_Test,f_Y_Rectangular_Test,f_Z_Rectangular_Offset_Test,f_X_Rectangular_Magnitude_Test,f_Y_Rectangular_Magnitude_Test,f_Z_Rectangular_Offset_Magnitude_Test]=NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Offset_Grid,k_Y_Rectangular_Offset_Grid,k_Z_Rectangular_Offset_Grid,MI,NI);

    NFtoFourierSpace_struct(sdata);

    all(sdata.f_X_Rectangular==f_X_Rectangular,'all');
    all(sdata.f_Y_Rectangular==f_Y_Rectangular,'all');
    all(sdata.f_Y_Rectangular==f_Y_Rectangular,'all');

    % plotting in Fourier space--cannot add any offset here because it's a
    % function of kx, ky, not x, y
%     fourier_fig=figure;
%     title(sprintf('f = %f GHz',f/1000000000));
%     subplot(3,2,1);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_X_Rectangular_Magnitude');
%     title('With no offset')
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{x}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(3,2,3);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_Y_Rectangular_Magnitude');
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{y}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(3,2,5);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_Z_Rectangular_Magnitude');
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{z}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
% 
%     % plotting offset version of this
%     subplot(3,2,2);
%     surf(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,f_X_Rectangular_Magnitude');
%     title(sprintf('With a %f mm offset, %f percent of Nyquist spacing',current_offset*1000,pct_offsets(index)*100));
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{x}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular_Offset) max(k_X_Rectangular_Offset)]);
%     set(gca,'YLim',[min(k_Y_Rectangular_Offset) max(k_Y_Rectangular_Offset)]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(3,2,4);
%     surf(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,f_Y_Rectangular_Magnitude');
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{y}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular_Offset) max(k_X_Rectangular_Offset)]);
%     set(gca,'YLim',[min(k_Y_Rectangular_Offset) max(k_Y_Rectangular_Offset)]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(3,2,6);
%     surf(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,f_Z_Rectangular_Offset_Magnitude');
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{z}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular_Offset) max(k_X_Rectangular_Offset)]);
%     set(gca,'YLim',[min(k_Y_Rectangular_Offset) max(k_Y_Rectangular_Offset)]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;

    % fx, fy, fz residuals--fx and fy are the same, so there is not need to
    % get residuals on that (if we are doing the subtracting grids rather
    % than interpolation method)

    f_Z_Rectangular_residuals=f_Z_Rectangular-f_Z_Rectangular_Offset;
    f_Z_Rectangular_residuals_phase=abs(angle(f_Z_Rectangular_residuals));
%     f_Z_Rectangular_Magnitude_residuals=f_Z_Rectangular_Magnitude-f_Z_Rectangular_Offset_Magnitude;
%     f_Z_Rectangular_residuals_Magnitude=20*log10(abs(f_Z_Rectangular_residuals));
    f_Z_Rectangular_residuals_Magnitude=abs(f_Z_Rectangular_residuals);
    f_Z_Rectangular_residuals_phase_percent_error=f_Z_Rectangular_residuals_phase/abs(angle(f_Z_Rectangular))*100;
    f_Z_Rectangular_residuals_Magnitude_percent_error=f_Z_Rectangular_residuals_Magnitude/abs(f_Z_Rectangular)*100;

    fprintf('Mean of the residuals amplitudes: %f\n', mean(f_Z_Rectangular_residuals_Magnitude, 'all'));
    fprintf('Maximum of residuals amplitudes (max error): %f\n', max(f_Z_Rectangular_residuals_Magnitude, [], 'all'));
    fprintf('Mean percent error on residuals amplitudes: %f\n', mean(f_Z_Rectangular_residuals_Magnitude_percent_error, 'all'));
    fprintf('Maximum percent error on residuals amplitudes (max error): %f\n', max(f_Z_Rectangular_residuals_Magnitude_percent_error, [], 'all'));
    fprintf('Mean of the phase residuals: %f\n', mean(f_Z_Rectangular_residuals_phase, 'all'));
    fprintf('Maximum of the phase residuals (max error): %f\n', max(f_Z_Rectangular_residuals_phase, [], 'all'));
    fprintf('Mean percent error on phase residuals: %f\n', mean(f_Z_Rectangular_residuals_phase_percent_error, 'all'));
    fprintf('Maximum percent error on phase residuals: %f\n', max(f_Z_Rectangular_residuals_phase_percent_error, [], 'all'));

    % plot magnitude residuals
    figure;
    subplot(2,1,1);
    surf(k_X_Rectangular,k_Y_Rectangular,f_Z_Rectangular_residuals_Magnitude);
    title('Plotting magnitude of complex residuals');
    xlabel(sprintf('k_{x} {m^{-1}}'));
    ylabel(sprintf('k_{y} {m_{-1}}'));
    zlabel('|f_{z}| residuals (dB)');
    colorbar;
    subplot(2,1,2);
    surf(k_X_Rectangular,k_Y_Rectangular,f_Z_Rectangular_residuals_phase);
    title('Plotting residual phase');
    xlabel(sprintf('k_{x} {m^{-1}}'));
    ylabel(sprintf('k_{y} {m_{-1}}'));
    zlabel('|f_{z}| residuals (dB)');
    colorbar;



    %% Holographic projection back to the NF from the FF
    
    % Holographic back projection
    for iy = 1:1:NI
        for ix = 1:1:MI
            if(isreal(k_Z_Rectangular_Grid(ix,iy))) % propagating or evanescent?
                f_X_z0_Rectangular(ix,iy)=f_X_Rectangular(ix,iy)*exp(j*k_Z_Rectangular_Grid(ix,iy)*z0);
                f_Y_z0_Rectangular(ix,iy)=f_Y_Rectangular(ix,iy)*exp(j*k_Z_Rectangular_Grid(ix,iy)*z0);
                f_Z_z0_Rectangular(ix,iy)=-(f_X_z0_Rectangular(ix,iy)*k_X_Rectangular_Grid(ix,iy)+f_Y_z0_Rectangular(ix,iy)*k_Y_Rectangular_Grid(ix,iy))/k_Z_Rectangular_Grid(ix,iy);
            else
                f_X_z0_Rectangular(ix,iy)=0;
                f_Y_z0_Rectangular(ix,iy)=0;
                f_Z_z0_Rectangular(ix,iy)=0;
            end 
        end
    end

    % Holographic back projection with offset
    for iy = 1:1:NI
        for ix = 1:1:MI
            if(isreal(k_Z_Rectangular_Offset_Grid(ix,iy))) % propagating or evanescent?
                f_X_z0_Rectangular_Offset(ix,iy)=f_X_Rectangular(ix,iy)*exp(j*k_Z_Rectangular_Offset_Grid(ix,iy)*z0);
                f_Y_z0_Rectangular_Offset(ix,iy)=f_Y_Rectangular(ix,iy)*exp(j*k_Z_Rectangular_Offset_Grid(ix,iy)*z0);
                f_Z_z0_Rectangular_Offset(ix,iy)=-(f_X_z0_Rectangular_Offset(ix,iy)*k_X_Rectangular_Offset_Grid(ix,iy)+f_Y_z0_Rectangular_Offset(ix,iy)*k_Y_Rectangular_Offset_Grid(ix,iy))/k_Z_Rectangular_Offset_Grid(ix,iy);
            else
                f_X_z0_Rectangular_Offset(ix,iy)=0;
                f_Y_z0_Rectangular_Offset(ix,iy)=0;
                f_Z_z0_Rectangular_Offset(ix,iy)=0;
            end 
        end
    end

    % Hologram is results of trasnformign back to the near-field from the
    % Fourier space
    
    Hologram_X_Complex_Zero_Padded=fft2(ifftshift(f_X_z0_Rectangular));
    Hologram_Y_Complex_Zero_Padded=fft2(ifftshift(f_Y_z0_Rectangular));
    Hologram_Z_Complex_Zero_Padded=fft2(ifftshift(f_Z_z0_Rectangular));
    
    Hologram_X_Complex=Hologram_X_Complex_Zero_Padded(1:M,1:N);
    Hologram_Y_Complex=Hologram_Y_Complex_Zero_Padded(1:M,1:N);
    Hologram_Z_Complex=Hologram_Z_Complex_Zero_Padded(1:M,1:N);
        
    Hologram_X_Magnitude=20*log10(abs(Hologram_X_Complex));
    Hologram_Y_Magnitude=20*log10(abs(Hologram_Y_Complex));
    Hologram_Z_Magnitude=20*log10(abs(Hologram_Z_Complex));

    % hologram actual with offset--same as above section, just adding
    % offset
    Hologram_X_Complex_Offset_Zero_Padded=fft2(ifftshift(f_X_z0_Rectangular_Offset));
    Hologram_Y_Complex_Offset_Zero_Padded=fft2(ifftshift(f_Y_z0_Rectangular_Offset));
    Hologram_Z_Complex_Offset_Zero_Padded=fft2(ifftshift(f_Z_z0_Rectangular_Offset));
    
    Hologram_X_Complex_Offset=Hologram_X_Complex_Offset_Zero_Padded(1:M,1:N);
    Hologram_Y_Complex_Offset=Hologram_Y_Complex_Offset_Zero_Padded(1:M,1:N);
    Hologram_Z_Complex_Offset=Hologram_Z_Complex_Offset_Zero_Padded(1:M,1:N);
        
    Hologram_X_Offset_Magnitude=20*log10(abs(Hologram_X_Complex_Offset));
    Hologram_Y_Offset_Magnitude=20*log10(abs(Hologram_Y_Complex_Offset));
    Hologram_Z_Offset_Magnitude=20*log10(abs(Hologram_Z_Complex_Offset));

    
    Hologram_Slots_X_Magnitude(Index,:)=interp2(x,y,Hologram_X_Magnitude',[0 0 0 0 0 0 0 0],0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],'spline');
    Hologram_Slots_Y_Magnitude(Index,:)=interp2(x,y,Hologram_Y_Magnitude',[0 0 0 0 0 0 0 0],0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],'spline');
    Hologram_Slots_Z_Magnitude(Index,:)=interp2(x,y,Hologram_Z_Magnitude',[0 0 0 0 0 0 0 0],0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],'spline');
    
    % my offsetted version
    Hologram_Slots_X_Offset_Magnitude(Index,:)=interp2(x_offset,y,Hologram_X_Magnitude',[0 0 0 0 0 0 0 0],0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],'spline');
    Hologram_Slots_Y_Offset_Magnitude(Index,:)=interp2(x_offset,y,Hologram_Y_Magnitude',[0 0 0 0 0 0 0 0],0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],'spline');
    Hologram_Slots_Z_Offset_Magnitude(Index,:)=interp2(x_offset,y,Hologram_Z_Magnitude',[0 0 0 0 0 0 0 0],0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],'spline');

    figure;
    subplot(3,2,1)
    surf(x*1000,y*1000,Hologram_X_Magnitude');
    title(sprintf('f = %f GHz (z = 0 mm, Hologram)',f/1000000000));
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{x}| (dB)');
    set(gca,'XLim',[min(x)*1000 max(x)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,3);
    surf(x*1000,y*1000,Hologram_Y_Magnitude');
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{y}| (dB)');
    set(gca,'XLim',[min(x)*1000 max(x)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,5);
    surf(x*1000,y*1000,Hologram_Z_Magnitude');
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{z}| (dB)');
    set(gca,'XLim',[min(x)*1000 max(x)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;

    % plotting with the offset
    subplot(3,2,2)
    surf(x_offset*1000,y*1000,Hologram_X_Offset_Magnitude');
    title(sprintf('f = %f GHz (z = 0 mm, Hologram)--wi',f/1000000000));
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{x}| (dB)');
    set(gca,'XLim',[min(x_offset)*1000 max(x_offset)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,4);
    surf(x_offset*1000,y*1000,Hologram_Y_Offset_Magnitude');
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{y}| (dB)');
    set(gca,'XLim',[min(x_offset)*1000 max(x_offset)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,6);
    surf(x_offset*1000,y*1000,Hologram_Z_Offset_Magnitude');
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{z}| (dB)');
    set(gca,'XLim',[min(x_offset)*1000 max(x_offset)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(-37.5,30);
    shading flat;
    colorbar;
    %print(gcf,'-dtiff',['NF_dB_' num2str(f) '_GHz']);
    clear Hologram_X_Magnitude Hologram_Y_Magnitude Hologram_Z_Magnitude Hologram_X_Offset_Magnitude Hologram_Y_Offset_Magnitude Hologram_Z_Offset_Magnitude;

%% random hologram phase stuff

%     try
%         
%         [Hologram_X_Phase] = GoldsteinUnwrap2D(Hologram_X_Complex);
%         [Hologram_Y_Phase] = GoldsteinUnwrap2D(Hologram_Y_Complex);
%         [Hologram_Z_Phase] = GoldsteinUnwrap2D(Hologram_Z_Complex);
%         Hologram_Slots_X_Phase(Index,:) = interp2(x,y,Hologram_X_Phase',0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],[0 0 0 0 0 0 0 0],'spline');
%         Hologram_Slots_Y_Phase(Index,:) = interp2(x,y,Hologram_Y_Phase',0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],[0 0 0 0 0 0 0 0],'spline');
%         Hologram_Slots_Z_Phase(Index,:) = interp2(x,y,Hologram_Z_Phase',0.0254*[-0.28 -0.20 -0.12 -0.04 0.04 0.12 0.20 0.28],[0 0 0 0 0 0 0 0],'spline');
% 
%         figure;
%         subplot(2,2,1)
%         surf(x*1000,y*1000,Hologram_X_Phase');
%         title(sprintf('f = %f GHz (z = 0 mm, Hologram)',f/1000000000));
%         xlabel('x (mm)');
%         ylabel('y (mm)');
%         zlabel('\angle E_{x} (rad)');
%         set(gca,'XLim',[min(x)*1000 max(x)*1000]);
%         set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%         view(-37.5,30);
%         shading flat;
%         colorbar;
%         subplot(2,2,2);
%         imagesc(Hologram_X_Phase');
%         colorbar;
%         subplot(2,2,3);
%         surf(x*1000,y*1000,Hologram_Y_Phase');
%         xlabel('x (mm)');
%         ylabel('y (mm)');
%         zlabel('\angle E_{y} (rad)');
%         set(gca,'XLim',[min(x)*1000 max(x)*1000]);
%         set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%         view(-37.5,30);
%         shading flat;
%         colorbar;
%         subplot(2,2,4);
%         imagesc(Hologram_Y_Phase');
%         colorbar;
%         %print(gcf,'-dtiff',['Hologram_rad_' num2str(f) '_GHz']);
%         clear Hologram_X_Phase Hologram_Y_Phase;
%         
%     end

%% 
    
    f_X_Spherical=interp2(k_X_Rectangular,k_Y_Rectangular,abs(f_X_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Y_Spherical=interp2(k_X_Rectangular,k_Y_Rectangular,abs(f_Y_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Z_Spherical=interp2(k_X_Rectangular,k_Y_Rectangular,abs(f_Z_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    
    f_X_Spherical_Offset=interp2(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,abs(f_X_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Y_Spherical_Offset=interp2(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,abs(f_Y_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Z_Spherical_Offset=interp2(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,abs(f_Z_Rectangular_Offset'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');

    r=10000;
    C=j*(k0*exp(-j*k0*r))/(2*pi*r);
    Etheta=C*(f_X_Spherical.*cos(phi)+f_Y_Spherical.*sin(phi));
    Ephi=C*cos(theta).*(-f_X_Spherical.*sin(phi)+f_Y_Spherical.*cos(phi));
    %[Etheta,Ephi]=ProbeCorrection(Etheta,Ephi,theta,phi,f);
    W=1/(2*120*pi).*(Etheta.*conj(Etheta)+Ephi.*conj(Ephi));

	U = (abs(Etheta).^2 + abs(Ephi).^2);

    % offset version of etheta, ephi calculations
    Etheta_Offset=C*(f_X_Spherical_Offset.*cos(phi)+f_Y_Spherical_Offset.*sin(phi));
    Ephi_Offset=C*cos(theta).*(-f_X_Spherical_Offset.*sin(phi)+f_Y_Spherical_Offset.*cos(phi));
    %[Etheta,Ephi]=ProbeCorrection(Etheta,Ephi,theta,phi,f);
    W_Offset=1/(2*120*pi).*(Etheta_Offset.*conj(Etheta_Offset)+Ephi_Offset.*conj(Ephi_Offset));

	U_Offset = (abs(Etheta_Offset).^2 + abs(Ephi_Offset).^2);
    
	% Calculation of radiated power through numerical integration
	e_theta = [1 4 repmat([2 4], 1, floor(length(theta(1,:))/2) - 1) 1];
    e_phi = [1 4 repmat([2 4], 1, floor(length(phi(:,1))/2) - 1) 1];
    P = dphi*dtheta*sum(sum(U.*(e_theta'*e_phi).*abs(sin(theta))))/9;
	
    D = 4*pi*U/P;
    U_Co = (abs(Etheta.*cos(phi)-Ephi.*sin(phi))).^2;
    U_Cross = (abs(Etheta.*sin(phi)+Ephi.*cos(phi))).^2;
    D_Co = 4*pi*U_Co/P;
    D_Cross = 4*pi*U_Cross/P;
	
%% plots

    % weird Ludwig co-pol stuff
	figure;
	x_sph=sin(theta).*cos(phi);
	y_sph=sin(theta).*sin(phi);
	z_sph=cos(theta);
	subplot(1,2,1)
	h = polar([0 2*pi], [0 1]);
	delete(h)
	hold on
	surf(x_sph,y_sph,z_sph,D_Co,'EdgeAlpha',0)
	title('Ludwig-3 Co-Pol. Dir. [dB]')
	caxis([-30 20])
	colorbar
	subplot(1,2,2)
	h = polar([0 2*pi], [0 1]);
	delete(h)
	hold on
	surf(x_sph,y_sph,z_sph,D_Cross,'EdgeAlpha',0)
	title('Ludwig-3 Cross-Pol. Dir. [dB]')
	caxis([-30 20])
	colorbar
    
    % plotting fx, fy, fz in terms of theta and phi
    figure;
    subplot(3,2,1);
    surf(theta,phi,20*log10(abs(f_X_Spherical)));
    title(sprintf('f = %f GHz',f/1000000000));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|f_{x}| (dB)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(f_X_Spherical)))) max(max(20*log10(abs(f_X_Spherical))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,3);
    surf(theta,phi,20*log10(abs(f_Y_Spherical)));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|f_{y}| (dB)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(f_Y_Spherical)))) max(max(20*log10(abs(f_Y_Spherical))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,5);
    surf(theta,phi,20*log10(abs(f_Z_Spherical)));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|f_{z}| (dB)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(f_Z_Spherical)))) max(max(20*log10(abs(f_Z_Spherical))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    % adding the offset stuff to this figure
    subplot(3,2,2);
    surf(theta,phi,20*log10(abs(f_X_Spherical_Offset)));
    title(sprintf('f = %f GHz',f/1000000000));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|f_{x}| (dB)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(f_X_Spherical_Offset)))) max(max(20*log10(abs(f_X_Spherical_Offset))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,4);
    surf(theta,phi,20*log10(abs(f_Y_Spherical_Offset)));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|f_{y}| (dB)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(f_Y_Spherical_Offset)))) max(max(20*log10(abs(f_Y_Spherical_Offset))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,6);
    surf(theta,phi,20*log10(abs(f_Z_Spherical_Offset)));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|f_{z}| (dB)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(f_Z_Spherical_Offset)))) max(max(20*log10(abs(f_Z_Spherical_Offset))))]);
    view(-37.5,30);
    shading flat;
    colorbar;

    % Etheta and Ephi as functions of phi and theta
    figure;
    subplot(2,2,1);
    surf(theta,phi,20*log10(abs(Etheta)));
    title(sprintf('f = %f GHz',f/1000000000));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|E_{\theta}| (dBi)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Etheta)))) max(max(20*log10(abs(Etheta))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(2,2,3);
    surf(theta,phi,20*log10(abs(Ephi)));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|E_{\phi}| (dBi)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Ephi)))) max(max(20*log10(abs(Ephi))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    %adding offset stuff to this figure
    subplot(2,2,2);
    surf(theta,phi,20*log10(abs(Etheta_Offset)));
    title(sprintf('f = %f GHz',f/1000000000));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|E_{\theta}| (dBi)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Etheta_Offset)))) max(max(20*log10(abs(Etheta_Offset))))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(2,2,4);
    surf(theta,phi,20*log10(abs(Ephi_Offset)));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|E_{\phi}| (dBi)');
    axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Ephi_Offset)))) max(max(20*log10(abs(Ephi_Offset))))]);
    view(-37.5,30);
    shading flat;
    colorbar;

    % 3D rendering of antenna pattern
    figure;
    subplot(2,2,1);
    sphere3d(20*log10(abs(Etheta))-max(max(20*log10(abs(Etheta')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
    title('|E_{\theta}| (dBi)');
    subplot(2,2,2);
    sphere3d(20*log10(abs(Ephi))-max(max(20*log10(abs(Ephi')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
    title('|E_{\phi}| (dBi)');
    % offset portion
    subplot(2,2,3);
    sphere3d(20*log10(abs(Etheta_Offset))-max(max(20*log10(abs(Etheta_Offset')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
    title('|E_{\theta}| (dBi) with offset');
    subplot(2,2,4);
    sphere3d(20*log10(abs(Ephi_Offset))-max(max(20*log10(abs(Ephi_Offset')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
    title('|E_{\phi}| (dBi) with offset');

    % W over theta and phi
    figure;
    subplot(1,2,1);
    surf(theta,phi,10*log10(W));
    title(sprintf('f = %f GHz',f/1000000000));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|W| (dBi)');
    axis([-pi/2 pi/2 0 pi min(min(10*log10(W))) max(max(10*log10(W)))]);
    view(-37.5,30);
    shading flat;
    colorbar;
    % plotting W with offset
    subplot(1,2,2);
    surf(theta,phi,10*log10(W_Offset));
    title(sprintf('f = %f GHz--with offset',f/1000000000));
    xlabel('\theta (rad)');
    ylabel('\phi (rad)');
    zlabel('|W| (dBi)');
    axis([-pi/2 pi/2 0 pi min(min(10*log10(W_Offset))) max(max(10*log10(W_Offset)))]);
    view(-37.5,30);
    shading flat;
    colorbar;

    W_Size=size(W);
    EPLANE(Index,:)=10*log10(W(floor(W_Size(2)/2),:))-max(10*log10(W(floor(W_Size(2)/2),:)));
    HPLANE(Index,:)=10*log10(W(1,:))-max(10*log10(W(1,:)));

    W_Offset_Size=size(W_Offset);
    EPLANE_Offset(Index,:)=10*log10(W_Offset(floor(W_Offset_Size(2)/2),:))-max(10*log10(W_Offset(floor(W_Offset_Size(2)/2),:)));
    HPLANE_Offset(Index,:)=10*log10(W_Offset(1,:))-max(10*log10(W_Offset(1,:)));

    Index=Index+1;

end

% E-plane and H-plane plots
figure;
subplot(2,2,1);
plot(180/pi*theta(1,:),EPLANE,[-25 -25],[-30 0],'k',[25 25],[-30 0],'k',[-90 90],[-13.6 -13.6],'k');
title('No Offset: E-Plane');
xlabel('LOAD \leftarrow \theta (Deg) \rightarrow INPUT');
ylabel('Directivity (dBi)');
set(gca,'XLim',[-90 90]);
set(gca,'YLim',[-30 0]);
subplot(2,2,2);
plot(180/pi*theta(1,:),HPLANE);
title('No Offset: H-Plane');
xlabel('\theta (Deg)');
ylabel('Directivity (dBi)');
set(gca,'XLim',[-90 90]);
set(gca,'YLim',[-30 0]);
% with offset
subplot(2,2,3);
plot(180/pi*theta(1,:),EPLANE_Offset,[-25 -25],[-30 0],'k',[25 25],[-30 0],'k',[-90 90],[-13.6 -13.6],'k');
title('With Offset: E-Plane');
xlabel('LOAD \leftarrow \theta (Deg) \rightarrow INPUT');
ylabel('Directivity (dBi)');
set(gca,'XLim',[-90 90]);
set(gca,'YLim',[-30 0]);
subplot(2,2,4);
plot(180/pi*theta(1,:),HPLANE_Offset);
title('With Offset: H-Plane');
xlabel('\theta (Deg)');
ylabel('Directivity (dBi)');
set(gca,'XLim',[-90 90]);
set(gca,'YLim',[-30 0]);

%save RESULTS EPLANE HPLANE Hologram_Slots_X_Magnitude Hologram_Slots_Y_Magnitude Hologram_Slots_X_Phase Hologram_Slots_Y_Phase freq

function plot_frequency_spectra_time_series(freq,sdata)

c=299792458; % Speed of light in vacuum [m/s]
N=length(freq);
f_start=freq(1); % first frequency
f_stop=freq(N);  % last frequency
df=(f_stop-f_start)/(N-1); 
dt=1/(N*df);
t=(0:N-1)*dt;
x=c*t;

% S21 in frequency domain
figure;
subplot(2,1,1);
plot(freq/1e9,20*log10(abs(sdata.s21{floor(sdata.ypoints/2),floor(sdata.xpoints/2)})),'k');
xlabel('Frequency (GHz)');
ylabel('S_{21} (dB)');
subplot(2,1,2);
plot(freq/1e9,180/pi*angle(sdata.s21{floor(sdata.ypoints/2),floor(sdata.xpoints/2)}),'k');
xlabel('Frequency (GHz)');
ylabel('\angle S_{21} (�)');

% S21 in time domain--plots real components of inverse Fourier transform of
% electric field data as function of frequency
figure;
subplot(2,1,1);
plot(t'*1e9,20*log10(real(ifft(squeeze(sdata.s21{floor(sdata.ypoints/2),floor(sdata.xpoints/2)})))),'k');
xlabel('Time (ns)');
ylabel('S_{21} (dB)');
subplot(2,1,2);
plot(x',20*log10(real(ifft(squeeze(sdata.s21{floor(sdata.ypoints/2),floor(sdata.xpoints/2)})))),'k');
xlabel('Distance (m)');
ylabel('S_{21} (dB)');

end

%% functions

function [fx,fy,fz,fx_magnitude_db,fy_magnitude_db,fz_magnitude_db]=NFtoFourierSpace(NF_x, NF_y,kx_grid,ky_grid,kz_grid, MI, NI)
    % See equations (16-7a) and (16-7b) in Balanis
    fx=ifftshift(ifft2(NF_x,MI,NI)); % does the inverse FFT over an MI x NI grid
    fy=ifftshift(ifft2(NF_y,MI,NI));
    fz=-(fx.*kx_grid+fy.*ky_grid)./kz_grid;
    % achieved by taking into account kx, ky, fx, fy

    % power in dB in Fourier space
    fx_magnitude_db=20*log10(abs(fx));
    fy_magnitude_db=20*log10(abs(fy));
    fz_magnitude_db=20*log10(abs(fz));
end

function NFtoFourierSpace_struct(sdata)
% See equations (16-7a) and (16-7b) in Balanis
    sdata.f_X_Rectangular=ifftshift(ifft2(sdata.NF_X_Complex,sdata.xpoints*4,sdata.ypoints*4)); % does the inverse FFT over an MI x NI grid
    sdata.f_Y_Rectangular=ifftshift(ifft2(sdata.NF_Y_Complex,sdata.xpoints*4,sdata.ypoints*4));
    sdata.f_Z_Rectangular=-(sdata.f_X_Rectangular.*sdata.k_X_Rectangular_Grid+sdata.f_Y_Rectangular.*sdata.k_Y_Rectangular_Grid)./sdata.k_X_Rectangular_Grid;
    % achieved by taking into account kx, ky, fx, fy

    % power in dB in Fourier space
    sdata.f_X_Rectangular_Magnitude=20*log10(abs(sdata.f_X_Rectangular));
    sdata.f_Y_Rectangular_Magnitude=20*log10(abs(sdata.f_Y_Rectangular));
    sdata.f_Z_Rectangular_Magnitude=20*log10(abs(sdata.f_Z_Rectangular));
end
