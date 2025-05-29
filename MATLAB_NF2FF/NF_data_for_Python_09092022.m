%% Continuing with jitter with no iteration and experimenting to find the best way to calculate that percent error

% I'm updating this script with the most recent method for calculating
% "percent error," which is calculating the change in fractional solid
% angle. This will also have the most up-to-date version of the
% complex_errors function, which I will copy and paste into other scripts.
% To-do in this script: exclude values outside of the "main beam" for this
% dataset, and only use those values to calculate the solid angle. 

% Here, we use x in columns, y in rows to interpolate and investigate the
% results of jitter on the final electric field patterns. The jitter is
% created by generating a jittered x and y dataset (using random jitter 
% within a certain threshold), and interpolating the data over the orignal
% grid to obtain NF values at the jittered sample locations. Then this data
% is used and assigned to the original spots in the grid, to replicate what
% it's going to look like if the location is off, but we use the original
% gridding.

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

% my interested variables
c=299792458; % Speed of light in vacuum [m/s]
broadcast_freq=94*10^9; % broadcast frequency in hertz

%% creating our gridding

% See equations (16-10a) and (16-10b) in Balanis
M=sdata.xpoints; % Amount of samples in the x direction (along table, left to right)
N=sdata.ypoints; % Amount of samples in the y direction (across table, front to back)
dx=sdata.x_step/1000; % Sample spacing in the x direction [m]--I assume spacing is then in mm in data structure
dy=sdata.y_step/1000; % Sample spacing in the y direction [m]

% See equations (16-10a) and (16-10b) in Balanis
a=dx*(M-1); % The length of the scanned area in the x direction [m]
b=dy*(N-1); % The length of the scanned area in the y direction [m]
x=[-a/2:a/(M-1):a/2]; % arrays of relevant x- and y-values created by spacing by dx/dy
y=[-b/2:b/(N-1):b/2]; % between -a/2 and a/2, -b/2 and b/2
z0=0.006;

% See equations (16-13a) and (16-13b) in Balanis
% Zero padding is used to increase the resolution of the plane wave spectral domain.
MI=4*M;%2^(ceil(log2(M))+1);
NI=4*N;%2^(ceil(log2(N))+1);
m=[-MI/2:1:MI/2-1];
n=[-NI/2:1:NI/2-1];
% defining k-vectors and their components, and making that into a grid for
% kX and kY
k_X_Rectangular=2.*pi.*m/(MI*dx);
k_Y_Rectangular=2.*pi.*n/(NI*dy);
[k_X_Rectangular_Grid,k_Y_Rectangular_Grid] = meshgrid(k_X_Rectangular,k_Y_Rectangular);

% creating theta and phi grids from -pi/2 to pi/2 and 0 to pi
% (respectively)
dtheta=0.05;
dphi=0.05;
theta=[-pi/2+dtheta:dtheta:pi/2-dtheta];
phi=[0+dphi:dphi:pi-dphi];
theta_vector=theta;
phi_vector=phi;
[theta,phi]=meshgrid(theta,phi);

% creating a jittered x, y grid for interpolation
lambda_m=c/broadcast_freq;
nyquist_sampling_m=lambda_m/2; % necessary grid spacing in m
jitter_percentage_threshold=0.10; % the max jitter in x and y as percentage of Nyquist spacing (for now)
jitter_threshold=nyquist_sampling_m*jitter_percentage_threshold;
jitter_x=-jitter_threshold+2.*jitter_threshold.*rand(1,M);
jitter_y=-jitter_threshold+2.*jitter_threshold.*rand(1,N);
x_jittered=x+jitter_x;
y_jittered=y+jitter_y;
[x_jittered_grid,y_jittered_grid]=meshgrid(x_jittered,y_jittered);

% % a check for our jitter
% max(abs(jitter_x))
% max(abs(jitter_y))
% jitter_threshold

%% calculations

Index = 1;
for f_Index = 201:1:201 %1:1:N Only does this at the 201st frequency?? why--maybe only interested in one frequency and you can adjust it
    
    f=sdata.freq(f_Index);
    
    %% loading in and plotting all NF data at given frequency

    lambda0=c/f;
    k0=2*pi/lambda0;
    k_Z_Rectangular_Grid = sqrt(k0^2-k_X_Rectangular_Grid.^2-k_Y_Rectangular_Grid.^2);

    for iy=1:1:N
        for ix=1:1:M
            NF_X_Complex(iy,ix)=sdata.s21{iy,ix}(f_Index);
            NF_Y_Complex(iy,ix)=sdata2.s21{iy,ix}(f_Index);
        end
    end

    % interpolating near-field data to obtain the jittered version of
    % near-field data
    NF_X_Complex_jittered=interp2(x,y,NF_X_Complex,x_jittered_grid,y_jittered_grid,'spline');
    NF_Y_Complex_jittered=interp2(x,y,NF_Y_Complex,x_jittered_grid,y_jittered_grid,'spline');

    NF_X_Magnitude = 20*log10(abs(NF_X_Complex));
    NF_Y_Magnitude = 20*log10(abs(NF_Y_Complex));

    NF_X_jittered_Magnitude=20*log10(abs(NF_X_Complex_jittered));
    NF_Y_jittered_Magnitude=20*log10(abs(NF_Y_Complex_jittered));

    % plotting NF Data
    figure;
    subplot(2,2,1)
    surf(x*1000,y*1000,NF_X_Magnitude);
    caxis([-Inf -30]);
    daspect([1 1 1]);
    title(sprintf('f = %f GHz (z = %i mm)--normal NF data',f/1000000000,z0*1000));
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{x}| (dB)');
    set(gca,'XLim',[min(x)*1000 max(x)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(0,90);
    shading flat;
    colorbar;
    subplot(2,2,2);
    surf(x*1000,y*1000,NF_Y_Magnitude);
    caxis([-Inf -30]);
    daspect([1 1 1]);
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{y}| (dB)');
    set(gca,'XLim',[min(x)*1000 max(x)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(0,90);
    shading flat;
    colorbar;
    %print(gcf,'-dtiff',['NF_dB_' num2str(f) '_GHz']);
    
    % adding offset plot
    subplot(2,2,3)
    surf(x*1000,y*1000,NF_X_jittered_Magnitude);
    caxis([-Inf -30]);
    daspect([1 1 1]);
    title(sprintf('Plotting interpolated jittered values on the original grid',f/1000000000,z0*1000));
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{x}| (dB)');
    set(gca,'XLim',[min(x)*1000 max(x)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(0,90);
    shading flat;
    colorbar;
    subplot(2,2,4);
    surf(x*1000,y*1000,NF_Y_jittered_Magnitude);
    caxis([-Inf -30]);
    daspect([1 1 1]);
    title('Plotting interpolated data at each jittered spot')
    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('|E_{y}| (dB)');
    set(gca,'XLim',[min(x)*1000 max(x)*1000]);
    set(gca,'YLim',[min(y)*1000 max(y)*1000]);
    view(0,90);
    shading flat;
    colorbar;

    figure;
    pcolor(x,y,angle(NF_X_Complex))

    clear NF_X_Magnitude NF_Y_Magnitude NF_X_jittered_Magnitude NF_Y_jittered_Magnitude;

%% exporting the data to CSV

writematrix(NF_X_Complex,'NF_X_Complex.csv');
writematrix(NF_Y_Complex,'NF_Y_Complex.csv');
writematrix(NF_X_Complex_jittered,'NF_X_Complex_jittered.csv');
writematrix(NF_Y_Complex_jittered,'NF_Y_Complex_jittered.csv');

end