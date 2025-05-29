%% beginning of drift experiment--drifting the grid by some amount and looking at the resulting errors

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

% creating x, y grid for interpolation that drifts
lambda_m=c/broadcast_freq;
nyquist_sampling_m=lambda_m/2; % necessary grid spacing in m
max_drift_percents=linspace(0,0.7,71);
% other_percents=[0.6353,0.658,0.570358,0.429384];
% max_drift_percents=cat(2,max_drift_percents,other_percents);
% max_drift_percents=sort(max_drift_percents);
max_drifts=nyquist_sampling_m.*max_drift_percents;


%% calculations

Index = 1;
for f_Index = 201:1:201 %1:1:N Only does this at the 201st frequency?? why--maybe only interested in one frequency and you can adjust it
    
    f=sdata.freq(f_Index);
    
    %% doing the transformation for the control beam case

    lambda0=c/f;
    k0=2*pi/lambda0;
    k_Z_Rectangular_Grid = sqrt(k0^2-k_X_Rectangular_Grid.^2-k_Y_Rectangular_Grid.^2);

    % read in data stuff
    for iy=1:1:N
        for ix=1:1:M
            NF_X_Complex(iy,ix)=sdata.s21{iy,ix}(f_Index);
            NF_Y_Complex(iy,ix)=sdata2.s21{iy,ix}(f_Index);
        end
    end

    % transform to far field
    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,f_X_Rectangular_Magnitude,...
        f_Y_Rectangular_Magnitude,f_Z_Rectangular_Magnitude]=NFtoFourierSpace...
        (NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Grid,k_Y_Rectangular_Grid,...
        k_Z_Rectangular_Grid);
    [Etheta,Ephi]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,...
        theta,phi,k_X_Rectangular,k_Y_Rectangular,k0);

    % for loop for each drift case
    for drift_index=1:length(max_drifts)

        % creating the offset stuff
        x_offsets=linspace(0,max_drifts(drift_index),M);
        y_offsets=linspace(0,max_drifts(drift_index),N);
        x_drifted=x+x_offsets;
        y_drifted=y+y_offsets;
        [x_drifted_grid,y_drifted_grid]=meshgrid(x_drifted,y_drifted);

        % interpolating near-field data to obtain the drifted version of
        % near-field data
        NF_X_drifted_Complex=interp2(x,y,NF_X_Complex,x_drifted_grid,y_drifted_grid,'spline');
        NF_Y_drifted_Complex=interp2(x,y,NF_Y_Complex,x_drifted_grid,y_drifted_grid,'spline');

        %% transforming to far field

        [f_X_Rectangular_drifted,f_Y_Rectangular_drifted,f_Z_Rectangular_drifted,...
            f_X_Rectangular_drifted_Magnitude,f_Y_Rectangular_drifted_Magnitude,...
            f_Z_Rectangular_drifted_Magnitude]=NFtoFourierSpace(NF_X_drifted_Complex,...
            NF_Y_drifted_Complex,k_X_Rectangular_Grid,k_Y_Rectangular_Grid,...
            k_Z_Rectangular_Grid);
        [Etheta_drifted,Ephi_drifted]=FouriertoFF(f_X_Rectangular_drifted,f_Y_Rectangular_drifted,...
            f_Z_Rectangular_drifted,theta,phi,k_X_Rectangular,k_Y_Rectangular,k0);

        %% running error metrics
        Etheta_error_metrics(drift_index)=electric_field_error(Etheta,Etheta_drifted,theta_vector,phi_vector,'theta','phi',0);
        Ephi_error_metrics(drift_index)=electric_field_error(Ephi,Ephi_drifted,theta_vector,phi_vector,'theta','phi',0);

    end

    % side by side heatmaps of weighted % difference in amplitude for all
    % the offsets, E_theta and E_phi
    figure;
    for drift_index=1:3
        subplot(2,3,drift_index);
        plot1=pcolor(theta,phi,Etheta_error_metrics(drift_index).weighted_percent_difference_amplitude);
        daspect([1 1 1]);
        plot1.LineWidth=0.5;
        caxis([-Inf 1]);
        title('Weighted percent difference in E_\theta: amplitude');
        xlabel('Theta');
        ylabel('Phi');
        colorbar;
    end
    for drift_index=1:3
        subplot(2,3,drift_index+3);
        plot1=pcolor(theta,phi,Ephi_error_metrics(drift_index).weighted_percent_difference_amplitude);
        daspect([1 1 1]);
        plot1.LineWidth=0.5;
        caxis([-Inf 1]);
        title('Weighted percent difference in E_\phi: amplitude');
        xlabel('Theta');
        ylabel('Phi');
        colorbar;
    end
	
    % scatter plot of average percent difference in amplitude vs percentage
    % drift
    for index=1:length(max_drifts)
        mean_pct_diff_amplitudes_Etheta(index)=Etheta_error_metrics(index).weighted_percent_difference_mean_amplitude;
        mean_pct_diff_amplitudes_Ephi(index)=Ephi_error_metrics(index).weighted_percent_difference_mean_amplitude;
    end
    figure;
    semilogx(max_drift_percents,mean_pct_diff_amplitudes_Etheta,'b.','MarkerSize',15);
    hold on;
    semilogx(max_drift_percents,mean_pct_diff_amplitudes_Ephi,'r.','MarkerSize',15);
    title('Mean percent difference vs drift');
    xlabel('Max drift as percentage of Nyquist spacing');
    ylabel('Mean percent difference between offset and original field');
    legend('E_\theta','E_\phi');

    % scatter plot of change in solid angle vs percentage drift
    for drift_index=1:length(max_drifts)
        solid_angle_changes_Etheta(drift_index)=Etheta_error_metrics(drift_index).solid_angle_error_fraction_excluded;
        solid_angle_changes_Ephi(drift_index)=Ephi_error_metrics(drift_index).solid_angle_error_fraction_excluded;
    end
    figure;
    plot(max_drift_percents,solid_angle_changes_Etheta,'b.','MarkerSize',15);
    hold on;
    semilogx(max_drift_percents,solid_angle_changes_Ephi,'r.','MarkerSize',15);
    title('Solid angle error vs drift');
    xlabel('Max drift as percentage of Nyquist spacing');
    ylabel('Fractional solid angle error');

    % fitting to the plot
    Etheta_solid_angle_fits=polyfit(solid_angle_changes_Etheta,max_drift_percents,1);
    Etheta_plot_solid_angle_fits=polyfit(max_drift_percents,solid_angle_changes_Etheta,1);
    Ephi_solid_angle_fits=polyfit(solid_angle_changes_Ephi,max_drift_percents,1);
    Ephi_plot_solid_angle_fits=polyfit(max_drift_percents,solid_angle_changes_Ephi,1);

    % plotting the lines
    x_values=linspace(0,0.2,100);
    plot(x_values,polyval(Etheta_plot_solid_angle_fits,x_values),'-b');
    plot(x_values,polyval(Ephi_plot_solid_angle_fits,x_values),'-r');

    % finding the cutoff
    Etheta_solid_angle_max_drift=polyval(Etheta_solid_angle_fits,0.018435);
    Ephi_solid_angle_max_drift=polyval(Ephi_solid_angle_fits,0.018435);

%     xline(Etheta_solid_angle_max_drift,'--k');
%     xline(Ephi_solid_angle_max_drift,'--k');
%     yline(0.018435,'--k');

    legend('E_\theta','E_\phi','','','','','','');

    fprintf('Max percentage drift for E_theta: %f\n',Etheta_solid_angle_max_drift);
    fprintf('Max drift percentage for E_phi: %f\n',Ephi_solid_angle_max_drift);


%% plots

%     % plotting in Fourier space
%     fourier_fig=figure;
%     title(sprintf('f = %f GHz',f/1000000000));
%     subplot(3,2,1);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_X_Rectangular_Magnitude);
%     caxis([-Inf -80]);
%     daspect([1 1 1]);
%     title('Original gridding and data')
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{x}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(0,90);
%     shading flat;
%     colorbar;
%     subplot(3,2,3);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_Y_Rectangular_Magnitude);
%     caxis([-Inf -80]);
%     daspect([1 1 1]);
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{y}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(0,90);
%     shading flat;
%     colorbar;
%     subplot(3,2,5);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_Z_Rectangular_Magnitude);
%     caxis([-Inf -80]);
%     daspect([1 1 1]);
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{z}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(0,90);
%     shading flat;
%     colorbar;
% 
% %     plotting offset version of this
%     subplot(3,2,2);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_X_Rectangular_drifted_Magnitude);
%     caxis([-Inf -80]);
%     daspect([1 1 1]);
%     title(sprintf('Plotting f_x, f_y, f_z over same k_x, k_y grid'));
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{x}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(0,90);
%     shading flat;
%     colorbar;
%     subplot(3,2,4);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_Y_Rectangular_drifted_Magnitude);
%     caxis([-Inf -80]);
%     daspect([1 1 1]);
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{y}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(0,90);
%     shading flat;
%     colorbar;
%     ax=subplot(3,2,6);
%     surf(k_X_Rectangular,k_Y_Rectangular,f_Z_Rectangular_drifted_Magnitude);
%     caxis([-Inf -80]);
%     daspect([1 1 1]);
%     xlabel(sprintf('k_{x} (m^{-1})'));
%     ylabel(sprintf('k_{y} (m^{-1})'));
%     zlabel('|f_{z}| (dB)');
%     set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
%     set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
%     view(0,90);
%     shading flat;
%     colorbar;

    % Etheta and Ephi as functions of phi and theta
%     figure;
%     subplot(2,2,1);
%     surf(theta,phi,20*log10(abs(Etheta)));
%     title(sprintf('f = %f GHz',f/1000000000));
%     xlabel('\theta (rad)');
%     ylabel('\phi (rad)');
%     zlabel('|E_{\theta}| (dBi)');
%     axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Etheta)))) max(max(20*log10(abs(Etheta))))]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(2,2,3);
%     surf(theta,phi,20*log10(abs(Ephi)));
%     xlabel('\theta (rad)');
%     ylabel('\phi (rad)');
%     zlabel('|E_{\phi}| (dBi)');
%     axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Ephi)))) max(max(20*log10(abs(Ephi))))]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     %adding offset stuff to this figure
%     subplot(2,2,2);
%     surf(theta,phi,20*log10(abs(Etheta_Offset)));
%     title(sprintf('f = %f GHz',f/1000000000));
%     xlabel('\theta (rad)');
%     ylabel('\phi (rad)');
%     zlabel('|E_{\theta}| (dBi)');
%     axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Etheta_Offset)))) max(max(20*log10(abs(Etheta_Offset))))]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(2,2,4);
%     surf(theta,phi,20*log10(abs(Ephi_Offset)));
%     xlabel('\theta (rad)');
%     ylabel('\phi (rad)');
%     zlabel('|E_{\phi}| (dBi)');
%     axis([-pi/2 pi/2 0 pi min(min(20*log10(abs(Ephi_Offset)))) max(max(20*log10(abs(Ephi_Offset))))]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
% 

    % an experiment with the spherical 3d thing
%     figure;
%     subplot(2,2,1);
%     sphere3d(20*log10(abs(Etheta)),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\theta} (dBi)');
%     subplot(2,2,2);
%     sphere3d(20*log10(abs(Ephi)),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\phi}|');
%     subplot(2,2,3);
%     sphere3d(20*log10(abs(Etheta_drifted)),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\theta}| with jitter');
%     subplot(2,2,4);
%     sphere3d(20*log10(abs(Ephi_drifted)),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\phi}| with jittered');

    % 3D rendering of antenna pattern
%     figure;
%     subplot(2,2,1);
%     sphere3d(20*log10(abs(Etheta))-max(max(20*log10(abs(Etheta')))),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\theta}| (dBi)');
%     subplot(2,2,2);
%     sphere3d(20*log10(abs(Ephi))-max(max(20*log10(abs(Ephi')))),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\phi}| (dBi)');
%     % offset portion
%     subplot(2,2,3);
%     sphere3d(20*log10(abs(Etheta_drifted))-max(max(20*log10(abs(Etheta_drifted')))),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\theta}| (dBi) with 0.5x sample density');
%     subplot(2,2,4);
%     sphere3d(20*log10(abs(Ephi_drifted))-max(max(20*log10(abs(Ephi_drifted')))),0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('|E_{\phi}| (dBi) with 0.5x sample density');

    Index=Index+1;

  
    

    %% plot of beams and errors for Laura

%     figure;
%     subplot(4,3,1);
%     pc=pcolor(theta,phi,20*log10(abs(Etheta)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\theta} control beam');
%     c=colorbar;
%     c.Label.String='E_{\theta} (dB)';
%     subplot(4,3,2);
%     pc=pcolor(theta,phi,20*log10(abs(Etheta_drifted)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\theta} perturbed beam');
%     c=colorbar;
%     c.Label.String='Perturbed E_{\theta} (dB)';
%     subplot(4,3,3);
%     pc=pcolor(theta,phi,20*log10(abs(Etheta_error_metrics.error)));
%     pc.EdgeColor='none';
%     daspect([1 1 1]);
%     xlabel('theta');
%     ylabel('phi');
%     title('Perturbed - control for E_{\theta}');
%     c=colorbar;
%     c.Label.String='Difference in E_{\theta} (dB)';
%     subplot(4,3,4);
%     pc=pcolor(theta,phi,20*log10(abs(Ephi)));
%     pc.EdgeColor='none';
%     daspect([1 1 1]);
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\phi} control beam');
%     c=colorbar;
%     c.Label.String='E_{\phi} (dB)';
%     subplot(4,3,5);
%     pc=pcolor(theta,phi,20*log10(abs(Ephi_drifted)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\phi} perturbed beam');
%     c=colorbar;
%     c.Label.String='E_{\phi} (dB)';
%     subplot(4,3,6);
%     pc=pcolor(theta,phi,20*log10(abs(Ephi_error_metrics.error)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('Perturbed - control for E_{\phi}');
%     c=colorbar;
%     c.Label.String='Difference in E_{\phi} (dB)';
%     subplot(4,3,7);
%     pc=pcolor(theta,phi,20*log10(abs(Etheta_excluded)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\theta} main beam');
%     c=colorbar;
%     c.Label.String='E_{\theta}';
%     subplot(4,3,8);
%     pc=pcolor(theta,phi,20*log10(abs(Etheta_perturbed_excluded)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\theta} perturbed main beam');
%     c=colorbar;
%     c.Label.String='E_{\theta}';
%     subplot(4,3,10);
%     pc=pcolor(theta,phi,20*log10(abs(Ephi_excluded)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\phi} main beam');
%     c=colorbar;
%     c.Label.String='E_{\phi}';
%     subplot(4,3,11);
%     pc=pcolor(theta,phi,20*log10(abs(Ephi_perturbed_excluded)));
%     daspect([1 1 1]);
%     pc.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\phi} perturbed main beam');
%     c=colorbar;
%     c.Label.String='E_{\phi}';


end


function plot_frequency_spectra_time_series(freq,sdata)

c=299792458; % Speed of light in vacuum [m/s]
N=length(freq);
f_start=freq(1); % first frequency
f_stop=freq(N);  % last frequency
df=(f_stop-f_start)/(N-1); 
dt=1/(N.*df);
t=(0:N-1).*dt;
x=c.*t;

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

function [fx,fy,fz,fx_magnitude_db,fy_magnitude_db,fz_magnitude_db]=NFtoFourierSpace(NF_x, NF_y,kx_grid,ky_grid,kz_grid)
    grid_size=size(kx_grid);
    MI=grid_size(1);
    NI=grid_size(2);

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

% complex error function
function error_metrics=complex_error(original_field,offset_field,x,y,label_1,label_2)

    % x and y are just vectors for plotting--could be phi and theta, kx and
    % ky, etc.
    [x_grid,y_grid]=meshgrid(x,y);

    error_metrics.error=offset_field-original_field;
    error_real=real(error_metrics.error);
    error_im=imag(error_metrics.error);
    original_field_real=real(original_field);
    original_field_im=imag(original_field);
    offset_field_real=real(offset_field);
    offset_field_im=imag(offset_field);
    error_amplitude=abs(error_metrics.error);
    error_phase=angle(error_metrics.error);


%     % doing it where I take amplitude/phase first, then subtract--per
%     % Will's request
%     original_field_phase=angle(original_field);
%     offset_field_phase=angle(offset_field);
%     original_field_amplitude=abs(original_field);
%     offset_field_amplitude=abs(offset_field);
%     error_phase=offset_field_phase-original_field_phase;
%     error_amplitude=offset_field_amplitude-original_field_amplitude;
% 
%     weighted_pct_error_amplitude=error_amplitude./((original_field_amplitude+offset_field_amplitude)./2);

    original_field_phase=angle(original_field);
    offset_field_phase=angle(offset_field);

    error_real=exclude_7sigma_outliers(error_real);
    error_im=exclude_7sigma_outliers(error_im);
%     error_amplitude=exclude_7sigma_outliers(error_amplitude);
%     error_phase=exclude_7sigma_outliers(error_phase);

    % exclude values where the original field is 0, because that will
    % result in infinity when we divide by non-offset field.
%     exclude_indices=find(original_field==0);
%     error_metrics.error(exclude_indices)=NaN;
%     error_amplitude(exclude_indices)=NaN;
%     error_amplitude(exclude_indices)=NaN;
%     exclude_indices_amplitude=cat(1,exclude_indices_amplitude,amplitude_outliers_indices);
%     exclude_indices_phase=cat(1,exclude_indices_phase,phase_outliers_indices);

    % calculating weighted percent difference--for amplitude and
    % phase--real and imaginary components separately
    weighted_percent_difference_real=error_real./((original_field_real+offset_field_real)/2);
    weighted_percent_difference_im=error_im./((original_field_im+offset_field_im)/2);
    error_metrics.weighted_percent_difference=complex(weighted_percent_difference_real,weighted_percent_difference_im);
    error_metrics.weighted_percent_difference_amplitude=abs(error_metrics.weighted_percent_difference);
    error_metrics.weighted_percent_difference_phase=angle(error_metrics.weighted_percent_difference);
    % taking mean of amplitude and phase errors
    error_metrics.weighted_percent_difference_amplitude_mean=mean(error_metrics.weighted_percent_difference_amplitude,'all','omitnan');
    error_metrics.weighted_percent_difference_phase_mean=mean(error_metrics.weighted_percent_difference_phase,'all','omitnan');
    % mean of imaginary and real components separately
    weighted_percent_difference_real_mean=mean(weighted_percent_difference_real,'all','omitnan');
    weighted_percent_difference_im_mean=mean(weighted_percent_difference_im,'all','omitnan');
    error_metrics.weighted_percent_difference_mean=complex(weighted_percent_difference_real_mean,weighted_percent_difference_im_mean);
    error_metrics.weighted_percent_difference_mean_amplitude=abs(error_metrics.weighted_percent_difference_mean);
    error_metrics.weighted_percent_difference_mean_phase=angle(error_metrics.weighted_percent_difference_mean);
  
    fprintf('Amplitude of mean weighted percent difference: %f\n', error_metrics.weighted_percent_difference_mean_amplitude);
    fprintf('Phase of mean weighted percent difference: %f\n', error_metrics.weighted_percent_difference_mean_phase);
    fprintf('Mean weighted percent difference in amplitude: %f\n',error_metrics.weighted_percent_difference_amplitude_mean);
    fprintf('Mean weighted percent difference in phase: %f\n',error_metrics.weighted_percent_difference_phase_mean);

    % calculating max - min of the error dataset
    error_range_real=max(error_real,[],'all')-min(error_real,[],'all');
    error_range_im=max(error_im,[],'all')-min(error_im,[],'all');
    error_metrics.error_range=complex(error_range_real,error_range_im);
    error_metrics.error_range_amplitude=abs(error_metrics.error_range);
    error_metrics.error_range_phase=angle(error_metrics.error_range);
    pct_difference_range_real=max(weighted_percent_difference_real,[],'all')-min(weighted_percent_difference_real,[],'all');
    pct_difference_range_im=max(weighted_percent_difference_im,[],'all')-min(weighted_percent_difference_im,[],'all');
    error_metrics.pct_difference_range=complex(pct_difference_range_real,pct_difference_range_im);
    error_metrics.pct_difference_range_amplitude=abs(error_metrics.pct_difference_range);
    error_metrics.pct_difference_range_phase=angle(error_metrics.pct_difference_range);

    fprintf('Amplitude of error range: %f\n',error_metrics.error_range_amplitude);
    fprintf('Phase of error range: %f\n',error_metrics.error_range_phase);
    fprintf('Amplitude of percent difference range: %f\n',error_metrics.pct_difference_range_amplitude);
    fprintf('Phase of percent difference range: %f\n',error_metrics.pct_difference_range_phase);

    % standard deviation of the error
    std_error_real=std(error_real,0,'all','omitnan');
    std_error_im=std(error_im,0,'all','omitnan');
    error_metrics.std_error=complex(std_error_real,std_error_im);
    error_metrics.std_error_amplitude=abs(error_metrics.std_error);
    error_metrics.std_error_phase=angle(error_metrics.std_error);
    pct_diff_std_error_real=std(weighted_percent_difference_real,0,'all','omitnan');
    pct_diff_std_error_im=std(weighted_percent_difference_im,0,'all','omitnan');
    error_metrics.pct_diff_std_error=complex(pct_diff_std_error_real,pct_diff_std_error_im);
    error_metrics.pct_diff_std_error_amplitude=abs(error_metrics.pct_diff_std_error);
    error_metrics.pct_diff_std_error_phase=angle(error_metrics.pct_diff_std_error);

    fprintf('Amplitude of standard deviation of the error: %f\n',error_metrics.std_error_amplitude);
    fprintf('Phase of standard deviation of the error: %f\n',error_metrics.std_error_phase);
    fprintf('Amplitude of standard deviation of the percent error: %f\n',error_metrics.pct_diff_std_error_amplitude);
    fprintf('Phase of standard deviation of the percent error: %f\n',error_metrics.pct_diff_std_error_phase);

    % subtracting abs(b1)-abs(b2)--an error that is more of a magnitude
    error_metrics.error_magnitudes=abs(original_field)-abs(offset_field);
    error_metrics.mean_error_magnitudes=mean(error_metrics.error_magnitudes,'all','omitnan');
    error_metrics.pct_error_magnitudes=error_metrics.error_magnitudes./((abs(original_field)+abs(offset_field))/2);
    error_metrics.pct_error_magnitudes_mean=mean(error_metrics.pct_error_magnitudes,'all','omitnan');

    fprintf("Mean of amplitude(b1)-amplitude(b2): %f\n",error_metrics.mean_error_magnitudes);
    fprintf("Mean of this error magnitude metric as percent difference: %f\n",error_metrics.pct_error_magnitudes_mean);

    % this is the wrong way to do abs(b1)-abs(b2)
%     error_magnitudes_real=abs(offset_field_real)-abs(offset_field_im);
%     error_magnitudes_im=abs(offset_field_im)-abs(offset_field_im);
%     error_metrics.error_magnitudes=complex(error_magnitudes_real,error_magnitudes_im);
%     error_magnitudes_real_mean=mean(error_magnitudes_real,'all','omitnan');
%     error_magnitudes_im_mean=mean(error_magnitudes_im,'all','omitnan');
%     error_metrics.error_magnitudes_mean=complex(error_magnitudes_real,error_magnitudes_im);
%     error_metrics.error_magnitudes_mean_amplitude=abs(error_metrics.error_magnitudes_mean);
%     error_metrics.error_magnitudes_mean_phase=angle(error_metrics.error_magnitudes_mean);
%     % turning error magnitudes into a weighted percentage (as above)
%     pct_error_magnitudes_real=error_magnitudes_real./((abs(original_field_real)+abs(offset_field_real))/2);
%     pct_error_magnitudes_real_mean=mean(pct_error_magnitudes_real,'all','omitnan');
%     pct_error_magnitudes_im=error_magnitudes_im./((original_field_im+offset_field_im)/2);
%     pct_error_magnitudes_im_mean=mean(pct_error_magnitudes_im,'all','omitnan');
%     error_metrics.pct_error_magnitudes_mean=complex(pct_error_magnitudes_real_mean,pct_error_magnitudes_im_mean);
%     error_metrics.pct_error_magnitudes_mean_amplitude=abs(error_metrics.pct_error_magnitudes_mean);
%     error_metrics.pct_error_magnitudes_mean_phase=angle(error_metrics.pct_error_magnitudes_mean);

%     phase_percent_error=error_phase./abs(angle(original_field))*100;
%     amplitude_percent_error=error_amplitude./abs(original_field)*100;

%     mean_amplitude_error=mean(error_amplitude,'all','omitnan');
%     max_amplitude_error=max(error_amplitude,[],'all');
%     mean_amplitude_percent_error=mean(amplitude_percent_error,'all','omitnan');
%     max_amplitude_percent_error=max(amplitude_percent_error,[],'all');
%     mean_phase_error=mean(error_phase,'all','omitnan');
%     max_phase_error=max(error_phase,[],"all");
%     mean_phase_percent_error=mean(phase_percent_error,"all",'omitnan');
%     max_phase_percent_error=max(phase_percent_error,[],"all");
% 
%     fprintf('Mean of the residuals amplitudes: %f\n', mean_amplitude_error);
%     fprintf('Maximum of residuals amplitudes (max error): %f\n', max_amplitude_error);
%     fprintf('Mean percent error on residuals amplitudes: %f percent\n', mean_amplitude_percent_error);
%     fprintf('Maximum percent error on residuals amplitudes (max error): %f percent\n', max_amplitude_percent_error);
%     fprintf('Mean of the phase residuals: %f\n', mean_phase_error);
%     fprintf('Maximum of the phase residuals (max error): %f\n', max_phase_error);
%     fprintf('Mean percent error on phase residuals: %f percent\n', mean_phase_percent_error);
%     fprintf('Maximum percent error on phase residuals: %f percent\n', max_phase_percent_error);

% %     plot magnitude residuals
%     figure;
%     subplot(2,1,1);
%     surf(x,y,error_amplitude);
%     title('Plotting magnitude of complex residuals');
%     xlabel(label_1);
%     ylabel(label_2);
%     zlabel('Difference in power on linear scale');
%     colorbar;
%     subplot(2,1,2);
%     surf(x,y,error_phase);
%     title('Plotting residual phase');
%     xlabel(label_1);
%     ylabel(label_2);
%     zlabel('Difference in power on linear scale');
%     colorbar;

    % 3D spherical plot--only helpful for datasets that are in a polar
    % rather than cartesian grid system
    figure;
    subplot(1,2,1);
    sphere3d(error_amplitude,0,2*pi,-pi/2,pi/2,1,1,'surf');
    title('Spherical representation of error in amplitude');
    subplot(1,2,2);
    sphere3d(error_phase,0,2*pi,-pi/2,pi/2,1,1,'surf');
    title('Spherical representation of error in phase');

    figure;
    ax1=subplot(1,2,1);
    plot1=pcolor(x,y,error_amplitude);
    daspect([1 1 1]);
    plot1.EdgeColor='none';
    title('Difference between offset and original field: amplitude');
    xlabel(label_1);
    ylabel(label_2);
    colorbar;
    ax2=subplot(1,2,2);
%     scatter(x_grid,y_grid,400/length(x),'filled');
    plot2=pcolor(x,y,error_phase);
    daspect([1 1 1]);
    plot2.EdgeColor='none';
    title('Difference between offset and original field: phase');
    xlabel(label_1);
    ylabel(label_2);
    colorbar;

    % plotting percent difference over the entire 2D grid
    figure;
%     ax1=subplot(1,2,1);
    plot1=pcolor(x,y,error_metrics.weighted_percent_difference_amplitude);
    daspect([1 1 1]);
    plot1.EdgeColor='none';
    caxis([-Inf 1]);
    title('Weighted percent difference: amplitude');
    xlabel(label_1);
    ylabel(label_2);
    colorbar;
%     ax2=subplot(1,2,2);
% %     scatter(x_grid,y_grid,400/length(x),'filled');
%     plot2=pcolor(x,y,error_metrics.weighted_percent_difference_phase);
%     daspect([1 1 1]);
%     plot2.EdgeColor='none';
%     caxis([-Inf 1]);
%     title('Weighted percent difference: phase');
%     xlabel(label_1);
%     ylabel(label_2);
%     colorbar;

    figure;
    sphere3d(error_metrics.weighted_percent_difference_amplitude,0,2*pi,-pi/2,pi/2,1,1,'surf','spline');

end

function [Etheta,Ephi]=FouriertoFF(f_x,f_y,f_z,theta,phi,k_x,k_y,k0);

% a function to convert from Fourier space to the far field electric field
% pattern

    f_X_Spherical=interp2(k_x,k_y,abs(f_x),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Y_Spherical=interp2(k_x,k_y,abs(f_y),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Z_Spherical=interp2(k_x,k_y,abs(f_z),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    
    % I think that this just sets a radial distance of 1000
    r=10000;
    C=j*(k0*exp(-j*k0*r))/(2*pi*r);
    Etheta=C*(f_X_Spherical.*cos(phi)+f_Y_Spherical.*sin(phi));
    Ephi=C*cos(theta).*(-f_X_Spherical.*sin(phi)+f_Y_Spherical.*cos(phi));
   
end

function output_data=exclude_7sigma_outliers(data)
    % finds all outliers greater than 7 standard deviations away from the
    % mean and sets them to NaN values in the dataset
    
    output_data=data;

    sigma_data=std(data,0,'all');
    preliminary_mean=mean(data,'all');
    outliers_indices=find(abs(data-preliminary_mean) > (7*sigma_data));
    output_data(outliers_indices)=NaN;

end

function [Etheta,Ephi,f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular]=NF2FF(data_pol1,data_pol2,f_Index,k_X_Rectangular,k_Y_Rectangular,M,N,theta,phi)
    c=299792458; % Speed of light in vacuum [m/s]
    
    [k_Y_Rectangular_Grid,k_X_Rectangular_Grid] = meshgrid(k_Y_Rectangular,k_X_Rectangular);

    f=data_pol1.freq(f_Index);
    lambda0=c/f;
    k0=2*pi/lambda0;
    k_Z_Rectangular_Grid = sqrt(k0^2-k_X_Rectangular_Grid.^2-k_Y_Rectangular_Grid.^2);

    %% loading in and plotting all NF data at given frequency

    for iy=1:1:N
       for ix=1:1:M 
           NF_X_Complex(ix,iy)=data_pol1.s21{iy,ix}(f_Index);
           NF_Y_Complex(ix,iy)=data_pol2.s21{iy,ix}(f_Index);
       end
    end

    %% transforming to Fourier space and plotting

    % achieved by taking into account kx, ky, fx, fy
    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,~,~,~]=...
        NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Grid,k_Y_Rectangular_Grid,k_Z_Rectangular_Grid);


    %% converting to the far field

    [Etheta,Ephi]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,theta,phi,k_X_Rectangular,k_Y_Rectangular,k0);
end