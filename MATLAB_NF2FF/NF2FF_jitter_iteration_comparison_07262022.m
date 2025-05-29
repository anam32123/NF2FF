%% Beginning of jitter experiment with iteration

% Here, we use x in columns, y in rows to interpolate and investigate the
% results of jitter on the final electric field patterns. The jitter is
% created by generating a jittered x and y dataset (using random jitter 
% within a certain threshold), and interpolating the data over the orignal
% grid to obtain NF values at the jittered sample locations. Then this data
% is used and assigned to the original spots in the grid, to replicate what
% it's going to look like if the location is off, but we use the original
% gridding. 
%
% We iterate through to produce many jittered grids and error metrics on
% these grids. Then the program takes stats like max, min, mean, median of
% the rror metrics over all of the iterations to understand the overall
% error that will generally result due to some random jitter within the
% threshold.

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

% % a check for our jitter
% max(abs(jitter_x))
% max(abs(jitter_y))
% jitter_threshold

%% calculations

Index = 1;
    
f_Index=201;
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

    %% transforming to far field

    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,f_X_Rectangular_Magnitude,...
        f_Y_Rectangular_Magnitude,f_Z_Rectangular_Magnitude]=NFtoFourierSpace...
        (NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Grid,k_Y_Rectangular_Grid,...
        k_Z_Rectangular_Grid);

    [Etheta,Ephi]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,...
        theta,phi,k_X_Rectangular,k_Y_Rectangular,k0);


    % cannot use the NF2FF function here, because that assumes all data
    % points are read into the NF matrix from sdata


%     Etheta_phase_unwrapped=unwrap(Etheta_phase);
%     Etheta_phase_unwrapped_jittered=unwrap(Etheta_phase_jittered);
%     Ephi_phase_unwrapped=unwrap(Ephi_phase);
%     Ephi_phase_jittered=unwrap(Ephi_phase_jittered);

%     Etheta_phase_unwrapped=GoldsteinUnwrap2D(Etheta);
%     Ephi_phase_unwrapped=GoldsteinUnwrap2D(Ephi);
%     Etheta_phase_jittered_unwrapped=GoldsteinUnwrap2D(Etheta_jittered);
%     Ephi_phase_jittered_unwrapped=GoldsteinUnwrap2D(Ephi_jittered);

    % plots of wrapped and unwrapped phase
%     figure;
%     subplot(2,2,1);
%     plt=pcolor(theta,phi,Etheta_phase);
%     daspect([1 1 1]);
%     plt.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\theta} phase');
%     colorbar;
%     subplot(2,2,2);
%     plt=pcolor(theta,phi,Etheta_phase_unwrapped);
%     daspect([1 1 1]);
%     plt.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\theta} unwrapped phase');
%     colorbar;
%     subplot(2,2,3);
%     plt=pcolor(theta,phi,Ephi_phase);
%     daspect([1 1 1]);
%     plt.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\phi} phase');
%     colorbar;
%     subplot(2,2,4);
%     plt=pcolor(theta,phi,Ephi_phase_unwrapped);
%     daspect([1 1 1]);
%     plt.EdgeColor='none';
%     xlabel('theta');
%     ylabel('phi');
%     title('E_{\phi} unwrapped phase');
%     colorbar;

%     fprintf('Error on E_theta\n');
%     Etheta1_errors=complex_error(Etheta,Etheta_Offset1,theta_vector,phi_vector,'theta','phi');
%     fprintf('Error on E_phi\n');
%     Ephi1_errors=complex_error(Ephi,Ephi_Offset1,theta_vector,phi_vector,'theta','phi');
%     Etheta2_errors=complex_error(Etheta,Etheta_Offset2,theta_vector,phi_vector,'theta','phi');
%     Ephi2_errors=complex_error(Ephi,Ephi_Offset2,theta_vector,phi_vector,'theta','phi');
%     Etheta3_errors=complex_error(Etheta,Etheta_Offset3,theta_vector,phi_vector,'theta','phi');
%     Ephi3_errors=complex_error(Ephi,Ephi_Offset3,theta_vector,phi_vector,'theta','phi');
% 	% Calculation of radiated power through numerical integration
% 	e_theta = [1 4 repmat([2 4], 1, floor(length(theta(1,:))/2) - 1) 1];
%     e_phi = [1 4 repmat([2 4], 1, floor(length(phi(:,1))/2) - 1) 1];
%     P = dphi*dtheta*sum(sum(U.*(e_theta'*e_phi).*abs(sin(theta))))/9;
% 	
%     D = 4*pi*U/P;
%     U_Co = (abs(Etheta.*cos(phi)-Ephi.*sin(phi))).^2;
%     U_Cross = (abs(Etheta.*sin(phi)+Ephi.*cos(phi))).^2;
%     D_Co = 4*pi*U_Co/P;
%     D_Cross = 4*pi*U_Cross/P;

    % side by side heatmaps of weighted % difference in amplitude for all
    % the offsets, E_theta and E_phi
%     errors=[Etheta1_errors,Etheta2_errors,Etheta3_errors,Ephi1_errors,Ephi2_errors,Ephi3_errors];
%     figure;
%     for index=1:6
%         subplot(2,3,index);
%         plot1=pcolor(theta,phi,errors(index).weighted_percent_difference_amplitude);
%         daspect([1 1 1]);
%         plot1.LineWidth=0.5;
%         caxis([-Inf 1]);
%         if index < 4
%             title('Weighted percent difference in E_\theta: amplitude');
%         else
%             title('Weighted percent difference in E_\phi: amplitude');
%         end
%         xlabel('Theta');
%         ylabel('Phi');
%         colorbar;
%     end
% 	
%     % scatter plot of average percent difference in amplitude vs percentage
%     % offset
%     for index=1:3
%         mean_pct_diff_amplitudes_Etheta(index)=errors(index).weighted_percent_difference_mean_amplitude;
%     end
%     i=1;
%     for index=4:6
%         mean_pct_diff_amplitudes_Ephi(i)=errors(index).weighted_percent_difference_mean_amplitude;
%         i=i+1;
%     end
%     figure;
%     semilogx(pct_offsets,mean_pct_diff_amplitudes_Etheta,'b.','MarkerSize',15);
%     hold on;
%     semilogx(pct_offsets,mean_pct_diff_amplitudes_Ephi,'r.','MarkerSize',15);
%     title('Mean percent difference vs offset');
%     xlabel('Offset as percentage of Nyquist spacing');
%     ylabel('Mean percent difference between offset and original field');
%     legend('E_\theta','E_\phi');

%% jittered stuff

% jitter_percentage_thresholds=[0.01,1/15,0.1]; % max jitter in x and y as percentage of Nyquist spacing (for now)
jitter_percentage_threshold=0.1;
lambda_m=c/broadcast_freq;
nyquist_sampling_m=lambda_m/2; % necessary grid spacing in m


% setting the 
jitter_threshold=nyquist_sampling_m*jitter_percentage_threshold;

% arrays to store error metrics
% weighted_percent_difference_amplitude_means,weighted_percent_difference_phase_means,...
%     weighted_percent_difference_mean_amplitudes,weighted_percent_difference_mean_phases,...
%     error_range_amplitudes,error_range_phases,pct_diff_range_amplitudes,pct_diff_range_phases,...
%     std_error_amplitudes,std_err_phases,pct_diff_std_error_amplitudes,pct_diff_std_error_phases,...
%     error_magnitudes_means,pct_error_magnitudes_means=[];
    
for index=1:1000
    jitter_x=-jitter_threshold+2.*jitter_threshold.*rand(1,M);
    jitter_y=-jitter_threshold+2.*jitter_threshold.*rand(1,N);
    x_jittered=x+jitter_x;
    y_jittered=y+jitter_y;
    [x_jittered_grid,y_jittered_grid]=meshgrid(x_jittered,y_jittered);

     % interpolating near-field data to obtain the jittered version of
     % near-field data
        NF_X_Complex_jittered=interp2(x,y,NF_X_Complex,x_jittered_grid,y_jittered_grid,'spline');
        NF_Y_Complex_jittered=interp2(x,y,NF_Y_Complex,x_jittered_grid,y_jittered_grid,'spline');
    
        NF_X_jittered_Magnitude=20*log10(abs(NF_X_Complex_jittered));
        NF_Y_jittered_Magnitude=20*log10(abs(NF_Y_Complex_jittered));

    [f_X_Rectangular_jittered,f_Y_Rectangular_jittered,f_Z_Rectangular_jittered,...
        f_X_Rectangular_jittered_Magnitude,f_Y_Rectangular_jittered_Magnitude,...
        f_Z_Rectangular_jittered_Magnitude]=NFtoFourierSpace(NF_X_Complex_jittered,...
        NF_Y_Complex_jittered,k_X_Rectangular_Grid,k_Y_Rectangular_Grid,...
        k_Z_Rectangular_Grid);

    [Etheta_jittered,Ephi_jittered]=FouriertoFF(f_X_Rectangular_jittered,f_Y_Rectangular_jittered,...
        f_Z_Rectangular_jittered,theta,phi,k_X_Rectangular,k_Y_Rectangular,k0);

    % error stuff
%     fprintf('E_theta errors:\n');
    Etheta_error_metrics=electric_field_error(Etheta,Etheta_jittered,theta_vector,phi_vector,'theta','phi',0);
%     fprintf('E_phi errors:\n');
    Ephi_error_metrics=electric_field_error(Ephi,Ephi_jittered,theta_vector,phi_vector,'theta','phi',0);

    % adding error metrics to arrays
    all_Etheta_errors.weighted_percent_difference_amplitude_means(index)=Etheta_error_metrics.weighted_percent_difference_amplitude_mean;
    all_Ephi_errors.weighted_percent_difference_amplitude_means(index)=Ephi_error_metrics.weighted_percent_difference_amplitude_mean;
    all_Etheta_errors.weighted_percent_difference_phase_means(index)=Etheta_error_metrics.weighted_percent_difference_phase_mean;
    all_Ephi_errors.weighted_percent_difference_phase_means(index)=Ephi_error_metrics.weighted_percent_difference_phase_mean;
    all_Etheta_errors.weighted_percent_difference_mean_amplitudes(index)=Etheta_error_metrics.weighted_percent_difference_mean_amplitude;
    all_Ephi_errors.weighted_percent_difference_mean_amplitudes(index)=Ephi_error_metrics.weighted_percent_difference_mean_amplitude;
    all_Etheta_errors.weighted_percent_difference_mean_phases(index)=Etheta_error_metrics.weighted_percent_difference_mean_phase;
    all_Ephi_errors.weighted_percent_difference_mean_phases(index)=Ephi_error_metrics.weighted_percent_difference_mean_phase;
    all_Etheta_errors.error_range_amplitudes(index)=Etheta_error_metrics.error_range_amplitude;
    all_Ephi_errors.error_range_amplitudes(index)=Ephi_error_metrics.error_range_amplitude;
    all_Etheta_errors.error_range_phases(index)=Etheta_error_metrics.error_range_phase;
    all_Ephi_errors.error_range_phases(index)=Ephi_error_metrics.error_range_phase;
    all_Etheta_errors.pct_difference_range_amplitudes(index)=Etheta_error_metrics.pct_difference_range_amplitude;
    all_Ephi_errors.pct_difference_range_amplitudes(index)=Ephi_error_metrics.pct_difference_range_amplitude;
    all_Etheta_errors.pct_difference_range_phases(index)=Etheta_error_metrics.pct_difference_range_phase;
    all_Ephi_errors.pct_difference_range_phases(index)=Ephi_error_metrics.pct_difference_range_phase;
    all_Etheta_errors.std_error_amplitudes(index)=Etheta_error_metrics.std_error_amplitude;
    all_Ephi_errors.std_error_amplitudes(index)=Ephi_error_metrics.std_error_amplitude;
    all_Etheta_errors.std_error_phases(index)=Etheta_error_metrics.std_error_phase;
    all_Ephi_errors.std_error_phases(index)=Ephi_error_metrics.std_error_phase;
    all_Etheta_errors.pct_diff_std_error_amplitudes(index)=Etheta_error_metrics.pct_diff_std_error_amplitude;
    all_Ephi_errors.pct_diff_std_error_amplitudes(index)=Ephi_error_metrics.pct_diff_std_error_amplitude;
    all_Etheta_errors.pct_diff_std_error_phases(index)=Etheta_error_metrics.pct_diff_std_error_phase;
    all_Ephi_errors.pct_diff_std_error_phases(index)=Ephi_error_metrics.pct_diff_std_error_phase;
    all_Etheta_errors.error_magnitudes_means(index)=Etheta_error_metrics.error_magnitudes_mean;
    all_Ephi_errors.error_magnitudes_means(index)=Ephi_error_metrics.error_magnitudes_mean;
    all_Etheta_errors.pct_error_magnitudes_means(index)=Etheta_error_metrics.pct_error_magnitudes_mean;
    all_Ephi_errors.pct_error_magnitudes_means(index)=Ephi_error_metrics.pct_error_magnitudes_mean;
    all_Etheta_errors.solid_angle_error_fractions(index)=Etheta_error_metrics.solid_angle_error_fraction;
    all_Ephi_errors.solid_angle_error_fractions(index)=Ephi_error_metrics.solid_angle_error_fraction;
    all_Etheta_errors.solid_angle_error_amplitudes(index)=Etheta_error_metrics.solid_angle_error_amplitude;
    all_Ephi_errors.solid_angle_error_amplitudes(index)=Ephi_error_metrics.solid_angle_error_amplitude;
    all_Etheta_errors.solid_angle_error_fractions_excluded(index)=Etheta_error_metrics.solid_angle_error_fraction_excluded;
    all_Ephi_errors.solid_angle_error_fractions_excluded(index)=Ephi_error_metrics.solid_angle_error_fraction_excluded;
    all_Etheta_errors.solid_angle_error_amplitudes_excluded(index)=Etheta_error_metrics.solid_angle_error_amplitude_excluded;
    all_Ephi_errors.solid_angle_error_amplitudes_excluded(index)=Ephi_error_metrics.solid_angle_error_amplitude_excluded;

end

%% analysis on the lists
% vector version of the mean
fprintf('Max of amplitudes of mean percent difference (vector) for E_theta: %f\n',max(all_Etheta_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Max of amplitudes of mean percent difference (vector) for E_phi: %f\n', max(all_Ephi_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Min of amplitudes of mean percent difference (vector) for E_theta: %f\n',min(all_Etheta_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Min of amplitudes of mean percent difference (vector) for E_phi: %f\n',min(all_Ephi_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Mean of amplitudes of mean percent difference (vector) for E_theta: %f\n',mean(all_Etheta_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Mean of amplitudes of mean percent differnece (vector) for E_phi: %f\n',mean(all_Ephi_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Median of amplitudes of mean percent difference (vector) for E_theta: %f\n',median(all_Etheta_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Median of amplitudes of mean percent difference (vector) for E_phi: %f\n',median(all_Ephi_errors.weighted_percent_difference_mean_amplitudes));
fprintf('Mean of mean percent difference amplitudes (non-vector) for E_theta: %f\n',mean(all_Etheta_errors.weighted_percent_difference_amplitude_means));
fprintf('Mean of mean percent difference amplitudes (non-vector) for E_phi: %f\n',mean(all_Ephi_errors.weighted_percent_difference_amplitude_means));
fprintf('Max of mean percent difference amplitudes (non-vector) for E_theta: %f\n',max(all_Etheta_errors.weighted_percent_difference_amplitude_means));
fprintf('Max of mean percent difference amplitudes (non-vector) for E_phi: %f\n',max(all_Ephi_errors.weighted_percent_difference_amplitude_means));
fprintf('Min of mean percent difference amplitudes (non-vector) for E_theta: %f\n',min(all_Etheta_errors.weighted_percent_difference_amplitude_means));
fprintf('Min of mean percent difference amplitudes (non-vector) for E_phi: %f\n',min(all_Ephi_errors.weighted_percent_difference_amplitude_means));
fprintf('Median of mean percent difference amplitudes (non-vector) for E_theta: %f\n',median(all_Etheta_errors.weighted_percent_difference_amplitude_means));
fprintf('Median of mean percent difference amplitudes (non-vector) for E_phi: %f\n',median(all_Ephi_errors.weighted_percent_difference_amplitude_means));
fprintf('Mean of mean abs(b1)-abs(b2) as percent error for E_theta: %f\n',mean(all_Etheta_errors.pct_error_magnitudes_means));
fprintf('Mean of mean abs(b1)-abs(b2) as percent error for E_phi: %f\n',mean(all_Ephi_errors.pct_error_magnitudes_means));
fprintf('Max of mean abs(b1)-abs(b2) as percent error for E_theta: %f\n',max(all_Etheta_errors.pct_error_magnitudes_means));
fprintf('Max of mean abs(b1)-abs(b2) as percent error for E_phi: %f\n',max(all_Etheta_errors.pct_error_magnitudes_means));
fprintf('Min of mean abs(b1)-abs(b2) as percent error for E_theta: %f\n',min(all_Etheta_errors.pct_error_magnitudes_means));
fprintf('Min of mean abs(b1)-abs(b2) as percent error for E_phi: %f\n',min(all_Etheta_errors.pct_error_magnitudes_means));
fprintf('Median of mean abs(b1)-abs(b2) as percent error for E_theta: %f\n',median(all_Etheta_errors.pct_error_magnitudes_means));
fprintf('Median of mean abs(b1)-abs(b2) as percent error for E_phi: %f\n',median(all_Ephi_errors.pct_error_magnitudes_means));
fprintf('Max of solid angle error fraction for E_theta: %f\n',max(all_Etheta_errors.solid_angle_error_fractions));
fprintf('Max of solid angle error fraction for E_phi: %f\n',max(all_Ephi_errors.solid_angle_error_fractions));
fprintf('Min of solid angle error fraction for E_theta: %f\n',min(all_Etheta_errors.solid_angle_error_fractions));
fprintf('Min of solid angle error fraction for E_phi: %f\n',min(all_Ephi_errors.solid_angle_error_fractions));
fprintf('Mean of solid angle error fraction for E_theta: %f\n',mean(all_Etheta_errors.solid_angle_error_fractions));
fprintf('Mean of solid angle error fraction for E_phi: %f\n',mean(all_Ephi_errors.solid_angle_error_fractions));
fprintf('Median of solid angle error fraction for E_theta: %f\n',median(all_Etheta_errors.solid_angle_error_fractions));
fprintf('Median of solid angle error fraction for E_phi: %f\n',median(all_Ephi_errors.solid_angle_error_fractions));
fprintf('Max of solid angle error fraction amplitude (vector) for E_theta: %f\n',max(all_Etheta_errors.solid_angle_error_amplitudes));
fprintf('Max of solid angle error fraction amplitude (vector) for E_phi: %f\n',max(all_Ephi_errors.solid_angle_error_amplitudes));
fprintf('Min of solid angle error fraction amplitude (vector) for E_theta: %f\n',min(all_Etheta_errors.solid_angle_error_amplitudes));
fprintf('Min of solid angle error fraction amplitude (vector) for E_phi: %f\n',min(all_Ephi_errors.solid_angle_error_amplitudes));
fprintf('Mean of solid angle error fraction amplitude (vector) for E_theta: %f\n',mean(all_Etheta_errors.solid_angle_error_amplitudes));
fprintf('Mean of solid angle error fraction amplitude (vector) for E_phi: %f\n',mean(all_Ephi_errors.solid_angle_error_amplitudes));
fprintf('Median of solid angle error fraction amplitude (vector) for E_theta: %f\n',median(all_Etheta_errors.solid_angle_error_amplitudes));
fprintf('Median of solid angle error fraction amplitude (vector) for E_phi: %f\n',median(all_Ephi_errors.solid_angle_error_amplitudes));
fprintf('Max of solid angle error fraction (main beam only, non-vector) for E_theta: %f\n',max(all_Etheta_errors.solid_angle_error_fractions_excluded));
fprintf('Max of solid angle error fraction (main beam only, non-vector) for E_phi: %f\n',max(all_Ephi_errors.solid_angle_error_fractions_excluded));
fprintf('Min of solid angle error fraction (main beam only, non-vector) for E_theta: %f\n',min(all_Etheta_errors.solid_angle_error_fractions_excluded));
fprintf('Min of solid angle error fraction (main beam only, non-vector) for E_phi: %f\n',min(all_Ephi_errors.solid_angle_error_fractions_excluded));
fprintf('Mean of solid angle error fraction (main beam only, non-vector) for E_theta: %f\n',mean(all_Etheta_errors.solid_angle_error_fractions_excluded));
fprintf('Mean of solid angle error fraction (main beam only, non-vector) for E_phi: %f\n',mean(all_Ephi_errors.solid_angle_error_fractions_excluded));
fprintf('Median of solid angle error fraction (main beam only, non-vector) for E_theta: %f\n',median(all_Etheta_errors.solid_angle_error_fractions_excluded));
fprintf('Median of solid angle error fraction (main beam only, non-vector) for E_phi: %f\n',median(all_Ephi_errors.solid_angle_error_fractions_excluded));
fprintf('Max of solid angle error fraction (main beam only, vector) for E_theta: %f\n',max(all_Etheta_errors.solid_angle_error_amplitudes_excluded));
fprintf('Max of solid angle error fraction (main beam only, vector) for E_phi: %f\n',max(all_Ephi_errors.solid_angle_error_amplitudes_excluded));
fprintf('Min of solid angle error fraction (main beam only, vector) for E_theta: %f\n',min(all_Etheta_errors.solid_angle_error_amplitudes_excluded));
fprintf('Min of solid angle error fraction (main beam only, vector) for E_phi: %f\n',min(all_Ephi_errors.solid_angle_error_amplitudes_excluded));
fprintf('Mean of solid angle error fraction (main beam only, vector) for E_theta: %f\n',mean(all_Etheta_errors.solid_angle_error_amplitudes_excluded));
fprintf('Mean of solid angle error fraction (main beam only, vector) for E_phi: %f\n',mean(all_Ephi_errors.solid_angle_error_amplitudes_excluded));
fprintf('Median of solid angle error fraction (main beam only, vector) for E_theta: %f\n',median(all_Etheta_errors.solid_angle_error_amplitudes_excluded));
fprintf('Median of solid angle error fraction (main beam only, vector) for E_phi: %f\n',median(all_Ephi_errors.solid_angle_error_amplitudes_excluded));

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
%     weighted_percent_difference_real_mean=mean(weighted_percent_difference_real,'all','omitnan');
%     weighted_percent_difference_im_mean=mean(weighted_percent_difference_im,'all','omitnan');
%     error_metrics.weighted_percent_difference_mean=complex(weighted_percent_difference_real_mean,weighted_percent_difference_im_mean);
%     error_metrics.weighted_percent_difference_mean_amplitude=abs(error_metrics.weighted_percent_difference_mean);
%     error_metrics.weighted_percent_difference_mean_phase=angle(error_metrics.weighted_percent_difference_mean);
    % taking mean as the percentage of average absolute error calculated above
    error_real_mean=mean(error_real,'all','omitnan');
    error_im_mean=mean(error_im,'all','omitnan');
    weighted_percent_difference_real_mean=error_real_mean./max(original_field_real,[],'all');
    weighted_percent_difference_im_mean=error_im_mean./max(original_field_im,[],'all');
    error_metrics.weighted_percent_difference_mean=complex(weighted_percent_difference_real_mean,weighted_percent_difference_im_mean);
    error_metrics.weighted_percent_difference_mean_amplitude=abs(error_metrics.weighted_percent_difference_mean);
    error_metrics.weighted_percent_difference_mean_phase=angle(error_metrics.weighted_percent_difference_mean);    
    

%     fprintf('Amplitude of mean weighted percent difference: %f\n', error_metrics.weighted_percent_difference_mean_amplitude);
%     fprintf('Phase of mean weighted percent difference: %f\n', error_metrics.weighted_percent_difference_mean_phase);
%     fprintf('Mean weighted percent difference in amplitude: %f\n',error_metrics.weighted_percent_difference_amplitude_mean);
%     fprintf('Mean weighted percent difference in phase: %f\n',error_metrics.weighted_percent_difference_phase_mean);

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

%     fprintf('Amplitude of error range: %f\n',error_metrics.error_range_amplitude);
%     fprintf('Phase of error range: %f\n',error_metrics.error_range_phase);
%     fprintf('Amplitude of percent difference range: %f\n',error_metrics.pct_difference_range_amplitude);
%     fprintf('Phase of percent difference range: %f\n',error_metrics.pct_difference_range_phase);

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

%     fprintf('Amplitude of standard deviation of the error: %f\n',error_metrics.std_error_amplitude);
%     fprintf('Phase of standard deviation of the error: %f\n',error_metrics.std_error_phase);
%     fprintf('Amplitude of standard deviation of the percent error: %f\n',error_metrics.pct_diff_std_error_amplitude);
%     fprintf('Phase of standard deviation of the percent error: %f\n',error_metrics.pct_diff_std_error_phase);

    % subtracting abs(b1)-abs(b2)--an error that is more of a magnitude
    error_metrics.error_magnitudes=abs(original_field)-abs(offset_field);
    error_metrics.error_magnitudes_mean=mean(error_metrics.error_magnitudes,'all','omitnan');
    error_metrics.pct_error_magnitudes=error_metrics.error_magnitudes./((abs(original_field)+abs(offset_field))/2);
    error_metrics.pct_error_magnitudes_mean=mean(error_metrics.pct_error_magnitudes,'all','omitnan');

%     fprintf("Mean of amplitude(b1)-amplitude(b2): %f\n",error_metrics.error_magnitudes_mean);
%     fprintf("Mean of this error magnitude metric as percent difference: %f\n",error_metrics.pct_error_magnitudes_mean);

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
%     figure;
%     subplot(1,2,1);
%     sphere3d(error_amplitude,0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('Spherical representation of error in amplitude');
%     subplot(1,2,2);
%     sphere3d(error_phase,0,2*pi,-pi/2,pi/2,1,1,'surf');
%     title('Spherical representation of error in phase');

%     figure;
%     ax1=subplot(1,2,1);
%     plot1=pcolor(x,y,error_amplitude);
%     daspect([1 1 1]);
%     plot1.EdgeColor='none';
%     title('Difference between offset and original field: amplitude');
%     xlabel(label_1);
%     ylabel(label_2);
%     colorbar;
%     ax2=subplot(1,2,2);
%     scatter(x_grid,y_grid,400/length(x),'filled');
%     plot2=pcolor(x,y,error_phase);
%     daspect([1 1 1]);
%     plot2.EdgeColor='none';
%     title('Difference between offset and original field: phase');
%     xlabel(label_1);
%     ylabel(label_2);
%     colorbar;

    % plotting percent difference over the entire 2D grid
%     figure;
% %     ax1=subplot(1,2,1);
%     plot1=pcolor(x,y,error_metrics.weighted_percent_difference_amplitude);
%     daspect([1 1 1]);
%     plot1.EdgeColor='none';
%     caxis([-Inf 1]);
%     title('Weighted percent difference: amplitude');
%     xlabel(label_1);
%     ylabel(label_2);
%     colorbar;
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

%     figure;
%     sphere3d(error_metrics.weighted_percent_difference_amplitude,0,2*pi,-pi/2,pi/2,1,1,'surf','spline');

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