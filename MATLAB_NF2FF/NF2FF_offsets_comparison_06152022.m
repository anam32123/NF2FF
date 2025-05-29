%% what I'm doing in this script: removing fairly unnecesary stuff that was in the original script and comparing the errors produced by the various offsets we are comparing
% Current relevant offsets: 0.1%, 1%, and 10% of Nyquist spacing. The
% 10% spacing was already done.

% A NF2FF script comparing the effects of horizontal offsets on the data

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

% nyquist sampling and offsets from that
lambda_m=c/broadcast_freq;
nyquist_sampling_m=lambda_m/2; % necessary grid spacing in m
pct_offsets=[0.001,0.01,0.1,0.00001];
offsets=nyquist_sampling_m.*pct_offsets;
index=4;
current_offset=offsets(index);
x_offset=x+current_offset;

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
[k_Y_Rectangular_Grid,k_X_Rectangular_Grid] = meshgrid(k_Y_Rectangular,k_X_Rectangular);

% creating offset m, kx, ky, etc.
% m_offset=x_offset/dx;
% m_offset_spaced=linspace(m_offset(1)*4,m_offset(M)*4,MI);
m_offset_spaced=m+current_offset/dx;
% defining offset k grid
k_X_Rectangular_Offset=2*pi.*m_offset_spaced/(MI*dx);
k_Y_Rectangular_Offset=2*pi.*n/(NI*dy);
[k_Y_Rectangular_Offset_Grid,k_X_Rectangular_Offset_Grid] = meshgrid(k_Y_Rectangular_Offset,k_X_Rectangular_Offset);

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
    
    close all;
    f=freq(f_Index);
    lambda0=c/f;
    k0=2*pi/lambda0;
    k_Z_Rectangular_Grid = sqrt(k0^2-k_X_Rectangular_Grid.^2-k_Y_Rectangular_Grid.^2);

    % offset version of k_z
    k_Z_Rectangular_Offset_Grid=sqrt(k0^2-k_X_Rectangular_Offset_Grid.^2-k_Y_Rectangular_Offset_Grid.^2);

    %% loading in and plotting all NF data at given frequency

    for iy=1:1:N
       for ix=1:1:M 
           NF_X_Complex(ix,iy)=sdata.s21{iy,ix}(f_Index);
           NF_Y_Complex(ix,iy)=sdata2.s21{iy,ix}(f_Index);
       end
    end

%     NF_X_Magnitude = 20*log10(abs(NF_X_Complex));
%     NF_Y_Magnitude = 20*log10(abs(NF_Y_Complex));

    % plotting NF Data
%     figure;
%     subplot(2,2,1)
%     surf(x*1000,y*1000,NF_X_Magnitude');
%     title(sprintf('f = %f GHz (z = %i mm)--without offset',f/1000000000,z0*1000));
%     xlabel('x (mm)');
%     ylabel('y (mm)');
%     zlabel('|E_{x}| (dB)');
%     set(gca,'XLim',[min(x)*1000 max(x)*1000]);
%     set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(2,2,2);
%     surf(x*1000,y*1000,NF_Y_Magnitude');
%     title('Without offset')
%     xlabel('x (mm)');
%     ylabel('y (mm)');
%     zlabel('|E_{y}| (dB)');
%     set(gca,'XLim',[min(x)*1000 max(x)*1000]);
%     set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     %print(gcf,'-dtiff',['NF_dB_' num2str(f) '_GHz']);
%     
%     % adding offset plot
%     subplot(2,2,3)
%     surf(x_offset*1000,y*1000,NF_X_Magnitude');
%     title(sprintf('f = %f GHz (z = %i mm)--with x offset',f/1000000000,z0*1000));
%     xlabel('x (mm)');
%     ylabel('y (mm)');
%     zlabel('|E_{x}| (dB)');
%     set(gca,'XLim',[min(x_offset)*1000 max(x_offset)*1000]);
%     set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;
%     subplot(2,2,4);
%     surf(x_offset*1000,y*1000,NF_Y_Magnitude');
%     title('With x offset')
%     xlabel('x (mm)');
%     ylabel('y (mm)');
%     zlabel('|E_{y}| (dB)');
%     set(gca,'XLim',[min(x_offset)*1000 max(x_offset)*1000]);
%     set(gca,'YLim',[min(y)*1000 max(y)*1000]);
%     view(-37.5,30);
%     shading flat;
%     colorbar;

%     clear NF_X_Magnitude NF_Y_Magnitude;

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
    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,f_X_Rectangular_Magnitude,...
        f_Y_Rectangular_Magnitude,f_Z_Rectangular_Magnitude]=...
        NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Grid,k_Y_Rectangular_Grid,k_Z_Rectangular_Grid);

    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset,f_X_Rectangular_Magnitude,...
        f_Y_Rectangular_Magnitude,f_Z_Rectangular_Offset_Magnitude]...
        =NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Offset_Grid,...
        k_Y_Rectangular_Offset_Grid,k_Z_Rectangular_Offset_Grid);


    % plotting in Fourier space--cannot add any offset here because it's a
    % function of kx, ky, not x, y
    fourier_fig=figure;
    title(sprintf('f = %f GHz',f/1000000000));
    subplot(3,2,1);
    surf(k_X_Rectangular,k_Y_Rectangular,f_X_Rectangular_Magnitude');
    title('With no offset')
    xlabel(sprintf('k_{x} (m^{-1})'));
    ylabel(sprintf('k_{y} (m^{-1})'));
    zlabel('|f_{x}| (dB)');
    set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
    set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,3);
    surf(k_X_Rectangular,k_Y_Rectangular,f_Y_Rectangular_Magnitude');
    xlabel(sprintf('k_{x} (m^{-1})'));
    ylabel(sprintf('k_{y} (m^{-1})'));
    zlabel('|f_{y}| (dB)');
    set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
    set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,5);
    surf(k_X_Rectangular,k_Y_Rectangular,f_Z_Rectangular_Magnitude');
    xlabel(sprintf('k_{x} (m^{-1})'));
    ylabel(sprintf('k_{y} (m^{-1})'));
    zlabel('|f_{z}| (dB)');
    set(gca,'XLim',[min(k_X_Rectangular) max(k_X_Rectangular)]);
    set(gca,'YLim',[min(k_Y_Rectangular) max(k_Y_Rectangular)]);
    view(-37.5,30);
    shading flat;
    colorbar;

    % plotting offset version of this
    subplot(3,2,2);
    surf(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,f_X_Rectangular_Magnitude');
    title(sprintf('With a %f mm offset, %f percent of Nyquist spacing',current_offset*1000,pct_offsets(index)*100));
    xlabel(sprintf('k_{x} (m^{-1})'));
    ylabel(sprintf('k_{y} (m^{-1})'));
    zlabel('|f_{x}| (dB)');
    set(gca,'XLim',[min(k_X_Rectangular_Offset) max(k_X_Rectangular_Offset)]);
    set(gca,'YLim',[min(k_Y_Rectangular_Offset) max(k_Y_Rectangular_Offset)]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,4);
    surf(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,f_Y_Rectangular_Magnitude');
    xlabel(sprintf('k_{x} (m^{-1})'));
    ylabel(sprintf('k_{y} (m^{-1})'));
    zlabel('|f_{y}| (dB)');
    set(gca,'XLim',[min(k_X_Rectangular_Offset) max(k_X_Rectangular_Offset)]);
    set(gca,'YLim',[min(k_Y_Rectangular_Offset) max(k_Y_Rectangular_Offset)]);
    view(-37.5,30);
    shading flat;
    colorbar;
    subplot(3,2,6);
    surf(k_X_Rectangular_Offset,k_Y_Rectangular_Offset,f_Z_Rectangular_Offset_Magnitude');
    xlabel(sprintf('k_{x} (m^{-1})'));
    ylabel(sprintf('k_{y} (m^{-1})'));
    zlabel('|f_{z}| (dB)');
    set(gca,'XLim',[min(k_X_Rectangular_Offset) max(k_X_Rectangular_Offset)]);
    set(gca,'YLim',[min(k_Y_Rectangular_Offset) max(k_Y_Rectangular_Offset)]);
    view(-37.5,30);
    shading flat;
    colorbar;

    % fx, fy, fz residuals--fx and fy are the same, so there is not need to
    % get residuals on that (if we are doing the subtracting grids rather
    % than interpolation method)

    [b,bb,bbb,bbbb,c,cc,ccc,cccc]=complex_error(f_Z_Rectangular,f_Z_Rectangular_Offset,k_X_Rectangular,k_Y_Rectangular,'k_{x} {m^{-1}}','k_{y} {m_{-1}}');

%% converting to the far field

    [Etheta,Ephi]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,theta,phi,k_X_Rectangular,k_Y_Rectangular,k0);
    [Etheta_Offset,Ephi_Offset]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset,theta,phi,k_X_Rectangular_Offset,k_Y_Rectangular_Offset,k0);

   
    fprintf('Error on E_theta\n');
    [result1,result2,result3,result4,result5,result6,result7,result8]=complex_error(Etheta,Etheta_Offset,theta,phi,'theta','phi');
    fprintf('Error on E_phi\n');
    [result1,result2,result3,result4,result5,result6,result7,result8]=complex_error(Ephi,Ephi_Offset,theta,phi,'theta','phi');
    
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

    
	
%% plots

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

    Index=Index+1;

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
function [error_amplitude,error_phase,mean_amplitude_error,max_amplitude_error,...
    mean_amplitude_percent_error,max_amplitude_percent_error,mean_phase_error,...
    max_phase_error,mean_phase_percent_error,max_phase_percent_error]=...
    complex_error(original_field,offset_field,x,y,label_1,label_2)

    % x and y are just vectors for plotting--could be phi and theta, kx and
    % ky, etc.

    error=offset_field-original_field;
    error_phase=abs(angle(error));
    % all amplitude is done in linear scale (not log)
    error_amplitude=abs(error);

    % removing 7-sigma outliers
    sigma_amplitude=std(error_amplitude,0,'all');
    sigma_phase=std(error_phase,0,'all');
%     fprintf('Standard deviation of the amplitude: %f\n',sigma_amplitude);
%     fprintf('7 sigma for amplitude: %f\n',7*sigma_amplitude);
%     fprintf('Standard deviation of the phase: %f\n',sigma_phase);
%     fprintf('7 sigma for phase: %f\n',7*sigma_phase);

    original_field_phase=abs(angle(original_field));

    amplitude_preliminary_mean=mean(error_amplitude,'all');
    amplitude_outliers_indices=find(abs(error_amplitude-amplitude_preliminary_mean) > (7*sigma_amplitude));
    phase_preliminary_mean=mean(error_phase,'all');
    phase_outliers_indices=find(abs(error_phase-phase_preliminary_mean) > (7*sigma_phase));
    
    % exclude values where the original field is 0, because that will
    % result in infinity when we divide by non-offset field.
    exclude_indices_amplitude=find(original_field==0);
    exclude_indices_phase=find(abs(angle(original_field))==0);
    exclude_indices_amplitude=cat(1,exclude_indices_amplitude,amplitude_outliers_indices);
    exclude_indices_phase=cat(1,exclude_indices_phase,phase_outliers_indices);
    
    error_amplitude(exclude_indices_amplitude)=NaN;
    error_phase(exclude_indices_phase)=NaN;

    phase_percent_error=error_phase./abs(angle(original_field))*100;
    amplitude_percent_error=error_amplitude./abs(original_field)*100;

    mean_amplitude_error=mean(error_amplitude,'all','omitnan');
    max_amplitude_error=max(error_amplitude,[],'all');
    mean_amplitude_percent_error=mean(amplitude_percent_error,'all','omitnan');
    max_amplitude_percent_error=max(amplitude_percent_error,[],'all');
    mean_phase_error=mean(error_phase,'all','omitnan');
    max_phase_error=max(error_phase,[],"all");
    mean_phase_percent_error=mean(phase_percent_error,"all",'omitnan');
    max_phase_percent_error=max(phase_percent_error,[],"all");

    fprintf('Mean of the residuals amplitudes: %f\n', mean_amplitude_error);
    fprintf('Maximum of residuals amplitudes (max error): %f\n', max_amplitude_error);
    fprintf('Mean percent error on residuals amplitudes: %f percent\n', mean_amplitude_percent_error);
    fprintf('Maximum percent error on residuals amplitudes (max error): %f percent\n', max_amplitude_percent_error);
    fprintf('Mean of the phase residuals: %f\n', mean_phase_error);
    fprintf('Maximum of the phase residuals (max error): %f\n', max_phase_error);
    fprintf('Mean percent error on phase residuals: %f percent\n', mean_phase_percent_error);
    fprintf('Maximum percent error on phase residuals: %f percent\n', max_phase_percent_error);

    % plot magnitude residuals
    figure;
    subplot(2,1,1);
    surf(x,y,error_amplitude);
    title('Plotting magnitude of complex residuals');
    xlabel(label_1);
    ylabel(label_2);
    zlabel('Difference in power on linear scale');
    colorbar;
    subplot(2,1,2);
    surf(x,y,error_phase);
    title('Plotting residual phase');
    xlabel(label_1);
    ylabel(label_2);
    zlabel('Difference in power on linear scale');
    colorbar;

end

function [Etheta,Ephi]=FouriertoFF(f_x,f_y,f_z,theta,phi,k_x,k_y,k0);

% a function to convert from Fourier space to the far field electric field
% pattern

    f_X_Spherical=interp2(k_x,k_y,abs(f_x'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Y_Spherical=interp2(k_x,k_y,abs(f_y'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Z_Spherical=interp2(k_x,k_y,abs(f_z'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    
    % I think that this just sets a radial distance of 1000
    r=10000;
    C=j*(k0*exp(-j*k0*r))/(2*pi*r);
    Etheta=C*(f_X_Spherical.*cos(phi)+f_Y_Spherical.*sin(phi));
    Ephi=C*cos(theta).*(-f_X_Spherical.*sin(phi)+f_Y_Spherical.*cos(phi));
   
end