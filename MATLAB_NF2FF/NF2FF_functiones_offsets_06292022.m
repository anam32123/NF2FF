%% This script consolidates the whole NF to FF transformation into one function for ease of use and plotting, to make the main body of the script significantly shorter
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
pct_offsets=[0.001,0.01,0.1];
offsets=nyquist_sampling_m.*pct_offsets;
x_offset1=x+offsets(1);
x_offset2=x+offsets(2);
x_offset3=x+offsets(3);

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
m_offset1_spaced=m+offsets(1)/dx; % see explanation page 40 of lab notebook
% defining offset k grid
k_X_Rectangular_Offset1=2*pi.*m_offset1_spaced/(MI*dx);
k_Y_Rectangular_Offset1=2*pi.*n/(NI*dy);
[k_Y_Rectangular_Offset1_Grid,k_X_Rectangular_Offset1_Grid] = meshgrid(k_Y_Rectangular_Offset1,k_X_Rectangular_Offset1);

% offset 2 version
m_offset2_spaced=m+offsets(2)/dx; % see explanation page 40 of lab notebook
k_X_Rectangular_Offset2=2*pi.*m_offset2_spaced/(MI*dx);
k_Y_Rectangular_Offset2=2*pi.*n/(NI*dy);
[k_Y_Rectangular_Offset2_Grid,k_X_Rectangular_Offset2_Grid] = meshgrid(k_Y_Rectangular_Offset2,k_X_Rectangular_Offset2);

% offset version 3
m_offset3_spaced=m+offsets(3)/dx; % see explanation page 40 of lab notebook
k_X_Rectangular_Offset3=2*pi.*m_offset3_spaced/(MI*dx);
k_Y_Rectangular_Offset3=2*pi.*n/(NI*dy);
[k_Y_Rectangular_Offset3_Grid,k_X_Rectangular_Offset3_Grid] = meshgrid(k_Y_Rectangular_Offset3,k_X_Rectangular_Offset3);

% creating theta and phi grids from -pi/2 to pi/2 and 0 to pi
% (respectively)
dtheta=0.05;
dphi=0.05;
theta=[-pi/2+dtheta:dtheta:pi/2-dtheta];
phi=[0+dphi:dphi:pi-dphi];
theta_vector=theta;
phi_vector=phi;
[theta,phi]=meshgrid(theta,phi);

%% calculations

Index = 1;
for f_Index = 201:1:201 %1:1:N Only does this at the 201st frequency?? why--maybe only interested in one frequency and you can adjust it
    
    f=freq(f_Index);
    lambda0=c/f;
    k0=2*pi/lambda0;
    k_Z_Rectangular_Grid = sqrt(k0^2-k_X_Rectangular_Grid.^2-k_Y_Rectangular_Grid.^2);

    % offset versions of k_z
    k_Z_Rectangular_Offset1_Grid=sqrt(k0^2-k_X_Rectangular_Offset1_Grid.^2-k_Y_Rectangular_Offset1_Grid.^2);
    k_Z_Rectangular_Offset2_Grid=sqrt(k0^2-k_X_Rectangular_Offset2_Grid.^2-k_Y_Rectangular_Offset2_Grid.^2);
    k_Z_Rectangular_Offset3_Grid=sqrt(k0^2-k_X_Rectangular_Offset3_Grid.^2-k_Y_Rectangular_Offset3_Grid.^2);

    %% loading in and plotting all NF data at given frequency

    for iy=1:1:N
       for ix=1:1:M 
           NF_X_Complex(ix,iy)=sdata.s21{iy,ix}(f_Index);
           NF_Y_Complex(ix,iy)=sdata2.s21{iy,ix}(f_Index);
       end
    end

    NF_X_Phase=angle(NF_X_Complex);
    NF_Y_Phase=angle(NF_Y_Complex);

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

    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset1,f_X_Rectangular_Magnitude,...
        f_Y_Rectangular_Magnitude,f_Z_Rectangular_Offset1_Magnitude]...
        =NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Offset1_Grid,...
        k_Y_Rectangular_Offset1_Grid,k_Z_Rectangular_Offset1_Grid);
    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset2,f_X_Rectangular_Magnitude,...
        f_Y_Rectangular_Magnitude,f_Z_Rectangular_Offset2_Magnitude]...
        =NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Offset2_Grid,...
        k_Y_Rectangular_Offset2_Grid,k_Z_Rectangular_Offset2_Grid);
    [f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset3,f_X_Rectangular_Magnitude,...
        f_Y_Rectangular_Magnitude,f_Z_Rectangular_Offset3_Magnitude]...
        =NFtoFourierSpace(NF_X_Complex,NF_Y_Complex,k_X_Rectangular_Offset3_Grid,...
        k_Y_Rectangular_Offset3_Grid,k_Z_Rectangular_Offset3_Grid);


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

%     fz_errors=complex_error(f_Z_Rectangular,f_Z_Rectangular_Offset,k_X_Rectangular,k_Y_Rectangular,'k_{x} {m^{-1}}','k_{y} {m_{-1}}');

%% converting to the far field

    [Etheta,Ephi]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular,theta,phi,k_X_Rectangular,k_Y_Rectangular,k0);
    [Etheta_Offset1,Ephi_Offset1]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset1,theta,phi,k_X_Rectangular_Offset1,k_Y_Rectangular_Offset1,k0);
    [Etheta_Offset2,Ephi_Offset2]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset2,theta,phi,k_X_Rectangular_Offset2,k_Y_Rectangular_Offset2,k0);
    [Etheta_Offset3,Ephi_Offset3]=FouriertoFF(f_X_Rectangular,f_Y_Rectangular,f_Z_Rectangular_Offset3,theta,phi,k_X_Rectangular_Offset3,k_Y_Rectangular_Offset3,k0);

    [Etheta_Test,Ephi_Test,f_X_Test,f_Y_Test,f_Z_Test]=NF2FF(sdata,sdata2,f_Index,k_X_Rectangular,k_Y_Rectangular,M,N,theta,phi);

    fprintf('Error on E_theta\n');
    Etheta1_errors=complex_error(Etheta,Etheta_Offset1,theta_vector,phi_vector,'theta','phi');
    fprintf('Error on E_phi\n');
    Ephi1_errors=complex_error(Ephi,Ephi_Offset1,theta_vector,phi_vector,'theta','phi');
    Etheta2_errors=complex_error(Etheta,Etheta_Offset2,theta_vector,phi_vector,'theta','phi');
    Ephi2_errors=complex_error(Ephi,Ephi_Offset2,theta_vector,phi_vector,'theta','phi');
    Etheta3_errors=complex_error(Etheta,Etheta_Offset3,theta_vector,phi_vector,'theta','phi');
    Ephi3_errors=complex_error(Ephi,Ephi_Offset3,theta_vector,phi_vector,'theta','phi');
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
    errors=[Etheta1_errors,Etheta2_errors,Etheta3_errors,Ephi1_errors,Ephi2_errors,Ephi3_errors];
    figure;
    for index=1:6
        subplot(2,3,index);
        plot1=pcolor(theta,phi,errors(index).weighted_percent_difference_amplitude);
        daspect([1 1 1]);
        plot1.LineWidth=0.5;
        caxis([-Inf 1]);
        if index < 4
            title('Weighted percent difference in E_\theta: amplitude');
        else
            title('Weighted percent difference in E_\phi: amplitude');
        end
        xlabel('Theta');
        ylabel('Phi');
        colorbar;
    end
	
    % scatter plot of average percent difference in amplitude vs percentage
    % offset
    for index=1:3
        mean_pct_diff_amplitudes_Etheta(index)=errors(index).weighted_percent_difference_mean_amplitude;
    end
    i=1;
    for index=4:6
        mean_pct_diff_amplitudes_Ephi(i)=errors(index).weighted_percent_difference_mean_amplitude;
        i=i+1;
    end
    figure;
    semilogx(pct_offsets,mean_pct_diff_amplitudes_Etheta,'b.','MarkerSize',15);
    hold on;
    semilogx(pct_offsets,mean_pct_diff_amplitudes_Ephi,'r.','MarkerSize',15);
    title('Mean percent difference vs offset');
    xlabel('Offset as percentage of Nyquist spacing');
    ylabel('Mean percent difference between offset and original field');
    legend('E_\theta','E_\phi');


%% plots

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
%     % 3D rendering of antenna pattern
%     figure;
%     subplot(2,2,1);
%     sphere3d(20*log10(abs(Etheta))-max(max(20*log10(abs(Etheta')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
%     title('|E_{\theta}| (dBi)');
%     subplot(2,2,2);
%     sphere3d(20*log10(abs(Ephi))-max(max(20*log10(abs(Ephi')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
%     title('|E_{\phi}| (dBi)');
%     % offset portion
%     subplot(2,2,3);
%     sphere3d(20*log10(abs(Etheta_Offset))-max(max(20*log10(abs(Etheta_Offset')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
%     title('|E_{\theta}| (dBi) with offset');
%     subplot(2,2,4);
%     sphere3d(20*log10(abs(Ephi_Offset))-max(max(20*log10(abs(Ephi_Offset')))),0,pi,-pi/2,pi/2,1,1,'surf','spline');
%     title('|E_{\phi}| (dBi) with offset');

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
    std_error_real=std(error_real,0,'all');
    std_error_im=std(error_im,0,'all');
    error_metrics.std_error=complex(std_error_real,std_error_im);
    error_metrics.std_error_amplitude=abs(error_metrics.std_error);
    error_metrics.std_error_phase=angle(error_metrics.std_error);
    pct_diff_std_error_real=std(weighted_percent_difference_real,0,'all');
    pct_diff_std_error_im=std(weighted_percent_difference_im,0,'all');
    error_metrics.pct_diff_std_error=complex(pct_diff_std_error_real,pct_diff_std_error_im);
    error_metrics.pct_diff_std_error_amplitude=abs(error_metrics.pct_diff_std_error);
    error_metrics.pct_diff_std_error_phase=angle(error_metrics.pct_diff_std_error);

    % subtracting abs(b1)-abs(b2)--an error that is more of a magnitude
    error_metrics.error_magnitudes=abs(original_field)-abs(offset_field);
    error_metrics.mean_error_magnitudes=mean(error_metrics.error_magnitudes,'all','omitnan');
    error_metrics.pct_error_magnitudes_mean=error_metrics.mean_error_magnitudes./((abs(original_field)+abs(offset_field))/2);

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
    sphere3d(error_amplitude,0,pi,-pi/2,pi/2,1,1,'surf','cubic');
    title('Spherical representation of error in amplitude');
    subplot(1,2,2);
    sphere3d(error_phase,0,pi,-pi/2,pi/2,1,1,'surf','cubic');
    title('Spherical representation of error in phase');

    figure;
    ax1=subplot(1,2,1);
    plot1=pcolor(x,y,error_amplitude);
    daspect([1 1 1]);
    plot1.LineWidth=0.5;
    title('Difference between offset and original field: amplitude');
    xlabel(label_1);
    ylabel(label_2);
    colorbar;
    ax2=subplot(1,2,2);
%     scatter(x_grid,y_grid,400/length(x),'filled');
    plot2=pcolor(x,y,error_phase);
    daspect([1 1 1]);
    plot2.LineWidth=0.5;
    title('Difference between offset and original field: phase');
    xlabel(label_1);
    ylabel(label_2);
    colorbar;

    % plotting percent difference over the entire 2D grid
    figure;
%     ax1=subplot(1,2,1);
    plot1=pcolor(x,y,error_metrics.weighted_percent_difference_amplitude);
    daspect([1 1 1]);
    plot1.LineWidth=0.5;
    caxis([-Inf 1]);
    title('Weighted percent difference: amplitude');
    xlabel(label_1);
    ylabel(label_2);
    colorbar;
%     ax2=subplot(1,2,2);
% %     scatter(x_grid,y_grid,400/length(x),'filled');
%     plot2=pcolor(x,y,error_metrics.weighted_percent_difference_phase);
%     daspect([1 1 1]);
%     plot2.LineWidth=0.5;
%     caxis([-Inf 1]);
%     title('Weighted percent difference: phase');
%     xlabel(label_1);
%     ylabel(label_2);
%     colorbar;

    figure;
    sphere3d(error_metrics.weighted_percent_difference_amplitude,0,pi,-pi/2,pi/2,1,1,'surf','spline');

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