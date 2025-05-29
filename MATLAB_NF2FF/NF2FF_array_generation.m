% In this script, I'm generating a arrays of the near-field data ad all the
% intermediate steps for the purposes of comparison and subtraction wiht
% the Python results.

%%
clc;
close all;
clear all;

%% loading in data and relevant variables
c=299792458; % Speed of light in vacuum [m/s]
% loads in two different measurement data structures
load('scanarray_pol2_h6mm-10_2_2009.mat'); % Measurement data
sdata2=sdata;
load('scanarray_pol1_h6mm-10_2_2009.mat'); % Measurement data
freq=sdata.freq; % Measured frequency points [Hz]
N=length(freq);
f_start=freq(1); % first frequency
f_stop=freq(N);  % last frequency
df=(f_stop-f_start)/(N-1); 
dt=1/(N*df);
t=(0:N-1)*dt;
x=c*t;

%% Plotting initial data

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

%%

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
k_X_Rectangular=2*pi*m/(MI*dx);
k_Y_Rectangular=2*pi*n/(NI*dy);
[k_Y_Rectangular_Grid,k_X_Rectangular_Grid] = meshgrid(k_Y_Rectangular,k_X_Rectangular);

% creating theta and phi grids from -pi/2 to pi/2 and 0 to pi
% (respectively)
dtheta=0.05;
dphi=0.05;
theta=[-pi/2+dtheta:dtheta:pi/2-dtheta];
phi=[0+dphi:dphi:pi-dphi];
[theta,phi]=meshgrid(theta,phi);

f_X_Rectangular_array=[];
f_Y_Rectangular_array=[];
f_Z_Rectangular_array=[];
f_X_Spherical_array=[];
f_Y_Spherical_array=[];
f_Z_Spherical_array=[];
Etheta_array=[];
Ephi_array=[];
NF_X_Complex_array=[];
NF_Y_Complex_array=[];

Index = 1;
for f_Index = 1:1:201
    
    close all;
    f=freq(f_Index);
    lambda0=c/f;
    k0=2*pi/lambda0;
    k_Z_Rectangular_Grid = sqrt(k0^2-k_X_Rectangular_Grid.^2-k_Y_Rectangular_Grid.^2);
      
    for iy=1:1:N
       for ix=1:1:M
           NF_X_Complex(ix,iy)=sdata.s21{iy,ix}(f_Index);
           NF_Y_Complex(ix,iy)=sdata2.s21{iy,ix}(f_Index);
       end
    end

    NF_X_Complex_array(:,:,f_Index)=NF_X_Complex;
    NF_Y_Complex_array(:,:,f_Index)=NF_Y_Complex;

    NF_X_Magnitude = 20*log10(abs(NF_X_Complex));
    NF_Y_Magnitude = 20*log10(abs(NF_Y_Complex));

    % See equations (16-7a) and (16-7b) in Balanis
    f_X_Rectangular=ifftshift(ifft2(NF_X_Complex,MI,NI));
    f_Y_Rectangular=ifftshift(ifft2(NF_Y_Complex,MI,NI));
    f_Z_Rectangular=-(f_X_Rectangular.*k_X_Rectangular_Grid+f_Y_Rectangular.*k_Y_Rectangular_Grid)./k_Z_Rectangular_Grid;

    
    f_X_Rectangular_array(:,:,f_Index) = f_X_Rectangular;
    f_Y_Rectangular_array(:,:,f_Index) = f_Y_Rectangular;
    f_Z_Rectangular_array(:,:,f_Index) = f_Z_Rectangular;

    f_X_Rectangular_Magnitude=20*log10(abs(f_X_Rectangular));
    f_Y_Rectangular_Magnitude=20*log10(abs(f_Y_Rectangular));
    f_Z_Rectangular_Magnitude=20*log10(abs(f_Z_Rectangular));
    
    f_X_Spherical=interp2(k_X_Rectangular,k_Y_Rectangular,abs(f_X_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Y_Spherical=interp2(k_X_Rectangular,k_Y_Rectangular,abs(f_Y_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    f_Z_Spherical=interp2(k_X_Rectangular,k_Y_Rectangular,abs(f_Z_Rectangular'),k0*sin(theta).*cos(phi),k0*sin(theta).*sin(phi),'spline');
    
    f_X_Spherical_array(:,:,f_Index)=f_X_Spherical;
    f_Y_Spherical_array(:,:,f_Index)=f_Y_Spherical;
    f_Z_Spherical_array(:,:,f_Index)=f_Z_Spherical;

    r=10000;
    C=j*(k0*exp(-j*k0*r))/(2*pi*r);
    Etheta=C*(f_X_Spherical.*cos(phi)+f_Y_Spherical.*sin(phi));
    Ephi=C*cos(theta).*(-f_X_Spherical.*sin(phi)+f_Y_Spherical.*cos(phi));

    Etheta_array(:,:,f_Index)=Etheta;
    Ephi_array(:,:,f_Index)=Ephi;

    %[Etheta,Ephi]=ProbeCorrection(Etheta,Ephi,theta,phi,f);
    W=1/(2*120*pi).*(Etheta.*conj(Etheta)+Ephi.*conj(Ephi));

	U = (abs(Etheta).^2 + abs(Ephi).^2);
    
	% Calculation of radiated power through numerical integration
	e_theta = [1 4 repmat([2 4], 1, floor(length(theta(1,:))/2) - 1) 1];
    e_phi = [1 4 repmat([2 4], 1, floor(length(phi(:,1))/2) - 1) 1];
    P = dphi*dtheta*sum(sum(U.*(e_theta'*e_phi).*abs(sin(theta))))/9;
	
    D = 4*pi*U/P;
    U_Co = (abs(Etheta.*cos(phi)-Ephi.*sin(phi))).^2;
    U_Cross = (abs(Etheta.*sin(phi)+Ephi.*cos(phi))).^2;
    D_Co = 4*pi*U_Co/P;
    D_Cross = 4*pi*U_Cross/P;

    W_Size=size(W);
    EPLANE(Index,:)=10*log10(W(floor(W_Size(2)/2),:))-max(10*log10(W(floor(W_Size(2)/2),:)));
    HPLANE(Index,:)=10*log10(W(1,:))-max(10*log10(W(1,:)));

    Index=Index+1;

end

pcolor(x,y,angle(NF_X_Complex_array(:,:,201)))

%save RESULTS EPLANE HPLANE Hologram_Slots_X_Magnitude Hologram_Slots_Y_Magnitude Hologram_Slots_X_Phase Hologram_Slots_Y_Phase freq
