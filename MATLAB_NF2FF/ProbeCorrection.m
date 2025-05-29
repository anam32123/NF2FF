function [Etheta_compensated,Ephi_compensated]=ProbeCorrection(Etheta,Ephi,theta,phi,f)
% Notes: 
% (1) Etheta_compensated and Ephi_compensated are the compensated far-field components.
% (2) Etheta and Ephi are the uncompensated far-field components corresponding to the integral
% (3) The functions Eprobe_Ex and Eprobe_Ey return the far-field pattern of the TE10 waveguide probe
% oriented in the x- and y-directions.

c=299792458; % Speed of light in vacuum [m/s]
lambda=c/f;
k = 2*pi/lambda;
z0 = 0.006; % z position of probe (m)

a_probe = 0.1*0.0254; % probe's x-dimension (m)
b_probe = 0.05*0.0254; % probe's y-dimension (m)

% S11 magnitude (short circuit) in vertical polarization
S11short_V = 10^(-15/20);
% S11 magnitude (short circuit) in horizontal polarization
S11short_H = 10^(-25/20);
%K = T_v / T_h; % measurement system transfer function ratio
K = sqrt(S11short_V/S11short_H);

% Vertical polarization, probe E-field is oriented along y-direction
[Eprobe_theta_V Eprobe_phi_V] = Eprobe_Ey(theta,-phi,a_probe,b_probe,k);
Mv = cos(theta).*exp(j*k*cos(theta)*z0).*Etheta;
% Horizontal polarization, probe E-field is oriented along x-direction
[Eprobe_theta_H Eprobe_phi_H] = Eprobe_Ex(theta,-phi,a_probe,b_probe,k);
Mh = cos(theta).*exp(j*k*cos(theta)*z0).*Ephi;
% Compensate for different polarization measurement gain
Mv = Mv/K;
% Calculate compensated Etheta and Ephi
Etheta_compensated = (Mv.*Eprobe_phi_H-Mh.*Eprobe_phi_V )./(Eprobe_theta_V.*Eprobe_phi_H-Eprobe_phi_V.*Eprobe_theta_H);
Ephi_compensated = (Mh.*Eprobe_theta_V-Mv.*Eprobe_theta_H )./(Eprobe_theta_V.*Eprobe_phi_H-Eprobe_phi_V.*Eprobe_theta_H);

function [Eprobe_theta_H,Eprobe_phi_H]=Eprobe_Ex(theta,phi,a,b,k)

X=k*a/2*sin(theta).*sin(phi);
Y=k*b/2*sin(theta).*cos(phi);
        
Eprobe_theta_H=cos(phi).*cos(X)./(X.^2-(pi/2).^2).*sin(Y)./Y;
Eprobe_phi_H=-cos(theta).*sin(phi).*cos(X)./(X.^2-(pi/2).^2).*sin(Y)./Y;

function [Eprobe_theta_V,Eprobe_phi_V]=Eprobe_Ey(theta,phi,a,b,k)

X=k*a/2*sin(theta).*cos(phi);
Y=k*b/2*sin(theta).*sin(phi);
        
Eprobe_theta_V=sin(phi).*cos(X)./(X.^2-(pi/2).^2).*sin(Y)./Y;
Eprobe_phi_V=cos(theta).*cos(phi).*cos(X)/(X.^2-(pi/2).^2).*sin(Y)./Y;
