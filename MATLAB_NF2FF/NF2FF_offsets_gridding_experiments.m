%% NF2FF offsets experiment--plotting different griddings

clc;
clear;
close all;

%% loading in information

load('scanarray_pol2_h6mm-10_2_2009.mat'); % Measurement data
sdata2=sdata;
load('scanarray_pol1_h6mm-10_2_2009.mat'); % Measurement data
freq=sdata.freq; % Measured frequency points [Hz]

%% plotting the original gridding
M=sdata.xpoints; % Amount of samples in the x direction (along table, left to right)
N=sdata.ypoints; % Amount of samples in the y direction (across table, front to back)
delta_x=sdata.x_step/1000; % Sample spacing in the x direction [m]--I assume spacing is then in mm in data structure
delta_y=sdata.y_step/1000; % Sample spacing in the y direction [m]

a=delta_x*(M-1); % The length of the scanned area in the x direction [m]
b=delta_y*(N-1); % The length of the scanned area in the y direction [m]
x=[-a/2:a/(M-1):a/2]; % arrays of relevant x- and y-values created by spacing by dx/dy
y=[-b/2:b/(N-1):b/2]; % between -a/2 and a/2, -b/2 and b/2

[x_grid,y_grid]=meshgrid(x,y);

%plotting
figure;
scatter(x_grid,y_grid,10,"black",'filled');
daspect([1 1 1]);
xlabel('x (m)');
ylabel('y (m)');
title('the general grid');
hold on;

%% offset grid spacing

c=299792458; % Speed of light in vacuum [m/s]
broadcast_freq=94*10^9; % broadcast frequency in hertz
lambda_m=c/broadcast_freq;
nyquist_sampling_m=lambda_m/2; % necessary grid spacing in m
pct_offsets=[0.001,0.01,0.1];
offsets=nyquist_sampling_m*pct_offsets;
index=3
current_offset=offsets(index)

x_offset=x+current_offset;
[x_offset_grid,y_grid]=meshgrid(x_offset, y);

% plotting offset grid(method 1)
scatter(x_offset_grid,y,10,'red','filled');
hold on;

%% trying to produce equivalent grid spacing with m (no offset)--according to Balanis

% we have M, N, delta_x, delta_y
m_test=linspace(-M/2,M/2-1,M);
n_test=linspace(-N/2,N/2-1,N);

x_test=m_test*delta_x;
y_test=n_test*delta_y;

all(x_test==x)
all(y_test==y)

% from this, I found that the gridding in the MATLAB script doesn't exactly
% match up with that in Balanis. It is not M=delta_x*m between -M/2 and
% M/2-1, which would make the grid somewhat asymmetrical about 0.

%% reconfiguring the offset grid spacing to fit within Balanis' framework--trying out my new method

% original m and n (taken from the original script)
MI=4*M;%2^(ceil(log2(M))+1);
NI=4*N;%2^(ceil(log2(N))+1);
m=[-MI/2:1:MI/2-1];
n=[-NI/2:1:NI/2-1];
m_spaced=[-MI/2:4:MI/2-4];

% finding the offset between m and m_offset by using a version of m_offset
% calculated as x/delta_x
m_offset_preliminary=4*x_offset/delta_x; % this m_offset has the right overall offset from m but the wrong starting and ending points--multiplied by 4 to make it on the same scale as m
middle_point_m_spaced=m_spaced(floor(length(m_spaced)/2));
middle_point_m_matrix=repmat(middle_point_m_spaced,1,length(m_offset_preliminary));
[diff_m,index_diff_m]=min(abs(middle_point_m_matrix-m_offset_preliminary));

diff_m==current_offset/delta_x

m_offset1_test=m+current_offset/delta_x;
m_offset1=linspace(-MI/2+diff_m,MI/2-1+diff_m,MI);

all(m_offset1==m_offset1_test)

% generate and plot the k-space grid