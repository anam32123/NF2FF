% MATLAB Movie Maker for OFDM Frequency Scanning Radar
% Koen Van Caekenberghe (vcaeken@umich.edu) - 09-11-2008

clc;
close all;
clear all;

load RESULTS;
fig1 = figure;
Size = get(fig1,'Position');
Size(1:2) = [0 0];
Movie = avifile('RESULTS','fps',25,'quality',100);
set(fig1,'NextPlot','replacechildren');
for f_Index=1:201
    subplot(1,2,1);
    plot(180/pi*[-pi/2+0.1:0.05:pi/2-0.1],EPLANE(f_Index,:)-max(EPLANE(f_Index,:)),[-25 -25],[-30 0],'k:',[25 25],[-30 0],'k:',[-90 90],[-13.6 -13.6],'k:');
    title(sprintf('f = %f GHz\nE-Plane',freq(f_Index)/1000000000));
    xlabel('INPUT \leftarrow \theta (Deg) \rightarrow LOAD');
    ylabel('Directivity (dBi)');
    set(gca,'XLim',[-90 90]);
    set(gca,'YLim',[-30 0]);
    subplot(1,2,2);
    plot(180/pi*[-pi/2+0.1:0.05:pi/2-0.1],HPLANE(f_Index,:)-max(HPLANE(f_Index,:)));
    title('H-Plane');
    xlabel('\theta (Deg)');
    ylabel('Directivity (dBi)');
    set(gca,'XLim',[-90 90]);
    set(gca,'YLim',[-30 0]);
    Movie = addframe(Movie,getframe(fig1,Size));
end
Movie = close(Movie);
