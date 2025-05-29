% MATLAB DAQ Script for the Agilent 8720ES (LINFREQ SWEEP)
% Koen van Caekenberghe

function []=AGILENT8720ES_LINFREQ(CHANNEL,FILENAME);

g = visa('agilent', 'GPIB0::16::0::INSTR');
set(g,'InputBufferSize',100000);
fopen(g)
fprintf(g,'*RST');

try

    fprintf(g,'INTD');
    fprintf(g,sprintf('TITF1 "%s"','FEMCIM'));
    fprintf(g,'LOAD1');
    pause(5);

    I_16 = N5767A_16VSUPPLY_READ;
    I_48 = N5767A_48VSUPPLY_READ;
    
    [POIN_STRING, COUNT, MSG] = query(g,'POIN?;');
    [POWE_STRING, COUNT, MSG] = query(g,'POWE?;');
    [START_STRING, COUNT, MSG] = query(g,'STAR?;');
    [STOP_STRING, COUNT, MSG] = query(g,'STOP?;');
    
    fprintf('\tMeasuring S11...\n');
    [S11_STRING, COUNT, MSG] = query(g,'S11;LOGM;CONT;FORM4;OUTPDATA');
    fprintf('\tMeasuring S12...\n');
    [S12_STRING, COUNT, MSG] = query(g,'S12;LOGM;CONT;FORM4;OUTPDATA');
    fprintf('\tMeasuring S21...\n'); 
    [S21_STRING, COUNT, MSG] = query(g,'S21;LOGM;CONT;FORM4;OUTPDATA');
    fprintf('\tMeasuring S22...\n');  
    [S22_STRING, COUNT, MSG] = query(g,'S22;LOGM;CONT;FORM4;OUTPDATA');
   
catch
    
    sprintf('AGILENT8720ES_LINFREQ CRASHED...')
    fclose(g); 

end

fclose(g);

POIN_AS_SET = str2num(POIN_STRING);
POWE_AS_SET = str2num(POWE_STRING);
START_AS_SET = str2num(START_STRING)/1000000000;
STOP_AS_SET = str2num(STOP_STRING)/1000000000;
FREQ = [START_AS_SET : (STOP_AS_SET-START_AS_SET)/(POIN_AS_SET-1) : STOP_AS_SET]';

S11 = str2num(S11_STRING);
S12 = str2num(S12_STRING);
S21 = str2num(S21_STRING);
S22 = str2num(S22_STRING);

S11_MAGNITUDE = 20*log10(abs(S11(:,1)+i*S11(:,2)));
S11_PHASE = 180/pi*atan2(S11(:,2),S11(:,1));
S12_MAGNITUDE = 20*log10(abs(S12(:,1)+i*S12(:,2)));
S12_PHASE = 180/pi*atan2(S12(:,2),S12(:,1));
S21_MAGNITUDE = 20*log10(abs(S21(:,1)+i*S21(:,2)));
S21_PHASE = 180/pi*atan2(S21(:,2),S21(:,1));
S22_MAGNITUDE = 20*log10(abs(S22(:,1)+i*S22(:,2)));
S22_PHASE = 180/pi*atan2(S22(:,2),S22(:,1));

clear DATA;
DATA = csvread('c:\\FEMCIM\\NWA_S12.csv');
S12_CORRECTION = DATA(:,str2num(CHANNEL)+1);
clear DATA;
DATA = csvread('c:\\FEMCIM\\NWA_S21.csv');
S21_CORRECTION = DATA(:,str2num(CHANNEL)+1);

FIG1 = figure;
subplot(2,2,1)
plot(FREQ,S11_MAGNITUDE,'k-');
hold on;    
HRS=plot([8.5 10.5],[-15 -15]);
set(HRS,'Color','k','LineWidth',2.5);
hold off;
xlabel('Frequency (GHz)');
ylabel('S_{11} (dB)');
title(sprintf('I_{16V} = %i mA, I_{48V} = %i mA, P = %i dBm',1000*I_16,1000*I_48,POWE_AS_SET));
subplot(2,2,2)
plot(FREQ,S12_MAGNITUDE,'k--',FREQ,S12_MAGNITUDE - S12_CORRECTION,'k-');
hold on;    
HRS=plot([8.5 10.5],[35-POWE_AS_SET 35-POWE_AS_SET]);
set(HRS,'Color','k','LineWidth',2.5);
hold off;
xlabel('Frequency (GHz)');
ylabel('S_{12} (dB)');
legend('MEASURED','DEEMBEDDED','Location','Best');
subplot(2,2,3)
plot(FREQ,S21_MAGNITUDE,'k--',FREQ,S21_MAGNITUDE - S21_CORRECTION,'k-');
hold on;    
HRS=plot([8.5 10.5],[23.8 23.8],[8.5 10.5],[28.3 28.3]);
set(HRS,'Color','k','LineWidth',2.5);
hold off;
xlabel('Frequency (GHz)');
ylabel('S_{21} (dB)');
legend('MEASURED','DEEMBEDDED','Location','Best');
subplot(2,2,4)
plot(FREQ,S22_MAGNITUDE,'k-');
hold on;    
HRS=plot([8.5 10.5],[-15 -15]);
set(HRS,'Color','k','LineWidth',2.5);
hold off;
xlabel('Frequency (GHz)');
ylabel('S_{22} (dB)');
print(FIG1,'-dmeta',[FILENAME '_dB']);
saveas(FIG1,[FILENAME '_dB'],'fig');

FIG2 = figure;
subplot(2,2,1)
plot(FREQ,S11_PHASE,'k-');
xlabel('Frequency (GHz)');
ylabel('S_{11} (°)');
title(sprintf('I_{16V} = %i mA, I_{48V} = %i mA, P = %i dBm',1000*I_16,1000*I_48,POWE_AS_SET));
subplot(2,2,2)
plot(FREQ,S12_PHASE,'k-');
xlabel('Frequency (GHz)');
ylabel('S_{12} (°)');
subplot(2,2,3)
plot(FREQ,S21_PHASE,'k-');
xlabel('Frequency (GHz)');
ylabel('S_{21} (°)');
subplot(2,2,4)
plot(FREQ,S22_PHASE,'k-');
xlabel('Frequency (GHz)');
ylabel('S_{22} (°)');
print(FIG2,'-dmeta',[FILENAME '_DEG']);
saveas(FIG2,[FILENAME '_DEG'],'fig');

close(FIG1);
close(FIG2);

csvwrite(strcat(FILENAME,'.csv'),[FREQ S11_MAGNITUDE S11_PHASE S12_MAGNITUDE S12_PHASE S21_MAGNITUDE S21_PHASE S22_MAGNITUDE S22_PHASE S11_MAGNITUDE S11_PHASE S12_MAGNITUDE - S12_CORRECTION S12_PHASE S21_MAGNITUDE - S21_CORRECTION S21_PHASE S22_MAGNITUDE S22_PHASE]);
