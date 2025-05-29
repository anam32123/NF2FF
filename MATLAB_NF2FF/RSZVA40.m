% MATLAB DAQ Script for the Rohde & Schwarz ZVA40 Vector Network Analyzer
% Koen van Caekenberghe

clear all;
close all;
clc;
% INSTRFIND

g = visa('agilent', 'GPIB0::20::0::INSTR');
set(g,'InputBufferSize',10000);
set(g,'TimeOut',1); 
fopen(g)	
% IDN = query(g, '*IDN?')
% query(g,'*RST')
 
%START=8;
%STOP=12;
NUMBEROFPOINTS=401;

try
    
    %[DATA, COUNT, MSG] = query(g,sprintf('SENSE1:FREQUENCY:START %f GHZ; STOP %f GHZ',START,STOP));
    [FREQ_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA:STIMULUS?');
    %[DATA, COUNT, MSG] = query(g,'DISPLAY:WINDOW1:TRACE1:Y:SCALE:AUTO ONCE');

    fprintf('\nMeasuring S11...\n');
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:PARAMETER:SDEFINE ''TRC1'', ''S11''');
    
    fprintf('\tMagnitude...\n');    
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT MAGNITUDE');
    [S11_MAGNITUDE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    fprintf('\tPhase...\n');   
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT PHASE');
    [S11_PHASE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');
    
    %fprintf('\tSmith chart...\n');      
    %[DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT SMITH');
    %[S11_SMITH_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    fprintf('Measuring S12...\n');
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:PARAMETER:SDEFINE ''TRC2'', ''S12''');

    fprintf('\tMagnitude...\n');    
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT MAGNITUDE');
    [S12_MAGNITUDE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    fprintf('\tPhase...\n');  
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT PHASE');
    [S12_PHASE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    %fprintf('\tSmith chart...\n');      
    %[DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT SMITH');
    %[S12_SMITH_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');
    
    fprintf('Measuring S21...\n');
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:PARAMETER:SDEFINE ''TRC3'', ''S21''');

    fprintf('\tMagnitude...\n');    
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT MAGNITUDE');
    [S21_MAGNITUDE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    fprintf('\tPhase...\n');  
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT PHASE');
    [S21_PHASE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    %fprintf('\tSmith chart...\n');      
    %[DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT SMITH');
    %[S21_SMITH_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    fprintf('Measuring S22...\n');
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:PARAMETER:SDEFINE ''TRC4'', ''S22''');
    
    fprintf('\tMagnitude...\n');    
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT MAGNITUDE');
    [S22_MAGNITUDE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');

    fprintf('\tPhase...\n');  
    [DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT PHASE');
    [S22_PHASE_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');
    
    %fprintf('\tSmith chart...\n');      
    %[DATA, COUNT, MSG] = query(g,'CALCULATE1:FORMAT SMITH');
    %[S22_SMITH_STRING, COUNT, MSG] = query(g,'CALCULATE1:DATA? FDATA');
    
catch
    sprintf('CRASHED...')
    fclose(g); 
end
fclose(g); 
fprintf('\n'); 

FREQ = str2num(FREQ_STRING);

S11_MAGNITUDE = str2num(S11_MAGNITUDE_STRING);
S11_PHASE = str2num(S11_PHASE_STRING);
%S11_SMITH = reshape(str2num(S11_SMITH_STRING),2,NUMBEROFPOINTS)';

S12_MAGNITUDE = str2num(S12_MAGNITUDE_STRING);
S12_PHASE = str2num(S12_PHASE_STRING);
%S12_SMITH = reshape(str2num(S12_SMITH_STRING),2,NUMBEROFPOINTS)';

S21_MAGNITUDE = str2num(S21_MAGNITUDE_STRING);
S21_PHASE = str2num(S21_PHASE_STRING);
%S21_SMITH = reshape(str2num(S21_SMITH_STRING),2,NUMBEROFPOINTS)';

S22_MAGNITUDE = str2num(S22_MAGNITUDE_STRING);
S22_PHASE = str2num(S22_PHASE_STRING);
%S22_SMITH = reshape(str2num(S22_SMITH_STRING),2,NUMBEROFPOINTS)';

figure;
subplot(2,2,1)
plot(FREQ/10^9,S11_MAGNITUDE);
xlabel('Frequency (GHz)');
ylabel('S_{11} (dB)');
subplot(2,2,2)
plot(FREQ/10^9,S12_MAGNITUDE);
xlabel('Frequency (GHz)');
ylabel('S_{12} (dB)');
subplot(2,2,3)
plot(FREQ/10^9,S21_MAGNITUDE);
xlabel('Frequency (GHz)');
ylabel('S_{21} (dB)');
subplot(2,2,4)
plot(FREQ/10^9,S22_MAGNITUDE);
xlabel('Frequency (GHz)');
ylabel('S_{22} (dB)');

figure;
subplot(2,2,1)
plot(FREQ/10^9,S11_PHASE);
xlabel('Frequency (GHz)');
ylabel('S_{11} (Deg)');
subplot(2,2,2)
plot(FREQ/10^9,S12_PHASE);
xlabel('Frequency (GHz)');
ylabel('S_{12} (Deg)');
subplot(2,2,3)
plot(FREQ/10^9,S21_PHASE);
xlabel('Frequency (GHz)');
ylabel('S_{21} (Deg)');
subplot(2,2,4)
plot(FREQ/10^9,S22_PHASE);
xlabel('Frequency (GHz)');
ylabel('S_{22} (Deg)');

%figure;
%subplot(2,2,1)
%polar(S11_SMITH(:,2),S11_SMITH(:,1));
%subplot(2,2,2)
%polar(S12_SMITH(:,2),S12_SMITH(:,1));
%subplot(2,2,3)
%polar(S21_SMITH(:,2),S21_SMITH(:,1));
%subplot(2,2,4)
%polar(S22_SMITH(:,2),S22_SMITH(:,1));

csvwrite('RX.csv',[FREQ' S11_MAGNITUDE' S11_PHASE' S12_MAGNITUDE' S12_PHASE' S21_MAGNITUDE' S21_PHASE' S22_MAGNITUDE' S22_PHASE']);
