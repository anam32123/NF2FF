%hp8510Cdemo      Demonstration code for control of an HP 8510C
%
% This code handles some basic communication with an HP 8510C, which is a 20 GHz network
%  analyzer.  This code performs the range of tasks that some specific users
%  will likely encounter.  They just need to add appropriate HP commands
%  for any additional functionality 
% I tried to show them useful MATLAB tricks for num2str/str2num conversion.
% Please keep in mind that this code was developed "on the fly" - so it is
%  likely quite unorderly!
%
% 5 Basic Steps for Instrument Control
%1.  Create the gpib object
%2.  Configure the gpib object
%3.  Connect to the object
%4.  Communicate with the instrumetn
%5.  Clean up

% Scott Hirsch
% shirsch@mathworks.com

%1.  Create the gpib object
BoardNumber = 0;        %The GPIB controller card number
DeviceNumber = 16;      %This is device settable
g=gpib('ni',BoardNumber,DeviceNumber);      %Create object

%Alternate: with serial port
% com_port = 'COM1';		%Use your serial port
% g = serial(com_port);		%

%2.  Configure the gpib object
%Large input buffer.  This makes room for bringing in a lot of data.
set(g,'InputBufferSize',200000);
%g.InputBufferSize = 200000;     %Alternate syntax

%3. Connect to the object
fopen(g)                %Open communication

%4.  Communicate with the instrumetn
%Calibration
fprintf(g,'CORRON')     %Turn on calibration
fprintf(g,'CALS1')      %Select calibration #1
%Insert your HT Basic commands here ...
% fprintf(g,'HT BASIC COMMAND TO INSTRUMENT');

%An example of how to incorporate user input into a command
%Read marker value
fo = input('Select marker frequency (GHz) ');
fprintf(g,['MARK1 ' num2str(fo) 'GHz']);        %num2str(fo) converts fo from a number to a string

%Configure the instrument
fprintf(g,'FORM4');     %Set data output format to ASCII

%Set start and stop sweep frequency
fi=2;           %Start frequency, GHz
fe=20;          %Stop frequency, GHz
fprintf(g,['STAR ' num2str(fi) ' GHz; STOP ' num2str(fe) ' GHz;']);

%Get some data.  
npoints=201;        %Number of data points
fprintf(g,['CHAN1; S11; POIN' num2str(npoints) ';']);
fprintf(g,'SING;');     %Trigger a single sweep
fprintf(g,'LOGM;');     %LogMeg
fprintf(g,'CONT;');     %Back to continuous sweep


fprintf(g,['CHAN1; S11; POIN' num2str(npoints) ';']);
fprintf(g,'SING;OUTPDATA');
%while BUSY, end;
pause(10)
%OK, I got really lazy here and put a pause in to wait for the
%  data transfer to finish.  You should query the instrument inside
%  a while loop which exists when the instrument is ready.

d=fscanf(g);		   %Get the data
m=strread(d,'%f','delimiter',',');
mR11=m(1:2:end);      %Extract real component
mI11=m(2:2:end);      %Extract imaginary component

fprintf(g,['CHAN2; S21; POIN' num2str(npoints) ';']);
fprintf(g,'SING;OUTPDATA');
%while BUSY, end;
pause(10)

d=fscanf(g);
m=strread(d,'%f','delimiter',',');
mR12=m(1:2:end);      %Extract real component
mI12=m(2:2:end);      %Extract imaginary component


%Data analysis
freq = linspace(fi,fe,npoints);

figure;
%Some sample visualization
subplot(211);plot(freq,[mR11 mR12]);
legend('11','12');title('Real');
subplot(212);plot(freq,[mI11 mI12]);
legend('11','12');title('Imag');xlabel('Frequency (GHz)');


%5.  Clean up
fclose(g)               %Close communication
delete(g)               %Delete object