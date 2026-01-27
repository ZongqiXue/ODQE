clc
clear all

M=5;
N=5;
buffer=2;
waittime=3;
parjob=12;
%Ajobtime=11;
%Pjobtime=[3,0,0, 2,1,0, 2,0,1, 2,0,0, 3,1,0];
Pjobtime=[4,1,0, 3,1,1, 3,1,1, 2,2,0, 3,2,0];
%6 8 8 6 7
% 11
%Ajobtime=[4,3,2];
Ajobtime=[6,4,2];
%Pjobtime=[M-2,M-1,M,M-3,M];
Time=100;

sini=struct('work',zeros((N+1)*waittime,N+2*buffer),...%0:finish
    'product',0,...
    'survival',0,...
    'remaining',0,...
    'score',1);

raw=[Pjobtime';Ajobtime'];
sraw=sini;
Pflow=[0,0,0,2,1,0, 2,0,1, 2,0,0, 3,1,0; 0,0,0,0,0,0, 2,0,1, 2,0,0, 3,1,0; 0,0,0,0,0,0,0,0,0, 2,0,0, 3,1,0; 0,0,0,0,0,0,0,0,0,0,0,0,3,1,0];
Aflow1=[3,3,1;2,2,1;2,1,1;0,1,0];
Aflow2=[4,1,2;4,1,1;2,1,1;2,0,0];
Aflow3=[0,3,2;0,3,1;0,2,1;0,1,0];
workflow1=[Pflow';Aflow1'];
workflow2=[Pflow';Aflow2'];
workflow3=[Pflow';Aflow3'];
workflow1=[repmat(raw,1, 1+buffer),workflow1,zeros(waittime*(N+1),buffer)];
workflow2=[repmat(raw,1, 1+buffer),workflow2,zeros(waittime*(N+1),buffer)];
workflow3=[repmat(raw,1, 1+buffer),workflow3,zeros(waittime*(N+1),buffer)];
sraw.work=[repmat(raw,1, 1+buffer),zeros(waittime*(N+1),N+buffer-1)];

actiontype=waittime*(2*(2*buffer+N)+1);

statenum=(2*buffer+N)*(N+1)*waittime;

SS(1:parjob)=sraw;

%load('SS.mat');
%load('NetS.mat');
%load('NetA.mat');
load('Net.mat');
%initialization
