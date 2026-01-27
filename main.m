clc

batchsize=20;

traintime=50;

velocityA=[];
velocityS=[];

learnRate=0.005;
c=1;
sigma=0.25;
tao=0.9;
momentum=0.9; 
decay = 0.01;
baserate=0.3;
QRv=20;

MCTSnum=500;
trainnum=1;
Timehorizon=200;
Deepth=20;
s_=sraw;

failureprobability=0.2;

%s_.survival=0;

rand('seed',sum(100*clock));

rem=[];
if ~exist('result_')
    result_=cell(0,0);
end

for tt=1:traintime
    fprintf('start train %d\n',tt);

    rand('seed',sum(100*clock));

    SS(1:parjob)=s_;

    for i=1:parjob
        eval(['Work=workflow',num2str(mod(i,3)+1),';']);
        SS(i).work=Work;
    end

    for i=1:parjob
        s0=SS(i);
        a = parcluster();
        eval(['j',num2str(i),'=createJob(a);']);
        eval(['t',num2str(i),'=createTask(j',num2str(i),',@traindata,1,{NetA,NetS,M,N,buffer,waittime,trainnum,c,Deepth,MCTSnum,s0,Timehorizon,sigma,tao,raw,QRv,failureprobability});']) ;
        eval(['submit(j',num2str(i),');']);
        %fprintf('fp%d\n',failureprobability);
        pause(0.1);
    end

    for i=1:parjob
        eval(['wait(j',num2str(i),');']);
        fprintf('finish job%d\n',i);
    end

    for i=1:parjob
        eval(['j=j',num2str(i),';']);
        if length(get(j.Tasks(1),'ErrorMessage'))==0
            eval(['outputs = fetchOutputs(j',num2str(i),');']);
            result_{end+1}=1.05*outputs{1};
        end
    end

    clear j1 j2 j3 j4 j5 j6 j7 j8 j9 j10 j11 j12 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 ta ts tv i    
    save result_ODQE.mat
    fprintf('finish train %d\n',tt);
end


