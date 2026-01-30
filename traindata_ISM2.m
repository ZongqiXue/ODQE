function [Ts,TA,TV]=traindata(NetA,NetS,M,N,buffer,waittime,trainnum,c,Deepth,MCTSnum,sini,Timehorizon,sigma,tao,raw)
%function [Ts,TA,TV]=traindata(NetA,NetS,M,N,buffer,waittime,trainnum,c,Deepth,MCTSnum,sini,Timehorizon,sigma,tao,raw,QRv)
%lamda=0.9;
discount=0.9;

cprior1=0.2;
cprior2=0.35;

%actiontype=(3+2*Ajobtype)^M;%0:no;1:P;2~A+1:A;A+2:nP;A+3~2A+2:nA
%statenum=(2+Ajobtype)*M*(2*neighbornum+1);
%allstatenum=(2+Ajobtype)*N*M;
%actionnum=M;

%stateraw=zeros(M,1,Ajobtype);
%for k=1:Ajobtype
%    stateraw(:,1,k)=Ajobtime(k)*ones(M,1);
%end

actiontype=waittime*(2*(2*buffer+N)+1);

statenum=(2*buffer+N)*(N+1)*waittime;

Ts=[];
TA=[];
TV=[];

for traintime=1:trainnum

    rand('seed',sum(100*clock));

    %traindatas=zeros(statenum+allstatenum+1,N*Timehorizon);%观测集合，行长为状态state大小
    %traindataa=zeros(actiontype,N*Timehorizon);%响应集合，行长为action种类

    %traindataS=-1*ones(allstatenum+1,Timehorizon);
    %traindatav=zeros(1,Timehorizon);%响应集合，value

    s0=sini;
    s0.score=1;
    s0.product=0;
    s0.survival=0;

    traindatas=zeros(statenum,1);
    traindataa=zeros(actiontype,1);%响应集合，行长为action种类
    traindatav=zeros(1,1);

    time=1;

    while(time<=Timehorizon)
    %for time=1:Timehorizon  

        saset=zeros(1,statenum+1);
        QRset=cell(1);
        Ns=zeros(1);
        hashTable = containers.Map;

        for mcts=1:MCTSnum
            s=s0;
            track=[];
            WP=[];
            WT=[];
            t=0;
            %score=zeros(1,min(Deepth,Timehorizon-time+1));

            %for t=1:min(Deepth,Timehorizon-time+1)            
            while t<=min(Deepth,Timehorizon-time+1)

                statenow=reshape(s.work,1,statenum);
                actp=predict(NetA,dlarray(statenow','CB'));
                action=findaction(s,M,N,raw,buffer,waittime);

                Qn=zeros(1,length(action));
                Nn=zeros(1,length(action));
                fn=zeros(1,length(action));

                pn=zeros(1,length(action));
                for j=1:length(action)
                    pn(j)=action2num(action(j),M,N,buffer,waittime);
                end

                Pn=zeros(1,length(action));
                for j=1:length(action)
                    Pn(j)=actp(pn(j));
                end
                if sum(Pn)==0
                    Pn=ones(1,length(action))/length(action);
                else
                    Pn=Pn/sum(Pn);
                end
                Pn=(1-sigma)*Pn+sigma*dirichletRnd(ones(1,length(action)),0.03);

                for j=1:length(action)
                    sa=[statenow,action(j)];
                    stateStr=mat2str(sa);
                    if isKey(hashTable, stateStr)
                        isa=hashTable(stateStr);
                        %isa=find(ismember(saset,sa,'rows'));
                        Qn(j)=mean(QRset{isa(1)});
                        Nn(j)=Ns(isa(1));
                    end
                end

                for i=1:length(action)
                    [~,www]=process_wt_failure(s,action(i),M,N,raw,buffer,waittime);
                    act=baselineselect(s,M,N,raw,buffer,waittime);
                    if action(i)==act
                        fn(i)=Qn(i)+(c+cprior1*mean(www)/M+cprior2)*Pn(i)*sqrt(sum(Nn))/(Nn(i)+1);
                    else
                        fn(i)=Qn(i)+(c+cprior1*mean(www)/M)*Pn(i)*sqrt(sum(Nn))/(Nn(i)+1);
                    end
                end
                f=find(fn==max(fn));
                if isempty(f)
                    choose=randperm(length(action),1);
                else
                    choose=f(randperm(length(f),1));
                end
                sa=[statenow,action(choose)];
                stateStr=mat2str(sa);
                if isKey(hashTable, stateStr)
                    isa=hashTable(stateStr);                    
                    track(end+1)=isa(1);
                else
                    Ns=[Ns;0];
                    QRset{end+1}=zeros(1,1);
                    saset(end+1,:)=sa;
                    I=size(Ns,1);
                    track(end+1)=I;
                    hashTable(stateStr) = I;
                end
                [s,wp]=process(s,action(choose),M,N,raw,buffer,waittime);
                [sl,wt]=actnum2action(action(choose),waittime);
                t=t+wt;
                WP(end+1:end+wt)=wp;
                WT(end+1)=wt;
            end
            stoppoint=length(track);

            statenow=reshape(s.work,1,statenum);
            S=dlarray(statenow','CB');%%%%%%%%此处输入为行向量              
            v=double(extractdata(predict(NetS,S)))';%%%%%%输出结果

            for t=stoppoint:-1:1
                wp=WP(sum(WT(1:(t-1)))+WT(t):-1:sum(WT(1:(t-1)))+1);
                for tt=1:WT(t)
                    v=wp(tt)/(1.05*N/(1-discount))+discount*v;
                end

                if Ns(track(t))>0
                    QRset{track(t)}=[QRset{track(t)},v];
                else
                    QRset{track(t)}=v;
                end
                Ns(track(t))=Ns(track(t))+1;
            end
        end
        
        action=findaction(s0,M,N,raw,buffer,waittime);
        Nn=zeros(1,length(action));
        Qn=zeros(1,length(action));
        statenow=reshape(s0.work,1,statenum);

        for j=1:length(action)
            sa=[statenow,action(j)];
            stateStr=mat2str(sa);
            if isKey(hashTable, stateStr)
                isa=hashTable(stateStr);
                %isa=find(ismember(saset,sa,'rows'));
                Nn(j)=Ns(isa(1));
                Qn(j)=mean(QRset{isa(1)});
            end
        end

        pn=zeros(1,length(action));
        pa=zeros(actiontype,1);

        for j=1:length(action)
            pn(j)=action2num(action(j),M,N,buffer,waittime);
            tao_=1/tao;
            pa(pn(j))=Nn(j)^tao_/sum(Nn.^tao_);
        end

        if length(find(pa~=0))>1
            traindatas(:,end+1)=dlarray(statenow','CB');
            traindataa(:,end+1)=dlarray(pa,'CB');
            traindatav(:,end+1)=dlarray(max(Qn),'CB');
        end

        f=find(Qn==max(Qn));
        choose=f(randperm(length(f),1));

        s0=process(s0,action(choose),M,N,raw,buffer,waittime);
        [sl,wt]=actnum2action(action(choose),waittime);
        time=time+wt;

        fprintf('time turn:%d\n',time);
    end

    D=[];
    for j=1:size(traindataa,2)
        if sum(traindataa(:,j))==0
            D(end+1)=j;
        end
    end
    traindatas(:,D)=[];
    traindataa(:,D)=[];
    traindatav(:,D)=[];
    D=[];
    for j=1:size(traindataa,2)
        if any(isnan(traindataa(:,j)))||any(isnan(traindatas(:,j)))||any(isnan(traindatav(:,j)))
            D(end+1)=j;
        end
    end
    traindatas(:,D)=[];
    traindataa(:,D)=[];
    traindatav(:,D)=[];
    Ts=[Ts,traindatas];
    TA=[TA,traindataa];
    TV=[TV,traindatav];
end
end