function [TV]=traindata(NetA,NetS,M,N,buffer,waittime,trainnum,c,Deepth,MCTSnum,sini,Timehorizon,sigma,tao,raw)
%function [Ts,TA,TV]=traindata(NetA,NetS,M,N,buffer,waittime,trainnum,c,Deepth,MCTSnum,sini,Timehorizon,sigma,tao,raw,QRv)
%lamda=0.9;
discount=0.9;

%%%%%%%%%%%%%%%%%
cvisit=50;
cscale=1;
samplenum=16;

cprior1=0.1;
cprior2=0.2;

%%%%%%%%%%%%%%%%%

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

        statenow=reshape(s0.work,1,statenum);
        actp=predict(NetA,dlarray(statenow','CB'));
        action=findaction(s0,M,N,raw,buffer,waittime);
        action_on_root=action;

        pn=zeros(1,length(action));
        for j=1:length(action)
            pn(j)=action2num(action(j),M,N,buffer,waittime);
        end
        
        Proot=zeros(1,length(action));
        for j=1:length(action)
            Proot(j)=actp(pn(j));
        end

        for i=1:length(action)
            [~,www]=process_wt_failure(s0,action(i),M,N,raw,buffer,waittime);
            act=baselineselect(s0,M,N,raw,buffer,waittime);
            if action(i)==act
                Proot(j)=Proot(j)+cprior1*mean(www)/M+cprior2;
            else
                Proot(j)=Proot(j)+cprior1*mean(www)/M;
            end
        end

        if sum(Proot)==0
            Proot=ones(1,length(action))/length(action);
        else
            Proot=Proot/sum(Proot);
        end

        Proot=Proot+gumbel(length(Proot),0,1);

        [sorted_values, sorted_indices] = sort(Proot, 'descend');

        M0=floor(min(length(action_on_root),samplenum));
        Mroot=M0;

        action_on_root = action_on_root(sorted_indices(1:Mroot));
        Proot=Proot(sorted_indices(1:Mroot));

        while(Mroot>1)
            for mchoose=1:length(action_on_root)
                for helving=1:max(1,floor(log(2)*MCTSnum/(Mroot*log(M0))))
                    %track=zeros(1,1+min(Deepth,Timehorizon-time+1));
                    track=0;
                    statenow=reshape(s0.work,1,statenum);
                    sa=[statenow,action_on_root(mchoose)];
                    stateStr=mat2str(sa);
                    if isKey(hashTable, stateStr)
                        isa=hashTable(stateStr);
                        %isa=find(ismember(saset,sa,'rows'));                  
                        track(1)=isa(1);
                    else
                        Ns=[Ns;0];
                        QRset{end+1}=zeros(1,1);
                        saset(end+1,:)=sa;
                        I=size(Ns,1);
                        track(1)=I;
                        hashTable(stateStr) = I;
                    end

                    t=0;
                    
                    if time<=(90)
                        [s,wp]=process_raw(s0,action_on_root(mchoose),M,N,raw,buffer,waittime);
                    else
                        [s,wp]=process(s0,action_on_root(mchoose),M,N,raw,buffer,waittime);
                    end

                    [sl,wt]=actnum2action(action_on_root(mchoose),waittime);
                    t=t+wt;
                    WP(1:wt)=wp;
                    WT(1)=wt;

                    while t<=min(Deepth,Timehorizon-time+1)
                    %for t=1:min(Deepth,Timehorizon-time+1)
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

                        for i=1:length(action)
                            [~,www]=process_wt_failure(s,action(i),M,N,raw,buffer,waittime);
                            act=baselineselect(s0,M,N,raw,buffer,waittime);
                            if action(i)==act
                                Pn(j)=Pn(j)+cprior1*mean(www)/M+cprior2;
                            else
                                Pn(j)=Pn(j)+cprior1*mean(www)/M;
                            end
                        end

                        if sum(Pn)<0.01
                            Pn=ones(1,length(action))/length(action);
                        else
                            Pn=Pn/sum(Pn);
                        end

                        Qvisit=0;
                        Pvisit=0;

                        for j=1:length(action)
                            sa=[statenow,action(j)];
                            stateStr=mat2str(sa);
                            if isKey(hashTable, stateStr)
                                isa=hashTable(stateStr);
                                %isa=find(ismember(saset,sa,'rows'));                                
                                if Ns(isa(1))>0
                                    Qn(j)=mean(QRset{isa(1)});
                                    Nn(j)=Ns(isa(1));
                                    Qvisit=Qvisit+Qn(j)*Pn(j);
                                    Pvisit=Pvisit+Pn(j);
                                end
                            end
                        end
                        if Pvisit>0.01
                            Qvisit=Qvisit/Pvisit;
                        end
                        for j=1:length(action)
                            sa=[statenow,action(j)];
                            stateStr=mat2str(sa);
                            if ~isKey(hashTable, stateStr)
                                Qn(j)=Qvisit;
                            else
                                isa=hashTable(stateStr);
                                if Ns(isa(1))<=0
                                    Qn(j)=Qvisit;
                                end
                            end
                        end
                        Qn=(cvisit+max(Nn))*cscale*Qn+Pn;
                        if isempty(Qn)
                            Qn=Pn;
                        end
                        Qn=exp(Qn)/sum(exp(Qn));
                        if isempty(Nn)
                            fn=Qn;
                        else
                            fn=Qn-Nn/(1+sum(Nn));
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
                            %isa=find(ismember(saset,sa,'rows'));                            
                            track(end+1)=isa(1);
                        else
                            Ns=[Ns;0];
                            QRset{end+1}=zeros(1,1);
                            saset(end+1,:)=sa;
                            I=size(Ns,1);
                            track(end+1)=I;
                            hashTable(stateStr) = I;
                        end
                        if (time+t)<=90
                            [s,wp]=process_raw(s,action(choose),M,N,raw,buffer,waittime);
                        else
                            [s,wp]=process(s,action(choose),M,N,raw,buffer,waittime);
                        end
                        [sl,wt]=actnum2action(action(choose),waittime);
                        t=t+wt;
                        WP(end+1:end+wt)=wp;
                        WT(end+1)=wt;
                    end
                    stoppoint=length(track);
                    
                    statenow=reshape(s.work,1,statenum);
                    S=dlarray(statenow','CB');%%%%%%%%此处输入为行向量                    
                    v=double(extractdata(predict(NetS,S)));%%%%%%输出结果

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
            end
            fn=zeros(length(action_on_root),1);
            Nn=zeros(length(action_on_root),1);
            Qn=zeros(length(action_on_root),1);
            Mroot=max(1,floor(length(action_on_root)/2));
            
            for mchoose=1:length(action_on_root)
                statenow=reshape(s0.work,1,statenum);
                sa=[statenow,action_on_root(mchoose)];
                stateStr=mat2str(sa);
                if isKey(hashTable, stateStr)
                    isa=hashTable(stateStr);
                    Qn(mchoose)=mean(QRset{isa(1)});
                    Nn(mchoose)=Ns(isa(1));
                end
            end
            fn=Proot'+(cvisit+max(Nn))*cscale*Qn;
            [sorted_values, sorted_indices] = sort(fn, 'descend');
            
            action_on_root = action_on_root(sorted_indices(1:Mroot));
            Proot=Proot(sorted_indices(1:Mroot));

        end

        actionchoose=action_on_root(1);
        statenow=reshape(s0.work,1,statenum);
        sa=[statenow,actionchoose];
        stateStr=mat2str(sa);

        DTB=0;

        if isKey(hashTable, stateStr)
            isa=hashTable(stateStr);
            %isa=find(ismember(saset,sa,'rows'));
            DTB=mean(QRset{isa});            
        end

        action=findaction(s0,M,N,raw,buffer,waittime);
        Nn=zeros(1,length(action));
        Qn=zeros(1,length(action));

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

        Qvisit=0;
        Pvisit=0;      
        for j=1:length(action)
            sa=[statenow,action(j)];
            stateStr=mat2str(sa);
            if isKey(hashTable, stateStr)
                isa=hashTable(stateStr);
                %isa=find(ismember(saset,sa,'rows'));
                if Ns(isa(1))>0
                    Nn(j)=Ns(isa(1));
                    Qn(j)=mean(QRset{isa(1)});
                    Qvisit=Qvisit+Qn(j)*Pn(j);
                    Pvisit=Pvisit+Pn(j);
                end
            end
        end

        if Pvisit~=0
            Qvisit=Qvisit/Pvisit;
        end

        for j=1:length(action)
            sa=[statenow,action(j)];
            stateStr=mat2str(sa);
            if ~isKey(hashTable, stateStr)
                Qn(j)=Qvisit;
            else
                if Ns(isa(1))<=0
                    Qn(j)=Qvisit;
                end
            end
        end
        Qn=(cvisit+max(Nn))*cscale*Qn+Pn;
        Qn=exp(Qn)/sum(exp(Qn));

        pn=zeros(1,length(action));
        pa=zeros(actiontype,1);
        
        for j=1:length(action)
            pn(j)=action2num(action(j),M,N,buffer,waittime);
            pa(pn(j))=Qn(j);
        end


        if time<=(90)
            s0=process_raw(s0,actionchoose,M,N,raw,buffer,waittime);
        else
            s0=process(s0,actionchoose,M,N,raw,buffer,waittime);
        end

        [sl,wt]=actnum2action(actionchoose,waittime);
        time=time+wt;
        if length(find(pa~=0))>1
            vvv=DTB*ones(1,wt);
            traindatav(:,end+1:end+wt)=dlarray(vvv,'CB');
        end
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
    TV=[TV,traindatav];
    
end
end
