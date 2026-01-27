function [s,wp]=process(s,action,M,N,raw,buffer,waittime,failureP)

%add buffer waittime

penalty=0.2;
failureprobability=[failureP,failureP,failureP/2];

S=s.work;
S0=s.work;
product=s.product;

[sl,wt]=actnum2action(action,waittime);

action=sl;

if action>0
    S=[repmat(raw, 1, action),S0(:,1:(2*buffer+N-action))];
elseif action<0
    S=[S0(:,(abs(action)+1):(2*buffer+N)),zeros(waittime*(N+1),abs(action))];
else
    S=S0;
end 

wp=zeros(1,wt);
remainingwork=zeros(N,2);

for t=1:wt
    remainingtime=wt-t+1;
    for i=1:N
        [wkst,rw]=missionselect(S(:,i+buffer),i,M,N,waittime,buffer,remainingtime,remainingwork(i,:));%P-1,2,3;A-4,5,6;none-0
        remainingwork(i,:)=rw;
        Rnum=rand(1,1);
        if wkst>=1&&wkst<=waittime
            remainingwork(i,2)=remainingwork(i,2)-1;
            if remainingwork(i,2)==0
                if (~(Rnum<failureprobability(1)&&wkst==1))&&(~(Rnum<failureprobability(2)&&wkst==2))&&(~(Rnum<failureprobability(3)&&wkst==3))
                    S(waittime*(i-1)+wkst,i+buffer)=S(waittime*(i-1)+wkst,i+buffer)-1;
                end
            end
            wp(t)=wp(t)+1;
            if S(:,i+buffer)==0
                product=product+1;
            end
        elseif wkst>waittime&&wkst<=2*waittime
            remainingwork(i,2)=remainingwork(i,2)-1;
            if remainingwork(i,2)==0
                if (~(Rnum<failureprobability(1)&&wkst==waittime+1))&&(~(Rnum<failureprobability(2)&&wkst==waittime+2))&&(~(Rnum<failureprobability(3)&&wkst==waittime+3))
                    S(waittime*N+wkst-waittime,i+buffer)=S(waittime*N+wkst-waittime,i+buffer)-1;
                end
            end
            wp(t)=wp(t)+1;
            if S(:,i+buffer)==0
                product=product+1;
            end
        end
    end
end

wp(1)=max(0,wp(1)-penalty*abs(action));

sss=find(~(max(S)==0));
s.work=zeros(waittime*(N+1),2*buffer+N);
s.work(:,1:length(sss))=S(:,sss);
%s.work=S;
%s.product=product;

end
