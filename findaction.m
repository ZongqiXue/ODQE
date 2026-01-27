function Action=findaction(s,M,N,raw,buffer,waittime)
%add waittime buffer
S=s.work;

sss1=(max(S)==0);
if isempty(sss1)
    topmove=0;
else
    for k=length(sss1):-1:1
        if sss1(k)~=1
            break
        end
    end
    topmove=2*buffer+N-k;
end
 
sss2=zeros(2*buffer+N,1);
for k=1:(2*buffer+N)
    sss2(k)=isequal(S(:,k),raw);
end

for k=1:(2*buffer+N)
    if sss2(k)~=1
        break
    end
end

botmove=-k+1;

action=[botmove:1:topmove];

if isempty(action)
    action=0;
end

Action=zeros(1,waittime*length(action));
for i=1:length(action)
    Action(waittime*(i-1)+1:waittime*i)=waittime*action(i)+(1:waittime);
end

end













