function [sl,wt]=actnum2action(action,waittime)
n=mod(action,waittime);
if n==0
    wt=waittime;
else
    wt=n;
end
sl=(action-wt)/waittime;
end 