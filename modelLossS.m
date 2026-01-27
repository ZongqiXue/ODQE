function [loss,gradients,state] = modelLossS(net,X,v)

QRv=20;
kapa=1;

% Forward data through network.
[e,state] = forward(net,X);

loss=0;
for i=1:size(v,1)
    for j=size(v,2)
        p=0;
        u=e(i,j)-v(i,j);
        if abs(u)<=kapa
            p=u^2/2;
        else
            p=kapa*(abs(u)-kapa/2);
        end
        if u<0
            p=p*abs((2*i-1)/(2*QRv)-1);
        else
            p=p*abs((2*i-1)/(2*QRv));
        end
        loss=loss+p; 
    end
end


% 定义
%loss = crossentropy(f,a)+l2loss(v,e);

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end