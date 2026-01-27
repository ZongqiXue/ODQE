function [loss,gradients,state] = modelLossA(net,X,a)

% Forward data through network.
[f,state] = forward(net,X);

% 定义
loss = crossentropy(f,a);%must be f,a
%loss = crossentropy(f,a)+l2loss(v,e);

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end 