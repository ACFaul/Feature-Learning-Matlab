clear;
x = -5:0.5:5;
% True function.
t = 1 + x + sin(x);
% Contstruct two layer network with 3 hidden neurons.
net = fitnet(3); 
% Remove normalization and de-normalization.
net.input.processFcns = { };
net.output.processFcns= { };
% Train the neural network.
[net,~,~,e] = train(net,x,t);
% Weights in the first layer.
IW = net.IW{1,1}; 
% Bias weights in the first layer.
b1 = net.b{1};
% Bias weight in the second layer.
b2 = net.b{2};
% Weights in the second layer.
LW = net.LW{2,1};

X = -5:0.1:5;
T = 1 + X + sin(X);
% Predictions.
Y = b2 + LW * tansig( b1 * ones(1,length(X)) + IW * X );
% Constructed basis functions.
Z = tansig( b1 * ones(1,length(X)) + IW * X );
figure;
% Plot true function.
plot(X,T,'r-')
hold on
% Plot predictions.
plot(X,Y,'b-')
% Plot bias.
plot(X,b2*ones(1,length(X)),'k-')
% Plot basis functions with coefficients.
plot(X,LW(1)*Z(1,:),'k:')
plot(X,LW(2)*Z(2,:),'k-.')
plot(X,LW(3)*Z(3,:),'k--')
legend('true function', 'prediction','bias','Location','northwest')
% Plot error at training data.
figure
plot(x,e,'ko', 'MarkerFaceColor','k','MarkerSize',3)
hold on
plot(x,zeros(1,length(x)),'k-')