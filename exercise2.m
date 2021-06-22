addpath("C:\\Users\\user\\Desktop\\svm")
addpath("C:\\Users\\user\\Desktop\lsm\\LSSVMlab")

 %uiregress
 %demofun
 
%(gam) is the regularization parameter, determining the trade-off
%between the fitting error minimization and smoothness of the
%estimated function. sigma^2 (sig2) is the kernel function
%parameter of the RBF kernel:


%% Regression of the sinc function

X=(-3:0.01:3)';
Y=sinc ( X ) + 0.1.* randn ( length ( X ) , 1) ;

Xtrain=X(1:2:end)
Ytrain=Y(1:2:end)

Xtest=X(2:2:end)
Ytest=Y(2:2:end)

gamvalues=[10,10^3,10^6]
sig2values=[0.01,1,100]
% report mean squared error for every combination
mselist=[]
big_mselist=ones(3)

type='function estimation'

for i = 1:length(sig2values),
    for j = 1: length(gamvalues),
       figure;
       [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'});
       Yt = simlssvm({Xtrain,Ytrain,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'},{alpha,b},Xtest);
       plotlssvm({Xtrain,Ytrain,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'},{alpha,b});
       hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
       mse_temp=sum((Yt-Ytest).^2)
       fprintf('%i %i %i \n', gamvalues(j),sig2values(i),mse_temp)
       mselist=[mselist,mse_temp]
       %plotlssvm({Xtest,Ytest,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'},{alpha,b});
    end 
    big_mselist(i,:)=mselist
    mselist=[]
end


[gam_s ,sig2_s ,cost_s] = tunelssvm ({Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10,'mse'});

% the algorithm can either be --> simplex Nelder Mead method or
% gridsearch--> brute force gridsearch
[gam_g ,sig2_g ,cost_g] = tunelssvm ({Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10,'mse'});






%% Application of the Bayesian framework

sig2=0.4
gam=10
crit_L1=bay_lssvm({Xtrain, Ytrain,'f',gam,sig2},1)
crit_L2=bay_lssvm({Xtrain, Ytrain,'f',gam,sig2},2)
crit_L3=bay_lssvm({Xtrain, Ytrain,'f',gam,sig2},3)

[~,alpha,b]=bay_optimize({Xtrain,Ytrain, 'f',gam,sig2},1)
[~,gam]=bay_optimize({Xtrain,Ytrain, 'f',gam,sig2},2)
[~,sig2]=bay_optimize({Xtrain,Ytrain, 'f',gam,sig2},3)


sig2e=bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure')


%% Automatic Relevance Determination

X=6 .*rand(100,3)-3
Y=sinc(X(:,1))+0.1.*randn(100,1)

Xtrain=X(1:2:length(X),:)
Ytrain=Y(1:2:length(Y),:)
Xtest=X(2:2:length(X),:)
Ytest=Y(2:2:length(Y),:)

[selected,ranking]=bay_lssvmARD({Xtrain,Ytrain,'f',gam,sig2})

erro=[]
[alpha,b] = trainlssvm({Xtrain(:,selected),Ytrain,'f',gam,sig2,'RBF_kernel'});
%figure;  
%plotlssvm({Xtrain,Ytrain,type,gam,sig2values(i),'RBF_kernel','preprocess'},{alpha,b});
[Yht, Zt] = simlssvm({Xtrain(:,selected),Ytrain,type,gam,sig2values(i),'RBF_kernel'}, {alpha,b}, Xtest);
err=sum((Yt-Ytest).^2)
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
erro=[erro; err]





%% Robust regression 

X=(-6:0.2:6)'
Y=sinc(X)+0.1.*rand(size(X))

out=[15 17 19]
k=0.7+0.3*rand(size(out))
Y(out)=k

out =[41 44 46]
Y(out)=1.5+0.2*rand(size(out))

costfunction='crossvalidatelssvm'
[gam,sig2,cost]=tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex',costfunction,{10,'mse'})
[alpha,b]=trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'})
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'},{alpha,b})


% robust model

model=initlssvm(X,Y,'f',[],[],'RBF_kernel')
costFun='rcrossvalidatelssvm'
wFun='whuber' %F(X)=2.101574e-01 
%wFun='whampel' %1.370212e-01 
%wFun='wlogistic'% 1.463065e-01 
%wFun='wmyriad' %1.400601e-01 
model=tunelssvm(model,'simplex',costFun,{10,'mae';},wFun)
model=robustlssvm(model)
plotlssvm(model)


%% Homework problems
% logmap dataset

load logmap.mat

order=10
X=windowize(Z,1:(order+1))
Y=X(:,end);
X=X(:,1:order)

gam=10
sig2=10
[alpha,b]=trainlssvm({X,Y,'f',gam,sig2})

Xs=Z(end-order+1:end,1)
nb=50
prediction=predict({X,Y,'f',gam,sig2},Xs,nb)

figure
hold on
plot(Ztest,'k')
plot(prediction,'r')
hold off


clear
load logmap.mat

k_fold = 10;
nb = 50;

orders = [1:100];
maes = zeros(length(orders), 1);

gamma=logspace(-4,4,100)
sigma=logspace(-4,4,100)
maes = zeros(length(orders), length(orders);


for i = 1:length(orders)
    for j =1:length(sigma)
        for k=1:length(gamma)
            X = windowize(Z, 1:(i+1));
            Y = X(:, end);
            X = X(:, 1:i);

            Xs = Z(end-i+1:end, 1);

            
            [alpha,b] = trainlssvm({X,Y,'f',gamma(k),sigma(j),'RBF_kernel'});
            prediction=predict({X,Y,'f',gamma(k),sigma(j)},Xs,nb);
            maes(i, 1) = sum(abs(prediction-Ztest));

        end
    end 
end

plot(maes,'color','magenta')
title("Median Absolute Error to determine optimal order using tunnelssvm")
xlabel("Order")
ylabel("Median Absolute Error")
[M,I] = min(maes)


