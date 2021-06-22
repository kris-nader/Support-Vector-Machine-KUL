%% Exercises 1.1 A simple Example
addpath("C:\\Users\\user\\Desktop\\svm")
addpath("C:\\Users\\user\\Desktop\lsm\\LSSVMlab")
% 2 classes of gaussians with the same covariance
X1=randn(50,2) +1;
X2=randn(51,2) -1;
% class labels
Y1=ones(50,1);
Y2=-ones(51,1);
% find the mean of each of the clusters
mean_x1=mean(X1);
mean_x2=mean(X2);


% plot the points
figure;
hold on;
plot(X1(:,1),X1(:,2),'ro');
plot(X2(:,1),X2(:,2),'bo');
x=-4:1:4
% out of curiosity check the y=-x classifier
y=-x
plot(x,y,'y')
title('Two Gaussians Classification-Same covariance matrices')


% Using the center- make a classifier that is exactly the middle of the points 
% find the slope of the line connecting the 2 means
slope = (mean_x1(2)-mean_x2(2)) / (mean_x1(1)-mean_x2(1))
% The perpandicular line that intersects that line
slope = 1/slope*(-1)
% find the midpoint of the line connecting the 2 means
midx=(mean_x2(1)+mean_x1(1))/2;
midy=(mean_x2(2)+mean_x1(2))/2;


% choose a point to draw the perpandicular bisector from 
x=2;
sep1= slope *(x-midx) +midy;
x2=-2;
sep2= slope *(x2-midx)+midy;


plot(mean_x1(1),mean_x1(2),'go'); %center green
plot(mean_x2(1),mean_x2(2),'go');

plot([mean_x1(1) mean_x2(1)], [mean_x1(2) mean_x2(2)],'g');
plot(midx, midy,'rx');
line([x2,x], [sep2, sep1],'color','m')

legend('Class 1','Class2','y=-x','midpoint x1','midpoint x2','line connecting midpts','biscect midpoint line','mid of means')
xlim([-6 6])
ylim([-6 6])
hold off;




%% Support Vector machine classifier- https://cs.stanford.edu/people/karpathy/svmjs/demo/. 

%% Least-squares support vector machine classifier
addpath('C:\\Users\\user\\Desktop\\svm_tut\\ex1')
load iris.mat

% using a polynomial kernel

degree_i=[1,2,3,4,5,6,7,8,9,10]
misclass=[]
for i=degree_i
    type='classification'; 
    gam = 1; % gamma
    t = 1; % intercept
    degree = i;
    disp('Polynomial kernel of degree 1-- a polynomial kernel'),

    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});
        
    figure;
    plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
    title (sprintf('Polynomial kernel with %d degree',degree))
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); 
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
    mis
end 

%use the RBF kernel- choose a good value for sigma
type='classification';
gam=1;
sig2values=logspace(-3,3,100) % reduces reach of the data

% sigma low bias high 
% sigma large then overfit
% low sigma then large vairnce and then low bias and we can overfit

erro=[]
% alpha  αi captures the weight of the ith training example as a
% support vector. Higher value of αi means that ith training 
% example holds more importance as a support vector

% b The bias is the distance to the origin of the hyperplane solution y
% intercept in a linear case
for i = 1:length(sig2values),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2values(i),'RBF_kernel'});
    %figure;
    
    %plotlssvm({Xtrain,Ytrain,type,gam,sig2values(i),'RBF_kernel','preprocess'},{alpha,b});
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2values(i),'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest);
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
    erro=[erro; err]
end

figure;
semilogx(sig2values, erro, 'm-');
title ('# of Misclassification using various values of sigma2')
xlabel('Sig2 Values')
ylabel('Number of Misclassification')


% Good choice for gam?
type='classification';
gam=1;
sig2=1 % reduces reach of the data
gamvalues=logspace(-3,3,100)
erro=[]

for i = 1:length(gamvalues),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamvalues(i),sig2,'RBF_kernel'});
    %figure;
    
    %plotlssvm({Xtrain,Ytrain,type,gamvalues(i),sig2,'RBF_kernel','preprocess'},{alpha,b});
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gamvalues(i),sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest);
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
    erro=[erro;err]
end

figure;
semilogx(gamvalues, erro, 'm-');
title ('Error using various values of gamma and sigma2=0.04')
xlabel('Gamma Values')
ylabel('Error')





%% 1.3.2 Tuning parameters using validation

% random split %% returns the validation error ?
%[perf_r,perfs_r] = rsplitvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , 0.80 , 'misclass') ;
%[perf_cv,perfs_cv] = crossvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , 10 , 'misclass') ;
%[perf_lcv,perfs_lcv] = leaveoneout ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , 'misclass');

sig2values=logspace(-3,3,50); 
gamvalues=logspace(-3,3,50);

full_error_sig_gam_r=zeros(1,50); % sig2 as the columns and gam as the rows
full_error_sig_gam_cv=zeros(1,50);
full_error_sig_gam_lcv=zeros(1,50);



for i = 1:length(sig2values),
    for j = 1: length(gamvalues),
        full_error_sig_gam_r(i,j)= rsplitvalidate({ Xtrain , Ytrain , 'c', gamvalues(j) , sig2values(i) ,'RBF_kernel'} , 0.80 , 'misclass') ;
        full_error_sig_gam_cv(i,j) = crossvalidate({ Xtrain , Ytrain , 'c', gamvalues(j) , sig2values(i) ,'RBF_kernel'} , 10 , 'misclass') ;
        full_error_sig_gam_lcv(i,j) = leaveoneout({ Xtrain , Ytrain , 'c', gamvalues(j) , sig2values(i) ,'RBF_kernel'} , 'misclass');
    end 
end

figure('Color',[1 1 1]);
surf(gamvalues, sig2values, full_error_sig_gam_r,'FaceAlpha',0.5);
set(gca,'Xdir','reverse','Ydir','reverse', 'XScale', 'log', 'YScale', 'log')
title('Validation set model performance plotted on Sigma and Gamma');
xlabel('Gamma (log)' ); ylabel('Sigma^2 (log)'); zlabel('% Misclassification');

plot3(gamvalues, sig2values, full_error_sig_gam_r)
xlabel('Gamma (log)' ); ylabel('Sigma^2 (log)'); zlabel('% Misclassification');

%% Automatic Paramter tuning %% run this maybe 2-3 times

[gam_s ,sig2_s ,cost_s] = tunelssvm ({Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10,'misclass'});



% the algorithm can either be --> simplex Nelder Mead method or
% gridsearch--> brute force gridsearch
[gam_g ,sig2_g ,cost_g] = tunelssvm ({Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

%% Using ROC curves
[alpha , b] = trainlssvm ({Xtrain , Ytrain , 'c', gam_s , sig2_s ,'RBF_kernel'});
[Yest , Ylatent] = simlssvm ({Xtrain , Ytrain , 'c', gam_s , sig2_s ,'RBF_kernel'}, {alpha , b}, Xtest);

roc(Ylatent,Ytest)

[alpha , b] = trainlssvm ({Xtrain , Ytrain , 'c', gam_s , sig2_s ,'RBF_kernel'});
[Yest , Ylatent] = simlssvm ({Xtrain , Ytrain , 'c', gam_s , sig2_s ,'RBF_kernel'}, {alpha , b}, Xtrain);

roc(Ylatent,Ytrain)

[alpha , b] = trainlssvm ({Xtrain , Ytrain , 'c', gam_g , sig2_g ,'RBF_kernel'});
[Yest , Ylatent] = simlssvm ({Xtrain , Ytrain , 'c', gam_g , sig2_g ,'RBF_kernel'}, {alpha , b}, Xtest);

roc(Ylatent,Ytest)

[alpha , b] = trainlssvm ({Xtrain , Ytrain , 'c', gam_g , sig2_g ,'RBF_kernel'});
[Yest , Ylatent] = simlssvm ({Xtrain , Ytrain , 'c', gam_g , sig2_g ,'RBF_kernel'}, {alpha , b}, Xtrain);

roc(Ylatent,Ytrain)

%% Bayesian framework
bay_modoutClass ({Xtrain , Ytrain , 'c', .1, 0.01}, 'figure'); 
bay_modoutClass ({Xtrain , Ytrain , 'c', .1, 0.1}, 'figure');
bay_modoutClass ({Xtrain , Ytrain , 'c', .1 , 1}, 'figure'); 

bay_modoutClass ({Xtrain , Ytrain , 'c', 1, 0.01}, 'figure'); 
bay_modoutClass ({Xtrain , Ytrain , 'c', 1, 0.1}, 'figure');
bay_modoutClass ({Xtrain , Ytrain , 'c', 1 , 1}, 'figure'); 

bay_modoutClass ({Xtrain , Ytrain , 'c', 10, 0.01}, 'figure'); 
bay_modoutClass ({Xtrain , Ytrain , 'c', 10, 0.1}, 'figure');
bay_modoutClass ({Xtrain , Ytrain , 'c', 10 , 1}, 'figure'); 

bay_modoutClass ({Xtrain , Ytrain , 'c', 100, 0.01}, 'figure'); 
bay_modoutClass ({Xtrain , Ytrain , 'c', 100, 0.1}, 'figure');
bay_modoutClass ({Xtrain , Ytrain , 'c', 100 , 1}, 'figure'); 


%% Homework problems

load ripley.mat

figure;
hold on;

x_train1=vertcat(Xtrain(:,1),Xtest(:,1))
x_train2=vertcat(Xtrain(:,2),Xtest(:,2))
y_data=vertcat(Ytrain,Ytest)

gscatter(x_train1,x_train2,y_data,'rgb','osd');
title('Ripley Dataset')
xlabel('X1');
ylabel('X2');

%kernel='lin_kernel' %0.1400
kernel='poly_kernel' %0.1320 gamma,t,degree 1.6699      6.4354           5
%kernel='RBF_kernel' %0.1200
lfold=10
model={Xtrain,Ytrain,'c',[],[],kernel,'csa'}
[gam,sig2,cost]=tunelssvm(model,'simplex','crossvalidatelssvm',{lfold,'misclass'})
[alpha,b]=trainlssvm({Xtrain,Ytrain,'c',gam,sig2,kernel})
plotlssvm({Xtrain,Ytrain,'c',gam,sig2,kernel,'preprocess'},{alpha,b})

[Ysim,Ylatent]=simlssvm({Xtrain,Ytrain,'c',gam,sig2,kernel},{alpha,b},Xtest)
roc(Ylatent,Ytest)



%% Homework data-breat


load breast.mat

data=vertcat(trainset,testset)
label=vertcat(labels_train,labels_test)

[l,m]=size(data)
Xmean=mean(data)
Xstandard_dev=std(data)

B=(data-repmat(Xmean,[l,1]))./repmat(Xstandard_dev,[l,1])

[W,pcomp]=pca(B)

class1=find(label==-1 )
class2=find(label==1 )

pca1=pcomp(class1,:)
pca2=pcomp(class2,:)

figure;
plot(pca1(:,1),pca1(:,2),'b.',pca2(:,1),pca2(:,2),'r.')
title('Breast Cancer PCA Dataset')
xlabel('PC1');
ylabel('PC2');

kernel='lin_kernel' %0.0375
kernel='poly_kernel' % 0.1925     [gamma t degree]: 0.26045      4.3993           4 
kernel='RBF_kernel' %0.0525
lfold=10
model={trainset,labels_train,'c',[],[],kernel,'csa'}
[gam,sig2,cost]=tunelssvm(model,'simplex','crossvalidatelssvm',{lfold,'misclass'})
[alpha,b]=trainlssvm({trainset,labels_train,'c',gam,sig2,kernel})
%plotlssvm({trainset,labels_train,'c',gam,sig2,kernel,'preprocess'},{alpha,b})

[Ysim,Ylatent]=simlssvm({trainset,labels_train,'c',gam,sig2,kernel},{alpha,b},testset)
roc(Ylatent,labels_test)
hold on;





%% Homework -diabetes

load diabetes.mat

%%%% standardize and center the data for a pca
data=vertcat(trainset,testset)
label=vertcat(labels_train,labels_test)

[l,m]=size(data)
Xmean=mean(data)
Xstandard_dev=std(data)

B=(data-repmat(Xmean,[l,1]))./repmat(Xstandard_dev,[l,1])

[W,pcomp]=pca(B)

class1=find(label==-1 )
class2=find(label==1 )

pca1=pcomp(class1,:)
pca2=pcomp(class2,:)

figure;
plot(pca1(:,1),pca1(:,2),'b.',pca2(:,1),pca2(:,2),'r.')
title('Diabetes Cancer PCA Dataset')
xlabel('PC1');
ylabel('PC2');


kernel='lin_kernel' %0.2533
kernel='poly_kernel' %0.6331    [gamma t degree]: 0.051727     0.77857           3
kernel='RBF_kernel' %0.2533
lfold=10
model={trainset,labels_train,'c',[],[],kernel,'csa'}
[gam,sig2,cost]=tunelssvm(model,'simplex','crossvalidatelssvm',{lfold,'misclass'})
[alpha,b]=trainlssvm({trainset,labels_train,'c',gam,sig2,kernel})
plotlssvm({trainset,labels_train,'c',gam,sig2,kernel,'preprocess'},{alpha,b})

[Ysim,Ylatent]=simlssvm({trainset,labels_train,'c',gam,sig2,kernel},{alpha,b},testset)
roc(Ylatent,labels_test)

