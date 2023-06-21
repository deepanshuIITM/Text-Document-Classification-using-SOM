clc
clear all
close all
%% UPLOAD FILE
%filename = "amazon_cells_labelled.csv";
filename = "a_Subject_data2.xlsx";
data = readtable(filename,'TextType','string'); 
data.Properties.VariableNames{1} = 'Var1';
data.Properties.VariableNames{2} = 'Var2';
head(data)

%% Class Distribution
data.Category = categorical(data.Var2);
figure(1)
histogram(data.Category)
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

%% Train and Test
cvp = cvpartition(data.Category,'Holdout',0.2);
dataTrain = data(cvp.training,:);
dataTest = data(cvp.test,:);

%% Extraction data and label 
textDataTrain = dataTrain.Var1;
textDataTest = dataTest.Var1;
YTrain = double(categorical(dataTrain.Var2));
YTest = double(categorical(dataTest.Var2));
% YTrain = dataTrain.Var2;
% YTest = dataTest.Var2;

%% Preprocess data
documents = preprocessText(textDataTrain);
documents(1:5)

%% Bag of word
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag,1);
[bag,idx] = removeEmptyDocuments(bag);
YTrain(idx) = [];
bag;

%% Training
XTrain = bag.Counts;
 mdl = fitcecoc(XTrain,YTrain,'Learners','linear');
% mdl = fitcecoc(XTrain,YTrain,'Learners','RBF');

%% Test %& Predict
documentsTest = preprocessText(textDataTest);
XTest = encode(bag,documentsTest);
YPred = predict(mdl,XTest);
figure(2)
confusionchart(YTest,YPred);
acc = sum(YPred == YTest)/numel(YTest);

%% %% Compare with raw data
rawDocuments = tokenizedDocument(textDataTrain);
rawBag = bagOfWords(rawDocuments);
cleanedBag = bag;
figure(3)
subplot(1,2,1)
wordcloud(rawBag);
title("Raw Data")
subplot(1,2,2)
wordcloud(cleanedBag);
title("Cleaned Data")

%% SOM algorithm
clc
label_train = int2str(YTrain);
sTrain  = som_data_struct(XTrain,'name','Train','labels',label_train);
label_test = int2str(YTest);
sTest  = som_data_struct(XTest,'name','Test','labels',label_test);
%sTrain = som_normalize(sTrain,'var');
%sTest = som_normalize(sTest,'var');
sM_train = som_supervised(sTrain,'algorithm','seq','mapsize','normal');
sM_train1 = som_autolabel(sTest,sM_train,'vote');
figure(4)
som_show(sM_train,'umat','all','empty','Labels')
%som_show(sM_train,'umat','all','empty','Labels','comp',[3,61,122,184])
som_show_add('label',sM_train.labels,'textsize',8,'textcolor','r','Subplot',2);

% figure(8)
% som_show(sM_train,'umat',[3,61,122,184])
%% CALCULATE ACCURACY
accu_count = 0;
for i = 1:size(label_test,1)
    if sM_train1.labels{i,1}== sM_train1.labels{i,2}
        accu_count = accu_count + 1;
    end
    som_pred(i,1) = str2num(sM_train1.labels{i,2});
end
accuracy = accu_count/size(label_test,1);  
figure(7)
confusionchart(YTest,som_pred);
%% Test Data
figure(5)
sM_pred = som_make(sM_train1,'munits',size(YTest,1)-5);
som_show(sM_pred,'umat','all','empty','Cross_Validation')
som_show_add('label',sM_train1.labels(1:min(size(YTest,1)-5,...
    sM_pred.topol.msize(1,1)*sM_pred.topol.msize(1,2)),:),...
    'textsize',8,'textcolor','r');
h1 = som_hits(sM_pred,sTest);
som_show_add('hit',h1,'MarkerColor','w','Subplot',1)
sM_test = som_make(sTest,'msize',sM_pred.topol.msize);
figure(6)
som_show(sM_test,'umat','all')
h2 = som_hits(sM_test,sTest);
som_show_add('hit',h2,'MarkerColor','w','Subplot',1)