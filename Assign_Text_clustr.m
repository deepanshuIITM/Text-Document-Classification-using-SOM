clc
clear all
close all
%% UPLOAD FILE
filename = "a_SMSSpam";
data = readtable(filename,'TextType','string'); 
head(data)

%% Class Distribution
data.Category = categorical(data.ham);
figure(1)
histogram(data.Category)
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")

%% Train and Test
cvp = cvpartition(data.Category,'Holdout',0.5);
dataTrain = data(cvp.training,:);
dataTest = data(cvp.test,:);

%% Extraction data and label
textDataTrain = dataTrain.GoUntilJurongPoint_Crazy__AvailableOnlyInBugisNGreatWorldLaEBuf;
textDataTest = dataTest.GoUntilJurongPoint_Crazy__AvailableOnlyInBugisNGreatWorldLaEBuf;
YTrain = double(categorical(dataTrain.ham));
YTest = double(categorical(dataTest.ham));

%% Preprocess data
documents = preprocessText(textDataTrain);
documents(1:5)

%% Bag of word
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag,2);
[bag,idx] = removeEmptyDocuments(bag);
YTrain(idx) = [];
bag;

%% Training
XTrain = bag.Counts;
mdl = fitcecoc(XTrain,YTrain,'Learners','linear');
%mdl = mnrfit(XTrain,YTrain);

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
label_train = int2str(YTrain(1:1000,:));
sTrain  = som_data_struct(XTrain(1:1000,:),'name','Train','labels',label_train);
label_test = int2str(YTest(1:1000,:));
sTest  = som_data_struct(XTest(1:1000,:),'name','Test','labels',label_test);
%sTrain = som_normalize(sTrain,'var');
%sTest = som_normalize(sTest,'var');
%sM = som_make(sTrain,'msize',[13,9]);
sM_train = som_supervised(sTrain,'msize',[13,9]);
sM_train1 = som_autolabel(sTest,sM_train,'vote');
figure(4)
som_show(sM_train,'umat','all','empty','Labels')
som_show_add('label',sM_train.labels,'textsize',8,'textcolor','r','Subplot',2);

sM_make = som_make(sTrain,'msize',[13,9]);
figure(5)
som_show(sM_make,'umat','all','empty','Labels')
som_show_add('label',sM_make.labels,'textsize',8,'textcolor','r','subplot',2);

%% CALCULATE ACCURACY
accu = 0;
for i = 1:size(label_test,1)
    if sM_train1.labels{i,1}== sM_train1.labels{i,2}
        accu = accu + 1;
    end
end
accuracy = accu/size(label_test,1);          

%% Test Data
sM_pred = som_make(sM_train1.data,'msize',[15,10]);
figure(6)
som_show(sM_pred,'umat','all','empty','Cross_Validation')
som_show_add('label',sM_train1.labels(1:150,:),...
    'textsize',8,'textcolor','r');
% h1 = som_hits(sM_pred,sTest);
% som_show_add('hit',h1,'MarkerColor','w','Subplot',1)
% sM_test = som_make(YTest,'msize',sM_pred.topol.msize);
% figure(7)
% som_show(sM_test,'umat','all')
% h2 = som_hits(sM_pred,sM_test);
% som_show_add('hit',h2,'MarkerColor','w','Subplot',1)
