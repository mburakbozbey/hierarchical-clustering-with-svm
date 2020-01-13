
clear all;close all;
clc;

dataset = readmatrix('/path/to/dataset/');

% Variables and placeholders are defined.

pc = 0.75 ;     % percentage 75%
rng('default')  % For reproducibility
classNum = 8;
dim = 600;

D = [];
link = [];

% Split each class, %75 to D, %25 to T

for k=1:8
    idx = dataset(:,1) == k;
    sampleC = dataset(idx,:);
    train = sampleC(1:round(pc*size(sampleC,1)),:);  % Split D set for kth class
    test = sampleC(round(pc*size(sampleC,1)):end,:); % Split T set for kth class
    D = [D;train];
    link = [link;test];
end

% Split patterns and classes of D & T set from private dataset
dataClass = D(:,1);
dataPatterns = D(:,[2:end-1]);
dataRowCount = size(dataPatterns, 1);

testClass = link(:,1);
testPatterns = link(:,[2:end-1]);
testRowCount = size(testPatterns, 1);

dataFeatures = zeros(6*length(dataPatterns), 100);

for i=1:length(dataPatterns)
    for k=0:5
        dataFeatures((i-1)*6+1+k,:) = dataPatterns(i, 1+100*k:100*(k+1));
    end
end

% Unsupervised Agglomerative Hiearchical Clustering

dist = pdist(dataFeatures);
link = linkage(dist, 'complete');
c = cluster(link, 'maxclust', 300);

dataHistogram = zeros(length(dataPatterns), 300);

for i=1:length(dataPatterns)
    for k=1:6
        dataHistogram(i, c(i*5+k-1)) = ...
                            dataHistogram(i, c(i*5+k-1))+1;
    end
end

% Build histogram for each instance

Mdl = fitcecoc(dataHistogram, dataClass);
disp("Training Error:");
disp(resubLoss(Mdl));


% Testing process

testClusters = -1*ones(6*testRowCount,1);
testFeatures = zeros(6*testRowCount, 100);
testHistogram = zeros(testRowCount, 300);

for i=1:testRowCount
    for k=0:5
        testFeatures((i-1)*6+1+k,:) = testPatterns(i, 1+100*k:100*(k+1));
    end
end


% Find cluster centroid
C = zeros(numel(unique(c)), size(dataFeatures,2));
for cid = unique(c)'
   C(cid,:) = mean(dataFeatures(c == cid,:)); 
end

% Find closest centroid

for k=1:testRowCount*6
    [~, idx_test] = pdist2(C, testFeatures(k,:),'euclidean','Smallest',1);
    testClusters(k) = idx_test;
end

for i=1:testRowCount
    for k=1:6
        testHistogram(i, testClusters(i*5+k-1)) = ...
                                testHistogram(i, testClusters(i*5+k-1))+1;
    end
end

[predClass Score Cost] = predict(Mdl, testHistogram);
accuracy = 1 - length(find(testClass~=predClass))/testRowCount; 
disp("Accuracy:");
disp(accuracy);
confM = confusionmat(testClass,predClass);
confM = confM';
confM = confM + eps; % Addition by epsilon due to division by zero error, etc.
testPrecision = diag(confM)./sum(confM,2);
testRecall = diag(confM)./sum(confM,1)';

cutoff = median([link(end-7,3) link(end-6,3)]);
dendrogram(link,300,'ColorThreshold',cutoff)
