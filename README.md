# Unsupervised Hierarchical Clustering with SVM

Several algorithms were applied on **private dataset**, including  [Maximum Likelihood Estimation](https://github.com/mburakbozbey/maximum-likelihood-estimation) and [Parzen Window Estimation](https://github.com/mburakbozbey/parzen-window-estimation).For overall comparison on same dataset and to implement unsupervised hierarchical clustering with SVM, feature extraction approach was used. Agglomerative Hierarchical Clustering algorithm was used for clustering six independent feature vectors from **private dataset** for all samples in same space:
1. Converting data from one sample with 6 feature vectors to one sample for one feature vector format for same sample.
2. Distance calculation of every pair in training set.
3. By using linkages between each pair, samples formed into hierarchical clusters with respect to smallest Euclidean distance between.
4. To create a partition of the training set, 300 clusters were formed which was used for codewords for classification.
After that, bag of codewords were generated for each sample which was mapped to its histogram that was formed with respect to frequencies of the codewords. After that, each sample’s histogram were classified with error coding-output coding with SVM method as mentioned in part d. Dendrogram of hierarchical clustering which was cut to create 300 clusters that found empirically is as follows:

<p align="center">
  <img src="https://i.ibb.co/cwLxtfy/Screenshot-1.png">
</p>

Training error was 16% and for test set, accuracy -exact match ratio- is 46%. For each test sample, its 6 feature vectors are mapped to closest cluster centroid with respect to Euclidean distances. After that, resulting histogram of the test sample was used for classification. Precision and recall for the overall classifier:

<p align="center">
  <img src="https://i.ibb.co/nM9gCQD/Screenshot-2.png">
</p>

## Overall comparison
On same dataset, best results were achieved by SVM and after that, MLE with PCA method had close results to SVM in my implementation:

<p align="center">
  <img src="https://i.ibb.co/48prVMR/Screenshot-3.png">
</p>
