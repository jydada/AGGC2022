# AGGC2022
Evaluation Code


The evaluation is done by comapring the index images (index 1= Stroma, index 2 =Normal, index 3 = G3, index 4=G4,index 5 = G5) of ground truth and predictions. 

The final F1 score of a subset will be calculated based on the confusion matrix summarzing all test images in that subset.

Please resize the images to 2x (i.e. downsizing the original images we provided by 10 times) for evaluation.

.\prediction_tif and .\GT_2x_indeximage are some examples of predicition results and the correpsonding ground truth of training images, respectively.

