# AGGC2022
Evaluation Code


The evaluation is done by comapring the index images (index 1= Stroma, index 2 =Normal, index 3 = G3, index 4=G4,index 5 = G5) of ground truth and predictions. 

The final F1 score of a subset will be calculated based on the confusion matrix summarzing all test images in that subset.(The evaluation code maybe not suitable for just testing one sample. It is designed for the whole dataset which contains all the indexes 1-5)

Please resize the images to 2x (i.e. downsizing the original images we provided by 10 times) for evaluation.

.\Subset1_Train_PredictionExample_2x_indeximage and .\Subset1_Train_GroundTruth_2x_indeximage are some examples of predicition results and the correpsonding ground truth of training images, respectively.

Since there are overlappings in the raw annotation (i.e. some pixels are assigned with more than one labels), in most cases we choose the "most malignant" label for those pixels when we generated the index image of ground truth. However, in some special cases, for example, pathologists elaborately drew a small G3 inside a larger G4 region, although the most malignant class is G4 for those overlapped pixels, we still stick to the lower class G3 instead.


