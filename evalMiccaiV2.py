#!/usr/bin/python

'''
Haoda Lu, jydada2018@gmail.com
2022/03/31

Evaluation metrics for AGGC2022 inspired by paper Fully Convolutional Networks for Semantic Segmentation.
'''
import matplotlib.pyplot as plt
import os.path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
import itertools

def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''

class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def plot_confusion_matrix(cm, Name,
                          target_names,
                          title='Confusion matrix',
                          cmap='Blues', 
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 7))
    #    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=15)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), size=15)
    plt.savefig('./' + Name + '_Confusion_Matrix.png', format='png', bbox_inches='tight')
    # plt.show()

def main():

    annotation_root = "./Subset1_Train_GroundTruth_2x_indeximage/"
    seg_root = "./Subset1_Train_PredictionExample_2x_indeximage/"

    # annotations = os.listdir(annotation_root)

    annotations = sorted(glob.glob(annotation_root + '\\' + '*.tif'))
    conf_mat = np.zeros((6, 6))
    for i, filename in enumerate(annotations):
        print(" %s  %d / %d" % (filename, i + 1, len(annotations)))
        dirname, Name = os.path.split(filename)
        gt_path = filename
        seg_path = seg_root + Name
        gt = Image.open(gt_path)
        gt = np.array(gt)
        seg = Image.open(seg_path)
        seg = np.array(seg)
        check_size(seg, gt)

        cont1 = 0
        cont2 = 0
        a = list([0]) * (gt.shape[0]*gt.shape[1])
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                a[cont1] = gt[i, j]
                cont1 = cont1 + 1

        b = list([0]) * (seg.shape[0]*seg.shape[1])
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                b[cont2] = seg[i, j]
                cont2 = cont2 + 1

        plot_confusion_matrix(confusion_matrix(y_true=a, y_pred=b), Name, normalize=False, target_names=["Background", "Stroma", "Normal", "G3", "G4", "G5"],
                              title='Confusion Matrix')
        conf_mat = conf_mat + confusion_matrix(y_true=a, y_pred=b)

    # Weighted-average F1-score = 0.25 * F1-score_G3 + 0.25 * F1-score_G4 +0.25 * F1-score_G5 +0.125 * F1-score_Normal +0.125 * F1-score_Stroma, where:
    #
    # F1-score=2×Precision×Recall/(Precision+ Recall);Precision=TP/(TP+FP);Recall=TP/(TP+FN)


    Stroma_Recall = conf_mat[1, 1] / np.sum(conf_mat[1, :])
    Normal_Recall = conf_mat[2, 2] / np.sum(conf_mat[2, :])
    G3_Recall = conf_mat[3, 3] / np.sum(conf_mat[3, :])
    G4_Recall = conf_mat[4, 4] / np.sum(conf_mat[4, :])
    G5_Recall = conf_mat[5, 5] / np.sum(conf_mat[5, :])


    Stroma_Pre = conf_mat[1, 1] / (np.sum(conf_mat[:, 1]) - conf_mat[0, 1])
    Normal_Pre = conf_mat[2, 2] / (np.sum(conf_mat[:, 2]) - conf_mat[0, 2])
    G3_Pre = conf_mat[3, 3] / (np.sum(conf_mat[:, 3]) - conf_mat[0, 3])
    G4_Pre = conf_mat[4, 4] / (np.sum(conf_mat[:, 4]) - conf_mat[0, 4])
    G5_Pre = conf_mat[5, 5] / (np.sum(conf_mat[:, 5]) - conf_mat[0, 5])

    F1_Stroma = 2 * Stroma_Pre * Stroma_Recall / (Stroma_Pre + Stroma_Recall)
    F1_Normal = 2 * Normal_Pre * Normal_Recall / (Normal_Pre + Normal_Recall)
    F1_G3 = 2 * G3_Pre * G3_Recall / (G3_Pre + G3_Recall)
    F1_G4 = 2 * G4_Pre * G4_Recall / (G4_Pre + G4_Recall)
    F1_G5 = 2 * G5_Pre * G5_Recall / (G5_Pre + G5_Recall)

    Weighted_average_F1score = 0.25 * F1_G3 + 0.25 * F1_G4 + 0.25 * F1_G5 + 0.125 * F1_Normal + 0.125 * F1_Stroma

    print(" %s = %.4f " % ('Weighted_average_F1score ', Weighted_average_F1score))
    print(Weighted_average_F1score)



if __name__ == "__main__":
    main()

