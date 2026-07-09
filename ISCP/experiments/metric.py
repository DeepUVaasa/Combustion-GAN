
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score,classification_report,confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def evaluate(labels, scores,res_th=None, saveto=None):
    '''
    metric for auc/ap
    :param labels:
    :param scores:
    :param res_th:
    :param saveto:
    :return:
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, ths = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    print('ths from roc_curve:', len(ths))
    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure(figsize=(8,6))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    # best f1
    best_f1 = 0
    best_threshold = 0
    for threshold in ths:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= threshold] = 1
        tmp_scores[tmp_scores < threshold] = 0
        cur_f1 = f1_score(labels, tmp_scores)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold


    #threshold f1
    if res_th is not None : #and saveto is  not None
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= res_th] = 1
        tmp_scores[tmp_scores < res_th] = 0

    res_th = best_threshold
    tmp_scores = scores.copy()
    tmp_scores[tmp_scores >= res_th] = 1
    tmp_scores[tmp_scores < res_th] = 0

    gt = labels 
    pred = tmp_scores
    #print('Before Adjustment:')
    auc_prcB=average_precision_score(labels,scores)
    f1_score_val = f1_score(labels,pred)
    print('AUC:', roc_auc, 'AP:', auc_prcB, 'F1 score:', f1_score_val)

    accuracy = accuracy_score(gt, pred)
    print('Confusion_matrix:', confusion_matrix(gt, pred))
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
    """
    print('After adjustment:')
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
        
    pred = np.array(pred)
    gt = np.array(gt)

    auc_prc=average_precision_score(gt, pred)
    print('ap:', auc_prc)

    accuracy = accuracy_score(gt, pred)
    print('Confusion_matrix:', confusion_matrix(gt, pred))
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

    """
    return auc_prcB,roc_auc,best_threshold,best_f1
