import pandas as pd
import numpy as np
import time


# read the file with scores and the file with labels into panda frames
pd_solutions = pd.read_csv('solution.csv')
pd_multiframe_submission = pd.read_csv('submission.csv')

# extract meaningful information from panda frames into numpy arrays
# every ligand has multiple frames in "predictions"
# predictions_by_ligand reduces multiple positions of the same ligand into one single prediction
predictions_by_ligand_names = np.array(list(pd_multiframe_submission[pd_multiframe_submission.columns[0]]))
multiframe_predictions = np.array(pd_multiframe_submission[pd_multiframe_submission.columns[1:]])
predictions_by_ligand = np.mean(multiframe_predictions[:,0:6],axis=1)
#predictions_by_ligand = np.mean(multiframe_predictions,axis=1)

solutions_by_ligand_names = np.array(pd_solutions.ix[:,0])
solutions_by_ligand = np.array(pd_solutions.ix[:,1])

# get the list of receptors (receptor is a base of the ligand's name ie nsbb for nsbp_10101)
def receptor_name_from_ligand_name(ligand_name):
    return ligand_name.split("_")[0]
all_receptors = sorted(set(map(receptor_name_from_ligand_name,predictions_by_ligand_names)))

# for each receptor append all of it's ligands and predictions into two lists
number_of_receptors = len(all_receptors)
predictions_by_receptor_names = [[] for _ in xrange(number_of_receptors)]
predictions_by_receptor = [[] for _ in xrange(number_of_receptors)]

def append_to_predictions(ligand_name,prediction,predictions_by_ligand_names=predictions_by_ligand_names,predictions_by_ligand=predictions_by_ligand):
    receptor_name = ligand_name.split("_")[0]
    receptor_idx = all_receptors.index(receptor_name)
    predictions_by_receptor_names[receptor_idx].append(ligand_name)
    predictions_by_receptor[receptor_idx].append(prediction)
    return None
    
map(append_to_predictions,predictions_by_ligand_names,predictions_by_ligand)

# for each receptor sort it's ligands by predictions
for receptor_idx in range(number_of_receptors):
    order = np.flipud(np.asarray(predictions_by_receptor[receptor_idx]).argsort())
    predictions_by_receptor[receptor_idx] = list(np.asarray(predictions_by_receptor[receptor_idx])[order])
    predictions_by_receptor_names[receptor_idx] = list(np.asarray(predictions_by_receptor_names[receptor_idx])[order])
  
    print predictions_by_receptor

# map each of the ligands to it's label
list_solutions_by_ligand_names = list(solutions_by_ligand_names)
labels_by_receptor = [[] for _ in xrange(number_of_receptors)]

def ligand_name_to_label_index(ligand_name,solutions_by_ligand_names=solutions_by_ligand_names,solutions_by_ligand=solutions_by_ligand):
    label_index = list_solutions_by_ligand_names.index(ligand_name)
    return label_index



for receptor_idx in range(number_of_receptors):
    label_indexes_for_receptor = np.squeeze(map(ligand_name_to_label_index,predictions_by_receptor_names[receptor_idx]))
    labels_for_receptor = list(solutions_by_ligand[label_indexes_for_receptor])
    labels_by_receptor[receptor_idx].append(labels_for_receptor)


# calculate AUC for each of the receptors"""


def auc_from_labels(labels):
    """calculates area under the curve AUC for binary predictions/labels needs
    sorted in descending order predictions"""

    # calculate the number of positive labels
    num_positives = np.sum(np.asarray(labels, dtype=bool) == True)

    num_predictions = len(labels)
    #print "predictions:",labels
    print "num predictions:",num_predictions


    # return NAN when no positives are present
    if num_positives == 0:
        return float('nan')

    # slide from top to the bottom;
    # each time slide the threshold so as to predict one more label as positive
    roc_curve = np.array([0.0, 0.0])
    TP_above_threshold = 0


    for idx in range(num_predictions):
        if labels[idx] == True:
            TP_above_threshold += 1

        # calculate True Positives Rate
        # TPR = TP / num_real_positives
        TPR = TP_above_threshold / float(num_positives)

        # FPR = FP / num_real_negatives
        FPR = (idx + 1 - TP_above_threshold) / (num_predictions - float(num_positives))

        roc_curve = np.vstack((roc_curve, [FPR, TPR]))

    roc_curve = np.vstack((roc_curve, [1.0, 1.0]))

    # reduce into TP and FP rate, integrate with trapezoid to calculate AUC
    auc = np.trapz(roc_curve[:, 1], x=roc_curve[:, 0])

    return auc


aucs = np.array([])
i = 0
for labels_for_receptor in labels_by_receptor:
  
    auc = auc_from_labels(np.squeeze(labels_for_receptor))
    aucs = np.append(aucs,auc)
    print "receptor:",all_receptors[i]
    print auc
    i+=1
print "average:",np.average(aucs)

def auc(self, predictions, labels):
    """calculates area under the curve AUC for binary predictions/labels needs
    sorted in descending order predictions"""

    # sort the array by predictions in descending order in case it has not been done
    order = np.flipud(predictions.argsort())
    labels = labels[order]

    # clean labels, calculate the number of positive labels
    labeled_true = (np.asarray(labels, dtype=bool) == True)
    num_positives = np.sum(labeled_true)
    num_predictions = len(labeled_true)

    # If no positive result return nan
    if num_positives == 0:
        return float('nan')

    # slide from top to the bottom;
    # each time slide the threshold so as to predict one more label as positive
    roc_curve = np.array([0.0, 0.0])
    TP_above_threshold = 0
    for predict_as_positive in range(num_predictions):
        if labeled_true[predict_as_positive] == True:
            TP_above_threshold += 1
        # calculate True Positives Rate
        # TPR = TP / num_real_positives
        TPR = TP_above_threshold / float(num_positives)

        # FPR = FP / num_real_negatives
        FPR = (predict_as_positive + 1 - TP_above_threshold) / (num_predictions - float(num_positives))

        roc_curve = np.vstack((roc_curve, [FPR, TPR]))

    roc_curve = np.vstack((roc_curve, [1.0, 1.0]))

    # reduce into TP and FP rate, integrate with trapezoid to calculate AUC
    auc = np.trapz(roc_curve[:, 1], x=roc_curve[:, 0])

    return auc
