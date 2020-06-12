import sklearn.metrics as sklearnMetrics
from keras import backend as K

#1d array-like
def getPrecision(y_true, y_pred,average='weighted'): #average=None if we want metric per class
    precision = sklearnMetrics.precision_score(y_true, y_pred=y_pred, average=average) #precission per class
    #sklearnMetrics.precision_score(y_true, y_pred=pred, average='weighted') #precission in general
    return precision

def getRecall(y_true, y_pred,average='weighted'):
    recall = sklearnMetrics.recall_score(y_true, y_pred=y_pred, average=average) #precission per class
    #sklearnMetrics.recall_score(y_true, y_pred=pred, average='weighted') #precission in general
    return recall

def getConfussionMatrix(y_true, y_pred, classes):
    cm = sklearnMetrics.confusion_matrix(y_true, y_pred, classes)
    return cm

def getAccucacy(y_true, y_pred):
    accuracy = sklearnMetrics.accuracy_score(y_true, y_pred)
    return accuracy

def getF1(y_true, y_pred, average='weighted'):
    f1 = sklearnMetrics.f1_score(y_true, y_pred, average=average)
    return f1

def get_specificity(FP, TN):
    return TN/(FP+TN)

def getFPR(FP,TN):
    return FP/(FP+TN)

def getFOR(FN, TN):
    return FN/(FN+TN)

def getNPV(FN, TN):
    return TN/(FN+TN)


def get_summary_metrics(train_pred, labels_train, test_pred, labels_test,classes= [1,-1], saveData=False, path2save = ""):
    # METRICS DEFINITION IN: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
    precission_train = getPrecision(y_pred=train_pred, y_true=labels_train)
    recall_train = getRecall(y_pred=train_pred, y_true=labels_train)
    accuracy_train = getAccucacy(y_pred=train_pred, y_true=labels_train)
    F1_train = getF1(y_pred=train_pred, y_true=labels_train)
    confussion_matrix_train = getConfussionMatrix(y_pred=train_pred, y_true=labels_train, classes=classes)
    TP_train = confussion_matrix_train[0][0]
    TN_train = confussion_matrix_train[1][1]
    FP_train = confussion_matrix_train[1][0]
    FN_train = confussion_matrix_train[0][1]
    specificity_train = get_specificity(FP_train, TN_train)
    FPR_train = getFPR(FP_train, TN_train)
    FOR_train = getFOR(FN_train, TN_train)
    NPV_train = getNPV(FN_train, TN_train)

    precission_test =getPrecision(y_pred=test_pred, y_true=labels_test)
    recall_test = getRecall(y_pred=test_pred, y_true=labels_test)
    accuracy_test = getAccucacy(y_pred=test_pred, y_true=labels_test)
    F1_test = getF1(y_pred=test_pred, y_true=labels_test)
    confussion_matrix_test = getConfussionMatrix(y_pred=test_pred, y_true=labels_test, classes=classes)
    TP_test = confussion_matrix_train[0][0]
    TN_test = confussion_matrix_train[1][1]
    FP_test = confussion_matrix_train[1][0]
    FN_test = confussion_matrix_train[0][1]
    specificity_test = get_specificity(FP_test,TN_test)
    FPR_test = getFPR(FP_test, TN_test)
    FOR_test = getFOR(FN_test, TN_test)
    NPV_test = getNPV(FN_test, TN_test)
    print("....................METRICS.................................")
    print(path2save)
    print("------TRAIN-------")
    print('Precission train: ' + str(precission_train))
    print('Recall train: ' + str(recall_train))
    print("Accuracy train: " + str(accuracy_train))
    print("F1 train: " + str(F1_train))
    print("classes: "+str(classes[0]) +","+str(classes[1]))
    print("Confussion matrix train:" + str(confussion_matrix_train))
    print("TP: "+ str(confussion_matrix_train[0][0]))
    print("TN: " + str(confussion_matrix_train[1][1]))
    print("FN: " + str(confussion_matrix_train[0][1]))
    print("FP: " + str(confussion_matrix_train[1][0]))
    print("Specificity "+str(specificity_train))
    print("FPR " + str(FPR_train))
    print("FOR " + str(FOR_train))
    print("NPV " + str(NPV_train))
    print("-------TEST-------")
    print('Precission test: ' + str(precission_test))
    print('Recall test: ' + str(recall_test))
    print("Accuracy test: " + str(accuracy_test))
    print("F1 test" + str(F1_test))
    print("classes: " + str(classes[0]) + "," + str(classes[1]))
    print("Confussion matrix test:" + str(confussion_matrix_test))
    print("TP: " + str(confussion_matrix_test[0][0]))
    print("TN: " + str(confussion_matrix_test[1][1]))
    print("FN: " + str(confussion_matrix_test[0][1]))
    print("FP: " + str(confussion_matrix_test[1][0]))
    print("Specificity " + str(specificity_test))
    print("FPR " + str(FPR_test))
    print("FOR " + str(FOR_test))
    print("NPV " + str(NPV_test))
    print("........................................................")

    if(saveData):
        if(path2save==""):
            path2save= "/home/cris/PycharmProjects/InterSpeech19/data/results/baseline/default.txt"
        with open(path2save, "w") as f:
            f.write("-----------TRAIN-------"+"\n")
            f.write("Precission: "+str(precission_train)+"\n")
            f.write("Recall: " + str(recall_train)+"\n")
            f.write("Accuracy:" + str(accuracy_train)+"\n")
            f.write("F1:" + str(F1_train)+"\n")
            f.write("Classes: " + str(classes)+"\n")
            f.write("Confussion matrix train:" + str(confussion_matrix_train)+"\n")
            f.write("-----------TEST-------"+"\n")
            f.write("Precission test: " + str(precission_test)+"\n")
            f.write("Recall test: " + str(recall_test)+"\n")
            f.write("Accuracy test: " + str(accuracy_test)+"\n")
            f.write("F1 test: " + str(F1_test)+"\n")
            f.write("Classes: " + str(classes) + "\n")
            f.write("Confussion matrix test:" + str(confussion_matrix_test)+"\n")
        with open("/home/cris/PycharmProjects/InterSpeech19/data/results/baseline/common_metrics.txt", "a+") as f:
            f.write(path2save+";"+str(confussion_matrix_train).replace("\n","")+";"+str(confussion_matrix_test).replace("\n","")+"\n")
    return precission_train, recall_train, accuracy_train, F1_train, confussion_matrix_train, precission_test, recall_test, accuracy_test, F1_test, confussion_matrix_test, NPV_train, NPV_test

#Previous implementations of keras functions
def keras_precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def keras_recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def keras_fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return keras_fbeta_score(y_true, y_pred, beta=1)


def keras_fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = keras_precision(y_true, y_pred)
    r = keras_recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score