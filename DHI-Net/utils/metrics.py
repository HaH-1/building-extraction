import level_inter as li

def cal_cm(y_true,y_pred):
    shape = y_true.shape
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    total_cm = []
    tp,tn,fp,fn = 0,0,0,0
    for i in range(shape[0]):
        y_true=y_true.reshape(1,-1).squeeze()
        y_pred=y_pred.reshape(1,-1).squeeze()
        cm=li.metrics.confusion_matrix(y_true,y_pred)

        if len(cm[0] == 1):
            tp += 0
            fp += 0
            fn += 0
        else:
            tp += cm[1][1]
            tn += cm[0][0]
            fp += cm[0][1]
            fn += cm[1][0]
        total_cm.append(cm)

    # return tp,tn,fp,fn
    return total_cm

def tp_fn(y_true,y_pred):
    shape = y_true.shape
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    total_cm = []
    tp,tn,fp,fn = 0,0,0,0
    for i in range(shape[0]):
        y_true=y_true.reshape(1,-1).squeeze()
        y_pred=y_pred.reshape(1,-1).squeeze()
        cm=li.metrics.confusion_matrix(y_true,y_pred)
        # print(cm)
        # print(cm)
        if len(cm[0]) == 1:
            tp += 0
            fp += 0
            fn += 0
        else:
            tp += cm[1][1]
            tn += cm[0][0]
            fp += cm[0][1]
            fn += cm[1][0]
        total_cm.append(cm)
    # print(tp)
    return tp,tn,fp,fn

def Precision(tp,fp):
    return tp / (tp + fp)

def Recall(tp,fn):
    return tp / (tp + fn)

def IoU(tp,fp,fn):
    return tp / (tp + fp + fn)
