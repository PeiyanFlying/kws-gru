import numpy as np

def evaluate(result, target):
    # assert len(result) == len(target)
    # print(target)
    # print(result)
    _, pred = result.topk(1, 1, True, True)
    result = pred.t()[0].tolist()
    for i in range(len(target)):
        result[i] = 0 if result[i] == 0 or result[i] == 1 else 1
        target[i] = 0 if target[i] == 0 or target[i] == 1 else 1

    xor = [a ^ b for a, b in zip(target, result)]
    miss = sum([a & b for a, b in zip(xor, target)])
    false_accept = sum([a & b for a, b in zip(xor, result)])
    return miss, sum(target), false_accept, len(target)

def ROC(results, target,save_dir):
    #print(results.shape[0])
    #print(len(target))
    # assert len(result) == len(target)
    # print(target)
    # print(result)
    threds = np.arange(1,-0.1,-0.001).tolist()
    roc = []
    for i in range(len(target)):
        target[i] = 0 if target[i] == 0 or target[i] == 1 else 1

    for thred in threds:
        #print(thred)
        softmax = results.tolist()
        result = []
        for i in range(len(target)):
            tmp = 1 if softmax[i][2] >= thred else 0
            result.append(tmp)
        #result = pred.t()[0].tolist()
       
        xor = [a ^ b for a, b in zip(target, result)]
        miss = sum([a & b for a, b in zip(xor, target)])
        false_accept = sum([a & b for a, b in zip(xor, result)])
        false_reject_rate = miss/sum(target)
        false_alarm_rate = false_accept/(len(target) - sum(target))
        roc.append((false_reject_rate, false_alarm_rate))
    fpw = open(save_dir+'/roc.txt', 'w')
    fpw.write('FRR\tFAR\n')
    for i,j in roc:
        fpw.write('%.3f\t%.3f\n' % (i, j))
    fpw.close()
