from matplotlib import pyplot as plt
import torch 
import os
import re
import time
import numpy as np

def main():
#    fp = open("/home/liucl/Proj/kws-pytorch/act_value/value/candidate_mm", "r")
#    fp = open("/home/liqin/kws-gru/act_value/input_0", "r")
#    lines = fp.readlines()

#    with open("/home/liqin/kws-gru/act_value/input_0",'rb') as fp:
#        lines = fp.readlines()
    
    # We can use the Log amplifier, so just use the Log function.
    # The range of Mel-frequency do no influence the accuracy.

    filedir = "/home/liqin/kws-gru/gru_10nodes_feature-divide10/yes-1w10n10in70unknown0.01lr/act_value/"
    filelist = os.listdir(filedir)
    pattern  = re.compile('Before_inputgate_[0-9]')

    # input inputgate resetgate newgate hy gate_x_mm gate_x gate_h gate_h_mm candidate_mm candidate
    # Before_resetgate Before_newgate Before_inputgate

    data = []
    
    for i in filelist:
        filename = os.path.splitext(i)[0]
        if pattern.match(filename):
            fp = torch.load(filedir+filename)
            liness = fp.tolist()

            for lines in liness:
                for line in lines:
                    if np.isnan(float(line)):
                        continue
                    data.append(float(line))
                #print(lines)
            print(filename)

    min_value = min(data)
    max_value = max(data)

    print(min_value)
    print(max_value)
    
    #ft = open('./test_hist.txt','w+')
    #print(data,file=ft)
    
    plt.hist(data,100)
    plt.show()
    #step = (max_value-min_value)/100

if __name__=="__main__":
    main()
