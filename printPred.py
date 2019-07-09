import numpy as np
a=np.load("/Users/ibm_siyuhuo/Github Repo/seq2seq/foodData/doinfer.npy")
ang1Pred=np.load("results/npyRes/ang1Pred.npy")
ang2Pred=np.load("results/npyRes/ang2Pred.npy")

thefile = open('./res_angle.txt', 'w')

for i in range(len(a)):
    tmp1=""
    tmp2=""
    tmp3=""
    for j in range(len(a[i])):

        tmp1=tmp1+str(a[i][j])+" "
        tmp2 = tmp2 + str(ang1Pred[i][j]) + " "
        tmp3 = tmp3 + str(ang2Pred[i][j]) + " "
    #
    # print(len(tmp1.split(" ")))
    # print(len(tmp2.split(" ")))
    # print(len(tmp3.split(" ")))

    thefile.write("seq: " + "%s\n" % tmp1.strip())
    thefile.write("preds angle1: " + "%s\n" % tmp2.strip())
    thefile.write("preds angle2: " + "%s\n" % tmp3.strip())

