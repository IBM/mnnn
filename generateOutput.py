
import numpy as np
import math
import datetime

with open("modelFood/modelSettings.txt") as f:
    modelSettings = f.readlines()

# print(modelSettings)
    use_triangle = False if (modelSettings[0].strip().split(":")[1] == "False") else True

thefile = open('./res_angle.txt', 'w')
thefile.write("date:" + str(datetime.datetime.now()))


def arcAngle(s,c):

    if(s>1):
        s=1
    if(c>1):
        c=1
    if(s<-1):
        s=-1
    if(c<-1):
        c=-1

    tmp1=np.arcsin(s) #-0.5pi ~ 0.5pi
    tmp2=np.arccos(c) # 0 ~ pi
    if(math.isnan(tmp2)):
        print("nan",[s,c])

    if(s >= 0.000 and c >=  0.000 ): #phase1
        return tmp1
    elif(s>=0.000 and c<= 0.000): #phase2
        return tmp2
    elif(s<=0.000 and c<=0.000): #phase3
        return -tmp2
    elif(s<=0.000 and c>=0.000):
        return tmp1 #phase4
def getAngle(s,c):
    tmp=arcAngle(s,c)
    return tmp/np.pi*180


print("ANG",getAngle(0,-1))

def tansform_angle(s_a,c_a):
    res=[]
    for i in range(len(s_a)):
        a = getAngle(s_a[i],c_a[i])
        res.append(a)
    return res


ang1Loss=[]
ang2Loss=[]

preds= np.load("results/npyRes/preds.npy")
ang1=np.load("results/npyRes/ang1.npy")
ang2= np.load("results/npyRes/ang2.npy")

print("output data:",preds.shape)
res_ang1=[]
res_ang2=[]

# print(preds.shape)

for i in range(ang1.shape[0]):
    if(i%10000==0):
        print(i)
    for j in range(ang1[i].shape[0]):
        ang1_tgt=ang1[i][j]
        #thefile.write("tgt angle1: " + "%s\n" % ang1_tgt)

        ang2_tgt=ang2[i][j]
        #thefile.write("tgt angle2: " + "%s\n" % ang2_tgt)

        if(use_triangle==True):

            ang1_pred = preds[i][j][:, 0]
            #hefile.write("pred_sin1: "+"%s\n" % ang1_pred)

            ang2_pred = preds[i][j][:, 1]
            #thefile.write("pred_cos1: "+"%s\n" % ang2_pred)
            ang1_pred_trans= tansform_angle(ang1_pred, ang2_pred)
            res_ang1.append(ang1_pred_trans)
            #thefile.write("pred angle1: " + "%s\n" % ang1_pred_trans)

            ang3_pred = preds[i][j][:, 2]
            #thefile.write("pred_sin2: "+"%s\n" % ang3_pred)

            ang4_pred = preds[i][j][:, 3]
            #thefile.write("pred_cos2: "+"%s\n" % ang4_pred)
            ang2_pred_trans=tansform_angle(ang3_pred, ang4_pred)
            res_ang2.append(ang2_pred_trans)
            #thefile.write("pred angle2: " + "%s\n" % ang2_pred_trans)
            #thefile.write("\n")

            ang1Loss.append(np.asarray(ang1_tgt) - np.asarray(ang1_pred_trans))
            ang2Loss.append(np.asarray(ang2_tgt) - np.asarray(ang2_pred_trans))
        else:
            ang1_pred=preds[i][j][:,0]
            res_ang1.append(ang1_pred)
            #thefile.write("preds angle1: " + "%s\n" % ang1_pred)

            ang2_pred = preds[i][j][:,1]
            res_ang2.append(ang2_pred)
            #thefile.write("preds angle2: " + "%s\n" % ang2_pred)


            ang1Loss.append(np.asarray(ang1_tgt) -np.asarray(ang1_pred))
            ang2Loss.append(np.asarray(ang2_tgt) - np.asarray(ang2_pred))


print("save!")
np.save("results/npyRes/ang1Loss.npy",ang1Loss)
np.save("results/npyRes/ang2Loss.npy",ang2Loss)

np.save("results/npyRes/ang1Pred.npy",res_ang1)
np.save("results/npyRes/ang2Pred.npy",res_ang2)

