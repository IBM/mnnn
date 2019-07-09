import numpy as np
import math

with open("modelFood/modelSettings.txt") as f:
    modelSettings = f.readlines()

# print(modelSettings)
    use_triangle = False if (modelSettings[0].strip().split(":")[1] == "False") else True

phase="val"

if(phase=="train"):
    a=np.load("foodData/src_angle_200.npy")
    a=(np.array(a[:20000]))
elif(phase=="val"):
    a = np.load("foodData/src_angle_200.npy")
    a = (np.array(a[20000:]))
    l= np.load("foodData/src_angleLabel_200.npy")
    l= (np.array(l[20000:]))

elif(phase=="infer"):
    a=np.load("foodData/doinfer.npy")



angle1Loss=np.load("results/npyRes/ang1Loss.npy")
angle2Loss=np.load("results/npyRes/ang2Loss.npy")
print("test:", len(angle1Loss))

if(use_triangle):

    for i in range(len(angle1Loss)):
        for j in range(len(angle1Loss[i])):
            angle1Loss[i][j]= min( abs(angle1Loss[i][j]), 360- abs(angle1Loss[i][j]) )

    for i in range(len(angle2Loss)):
        for j in range(len(angle2Loss[i])):
            angle2Loss[i][j]= min(abs(angle2Loss[i][j]), 360- abs(angle2Loss[i][j]) )

thefile = open('./debug.txt', 'w')

num=0
loss1L1=0
loss1L2=0
loss2L1=0
loss2L2=0
for i in range(len(a)):
    # loss1L1 = 0
    # loss1L2 = 0
    # loss2L1 = 0
    # loss2L2 = 0
    # num=0
    #thefile.write("pred angle2: " + "%s\n" % angle2Loss[i])
    for j in range(len(a[i])):


            num=num+1
            loss1L1=loss1L1+ abs( angle1Loss[i][j])
        #print(loss1L1)
            loss1L2=loss1L2+angle1Loss[i][j]**2


            loss2L1 = loss2L1 + abs(angle2Loss[i][j])

            #thefile.write("pred angle2: " + "%s\n" %  (str(loss2L1)+"  "+ str(abs(angle2Loss[i][j]))))


            loss2L2 = loss2L2 + angle2Loss[i][j] ** 2


    # print("overall angle1 l1 loss", loss1L1/num)
    # print("overall angle2 l1 loss", loss2L1/num)
    # print("overall angle1 l2 loss", loss1L2/num)
    # print("overall angle2 l2 loss", loss2L2/num)
    # print("-----------------------",i)

print("overall angle1 l1 loss", loss1L1/num)
print("overall angle2 l1 loss", loss2L1/num)
print("overall angle1 l2 loss", loss1L2/num)
print("overall angle2 l2 loss", loss2L2/num)

# overall angle1 l1 loss 30.0361517506
# overall angle2 l1 loss 47.2247654951

# overall angle1 l2 loss 3189.6693917
# overall angle2 l2 loss 6352.57799234


num = 0
loss1L1 = 0
loss1L2 = 0
loss2L1 = 0
loss2L2 = 0
for i in range(len(l)):
    # loss1L1 = 0
    # loss1L2 = 0
    # loss2L1 = 0
    # loss2L2 = 0
    # num=0
    # thefile.write("pred angle2: " + "%s\n" % angle2Loss[i])
    for j in range(len(l[i])):
        #print(l[i][j])
        if(l[i][j]=='T'):
            num = num + 1
            loss1L1 = loss1L1 + abs(angle1Loss[i][j])
            # print(loss1L1)
            loss1L2 = loss1L2 + angle1Loss[i][j] ** 2

            loss2L1 = loss2L1 + abs(angle2Loss[i][j])

            # thefile.write("pred angle2: " + "%s\n" %  (str(loss2L1)+"  "+ str(abs(angle2Loss[i][j]))))


            loss2L2 = loss2L2 + angle2Loss[i][j] ** 2


        # print("overall angle1 l1 loss", loss1L1/num)
        # print("overall angle2 l1 loss", loss2L1/num)
        # print("overall angle1 l2 loss", loss1L2/num)
        # print("overall angle2 l2 loss", loss2L2/num)
        # print("-----------------------",i)

print("overall angle1 l1 loss_T", loss1L1 / num)
print("overall angle2 l1 loss_T", loss2L1 / num)
print("overall angle1 l2 loss_T", loss1L2 / num)
print("overall angle2 l2 loss_T", loss2L2 / num)




