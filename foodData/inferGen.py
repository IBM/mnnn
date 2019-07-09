import numpy as np
import numpy as np
import tensorflow as tf
print(tf.__version__)
np.set_printoptions(threshold=np.nan)


# a=np.load("doinfer_ang1_tgt.npy")
#
with open("testData.txt") as f:
    dd = f.readlines()
c=[]
ang1=[]
ang2=[]
label=[]
count=-1
for d in dd:
    count = count + 1
    # if(count==100000):
    #     break
    if(len(d.strip())==0):
        print("empty line")
        continue

    d=d.strip().split(" ")[0]
    #print(d)
    #print(len(d))
    tmp=[]
    tmp_ang1 = []
    tmp_ang2 = []
    tmp_label=[]
    for t in d:


        tmp.append(t)
        tmp_ang1.append(1)
        tmp_ang2.append(1)
        tmp_label.append("~")

    c.append(tmp)
    ang1.append(tmp_ang1)
    ang2.append(tmp_ang2)
    label.append(tmp_label)

# res=np.asarray(res)
#
print("lengths",len(ang1))

np.save("doinfer.npy",c)
np.save("doinfer_ang1_tgt.npy",ang1)
np.save("doinfer_ang2_tgt.npy",ang2)
np.save("doinfer_label.npy",label)


#
# thefile = open('printseq.txt', 'w')
# a=np.load("src_angle_200.npy")[:20]
# for l in a:
#
#     thefile.write("".join(l)+"\n" )
#
#
# a_s=[]
# for aa in a:
#     tmp=" ".join(aa)
#     a_s.append(tmp)
#
# b=np.load("tgt_angle1_200.npy")[:2]
# c=np.load("src_angleLabel_200.npy")[:2]
# print("a",a)
# print("a_s",a_s)
# print("b",b)
# print("c",c)
#
# for l in b:
#     lena.append(len(l))
# print(len(lena))
# print("mean",np.mean(lena))
# #b=np.load("src_angle_200_9gram_central_char.npy")
#
# print(b)
#
# for i in range(100):
#     print(np.asarray(b[i]).shape==np.asarray(a[i]).shape==np.asarray(c[i]).shape)
#     # print(np.asarray(b[i]).shape)
#     # print(np.asarray(a[i]).shape)
#     # print(np.asarray(c[i]).shape)
# #a=(np.array(a[20000:]))
# #print(a[0])
# t=[]
# s_l=[]
# for i in a:
#
#     i=list(np.float_(i))
#     s_l.append(sum(i)/len(i))
#
#     #print(i)
#     t=t+i
# #list(np.float_(a))
# print(s_l)
# t=np.asarray(t)
# count=0
# print(t.shape)
# # print(t)
# for item in t:
#     if (item >= 0):
#         count = count + 1
#
# print('pos ratio',float(count)/t.shape[0])
# print('avg for whole ',np.average(t))
# print('std for whole ', np.std(t))
#
# print("avg for sequence mean ", np.average(s_l))
# print("std for sequence mean", np.std(s_l))
#
#
# a=np.load("tgt_angle2_200.npy")
# #b=np.load("src_angle_200.npy")
# #b=np.load("src_angle_200_9gram_central_char.npy")
# a=(np.array(a[:5000]))
# t=[]
# s_l=[]
# for i in a:
#
#     i=list(np.float_(i))
#     s_l.append(sum(i)/len(i))
#     #print(i)
#     t=t+i
# #list(np.float_(a))
#
# t=np.asarray(t)
#
# print(t.shape)
# # print(t)
# count=0
# for item in t:
#     if(item<=0):
#         count=count+1
#
# print('pos ratio',float(count)/t.shape[0])
# print('avg for whole ',np.average(t))
# print('std for whole ', np.std(t))
#
# print("avg for sequence mean ", np.average(s_l))
# print("std for sequence mean", np.std(s_l))

# for item in a:
#     print(item)
#print("a",a)
#print("b",b)
# print("b",b[0][0:10])
# for i in b[0]:
#     print(i)
