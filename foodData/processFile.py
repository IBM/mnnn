import numpy as np
with open("./foodData/doinfer.txt") as f:
    data=f.readlines()

seq=[]
ang1=[]
ang2=[]

for i in range(len(data)):
    if(i%3==0):
        line=np.asarray(list(data[i].strip()))
        seq.append(line)
    elif(i%3==1):
        ang1s=np.asarray(data[i].strip().split(" "))
        ang1.append(ang1s[np.where(ang1s!='')])
    else:
        ang2s=np.asarray(data[i].strip().split(" "))
        ang2.append(ang2s[np.where(ang2s!='')])

#
# res=np.asarray(res)
np.save("doinfer.npy", np.asarray(seq))
np.save("doinfer_ang1_tgt.npy", np.asarray(ang1))
np.save("doinfer_ang2_tgt.npy", np.asarray(ang2))


print(seq)
print(ang1)
print(ang2)