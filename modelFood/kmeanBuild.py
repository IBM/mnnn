from sklearn.cluster import KMeans
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


src_angle1=np.load("../foodData/tgt_angle1_200.npy")
print(src_angle1)

angle1=[]
for l in src_angle1:
    for a in l:
        angle1.append(np.float_(a))
print(len(angle1))

src_angle2=np.load("../foodData/tgt_angle2_200.npy")
#print(src_angle2)

angle2=[]
for l in src_angle2:
    for a in l:
        angle2.append(np.float_(a))

print(len(angle2))

angle=[]

angle1, angle2= shuffle(angle1, angle2)

angle1=angle1[:300000]
angle2=angle2[:300000]

for i in range(len(angle1)):
    angle.append([angle1[i],angle2[i]])
print(len(angle))

plt.scatter(angle1,angle2,marker='o',s=0.1)
plt.title( str(len(angle1))+ " samples ")
#plt.show()

kmeans_angle_100c=KMeans(n_clusters=100,random_state=0,verbose=True).fit(angle)
kmeans_center=np.asarray(kmeans_angle_100c.cluster_centers_)

joblib.dump(kmeans_angle_100c, '100c_300000s_kmean_angle_model.pkl')
print(kmeans_center)
