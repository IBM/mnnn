import numpy as np


index=np.load("../../foodData/doinfer.npy")
ang1Pred= np.load("ang1Pred.npy")
ang2Pred= np.load("ang2Pred.npy")
model_t="MNNN"

#print(ang1Pred)
with open("../../foodData/testData.txt") as f:
    dd = f.readlines()

ddd=[]
for d in dd:
    if(len(d.strip())==0):
        print("empty line")
        continue
    ddd.append(d)

dd=ddd
# print(ang1Pred[0])
split=0
count = 0
for i in range(len(ang1Pred)):
    # name=dd[i].strip().split("(")[-1]
    # name=name.strip().split(")")[0]

    name=dd[i].strip().split(" ")[1]



    org_seq=dd[i].strip().split(" ")[0]
    orgName=name


    name=name+"_seq_No_"+str(count)

    #print(name)
    l=len(index[i])

    # print(index[i])
    print("model name:", model_t,  " sequence name:",name, " length: " , l,file=open("output.txt", "a") )

    a1=ang1Pred[i][:l]
    a1[0]=180
    a1=[str(x) for x in a1]

    a2 = ang2Pred[i][:l]
    a2[-1] = 180
    a2 = [str(x) for x in a2]


    if(orgName[0]=='*'):
        split=split+1
        continue
    #thefile = open("model" + model_t + '_split' + str(split) + "/" + name + '_' + str(model_t) + '.txt', 'w')
    count = count + 1
    for j in range(len(a1)):

        print(org_seq[j]+" "+a1[j]+" "+a2[j],file=open("output.txt", "a"))
        newl = org_seq[j]+" "+a1[j]+" "+a2[j]+ "\n"
        #thefile.write(newl)

    # print(" ".join(a1))
    # print(( " ".join(a1)) , file=open("output.txt", "a"))
    # print((" ".join(a2) ),file=open("output.txt", "a"))

    print("",file=open("output.txt", "a"))


#
# a="GSHMKNSVSVDLPGSMKVLVSKSSNADGKYDLIATVDALELSGTSDKNNGSGVLEGVKADASKVKLTISDDLGQTTLEVFKSDGSTLVSKKVTSKDKSSTYELFNEKGELSFKYITRADKSSTYELFNEKGELSFKYITRADKSSTYELFNEKGELSFKYITRADKSSTYELFNEKGELSFKYITRADGTRLEYTGIKSDGSGKAKEVLKGYVLEGTLTAEKTTLVVKEGTVTLSKNISKSGEVSVELNDTDSSAATKKTAAWNSGTSTLTITVNSKKTKDLVFTSSNTITVQQYDSNGTSLEGSAVEITKLDEIKNALK"
# print(len(a))