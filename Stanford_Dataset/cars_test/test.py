import scipy.io
import numpy as np
# mat = scipy.io.loadmat('cars_train_annos.mat')
# matmeta = scipy.io.loadmat('cars_meta.mat')
# print(mat)
# print(mat['annotations'][0])
#
# d = {}
# for i in mat['annotations'][0]:
#     # print(i[0],i[1],i[2],i[3],i[4], i[5][0])
#     d[i[5][0]] = [i[0][0][0],i[1][0][0],i[2][0][0],i[3][0][0],i[4][0][0]]
#
# print(d)
# np.save("cars_train_annos.npy",d)


import os

print(os.walk("."))
for r, d, f in os.walk("./cars_test"):
    np.save('cars_test.npy', f)


li = np.load('cars_test.npy')

for i in li:
    print(i)
