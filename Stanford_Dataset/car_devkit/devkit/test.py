import scipy.io
import numpy as np
# mat = scipy.io.loadmat('cars_train_annos.mat')
# matt = scipy.io.loadmat('cars_test_annos.mat')
# matmeta = scipy.io.loadmat('cars_meta.mat')
# # print(mat)
# print(matt)
# print(matt['annotations'][0])
#
# d = {}
# for i in mat['annotations'][0]:
#     # print(i[0],i[1],i[2],i[3],i[4], i[5][0])
#     try:
#         l = d[i[5][0]]
#         print("--->",l)
#         b = []
#         for i in l:
#             b.append(l)
#         b.append([i[0][0][0],i[1][0][0],i[2][0][0],i[3][0][0],i[4][0][0]])
#         d[i[5][0]] = b
#     except:
#         d[i[5][0]] = [[i[0][0][0],i[1][0][0],i[2][0][0],i[3][0][0],i[4][0][0]]]
#
# print(d)
# np.save("cars_train_annos.npy",d)
#
#
#
# dt = {}
# for i in matt['annotations'][0]:
#     # print(i[0],i[1],i[2],i[3],i[4], i[5][0])
#     try:
#         lt = dt[i[4][0]]
#         print("--->",lt)
#         bt = []
#         for i in lt:
#             bt.append(l)
#         bt.append([i[0][0][0],i[1][0][0],i[2][0][0],i[3][0][0]])
#         dt[i[4][0]] = bt
#     except:
#         dt[i[4][0]] = [[i[0][0][0],i[1][0][0],i[2][0][0],i[3][0][0]]]
#
# print(dt)
# np.save("cars_test_annos.npy",dt)
#
import os

print(os.walk("."))
for r, d, f in os.walk("."):
    print(f)
    np.save(cars_test)


d = np.load("cars_test_annos.npy")
print(d)
