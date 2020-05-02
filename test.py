import numpy as np
# with open('data.txt') as f:
#     datas = np.array(f.read().splitlines())
#     datas = np.reshape(datas, (int(datas.shape[0] / 96), 96))
#     for data in datas:
#         for i in data:
#             print(i, end = ' ')
#         print()

with open('data1.txt') as f:
    print(f.read().splitlines().split())
    