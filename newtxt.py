import os

# path='./dataset/LEVIR-CD/'
# path_list=os.path.join(path,'list')
# if not os.path.exists(path_list):
#     os.makedirs(path_list)
# linenames=os.listdir(path)
# for name in linenames:
#     print(name)
#     if name != 'list':
#         txtfile_path= os.path.join(path,name)
#         txtname_path= os.path.join(txtfile_path,'A')
#         txtnames=os.listdir(txtname_path)
#         with open(f'./dataset/LEVIR-CD/list/{name}.txt','w') as f:
#             for txtname in txtnames:
#                 f.write(txtname+'\n')


# 定义文件路径
# file1 = "./dataset/LEVIR-CD/list/train.txt"
# file2 = "./dataset/LEVIR-CD/list/val.txt"
# output_file = "./dataset/LEVIR-CD/list/trainval.txt"
#
# # 打开输出文件进行写入
# with open(output_file, 'w') as outfile:
#     # 打开第一个文件并读取内容
#     with open(file1, 'r') as f1:
#         outfile.write(f1.read())
#
#     # 打开第二个文件并读取内容
#     with open(file2, 'r') as f2:
#         outfile.write(f2.read())
#
# print(f"文件已合并并保存为 {output_file}")

import torch

# 加载 checkpoint 文件
checkpoint = torch.load('./checkpoints/BIT_LEVIR/best_ckpt.pt')

# 打印 checkpoint 的所有键
print(checkpoint.keys())

# print(checkpoint['model_G_state_dict'])



