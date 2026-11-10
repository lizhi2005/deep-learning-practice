import torch

# reshape()
torch.manual_seed(24)
# Ctrl + Alt + V这是专门的「引入变量」快捷键
t = torch.randint(1, 10, (2, 3))
print(t.shape)
print(t)
t1 = t.reshape(1, 6)
t2 = t.reshape(3, 2)#不改变内容只改变形状
print(t1)
print(t2)
print("*-"*60)
# unsqueeze(),sequeeze()升维度，降维度
# 函数有下划线就会改变原来的东西
#  带下划线 _：原位修改（inplace）
# t.unsqueeze(0) → 返回新张量，不改原 t
# t.unsqueeze_(0) → 直接改原 t，不返回新张量
t3 = t.unsqueeze(0)
# 在0维添加一个维度
t4 = t.unsqueeze(1)
# 在1维添加一个维度
print(t3.shape)
print(t4.shape)
# unsqueeze(dim)：升维，在指定位置插入长度为 1 的维度
# squeeze(dim)：降维，删掉长度为 1 的维度（不指定就全删）
print(t3)
print(t4)
print("*-"*60)
t6=torch.randint(1,20,(2,3,4))
print(t6)
print("*-"*60)
# transpose(),permute()
t5 = t6.transpose(1, 0)
print(t5)
print(t5.shape)
print("*-"*60)
t7 = t6.transpose(1, -1)
print(t7)
print(t7.shape)
print("*-"*60)
# view(),contigous(),is_contigous
# 解释：view（）
t9=torch.randint(1,20,(2,3))
t8 = t9.view(3,2)
print(t9)
print("*-"*60)
print("is_contiguous()：判断张量在内存里是不是连续存放"
"contiguous()：把不连续的张量 → 变成连续（重新拷贝一份）"
"view()：只能对连续张量使用，用来改变形状（reshape 低配版）")
