import torch


t1=torch.tensor([1,2,3,4,5,6])

print(t1,type(t1))

t2=torch.tensor([1,2,3,4,5,6])


t3=t1+t2
print(t3,type(t3))
print("*"*50)
t4=t1*t2
t5=t1@t2
print(t4,type(t4))
print("*"*50)
print(t5,type(t5))
print("*"*50)

torch.manual_seed(24)

t6=torch.randint(1,10,(5,6))
print(t6)
print("*"*50)
# 简单行列索引
print(t1[1:])
print("取第一行所有列")



print("*"*50)
# 列表索引
print(t6[[1,2],[3,4]])
# 第一个中括号是行索引，第二个中括号是列索引
# 输出（1，3），（2，4）位置的数字5
# 高级索引
print("高级索引,取1，2行的3，4列共四个元素")
print(t6[[[1],[2]],[3,4]])#取1，2行的3，4列共四个元素
print("*"*50)

# 范围索引
# 前三行的1，2，3列
print("前三行的1，2，3列")
print(t6[:3,[1,2,3]])
print("*"*50)

# 第二行到最后一行，只要前两列
# 有点像python切片
print("第二行到最后一行，只要前两列")
print(t6[1:,:2])
print("*"*50)

print("所有奇数行，所有偶数列")
print(t6[1::2,0::2])
print("*"*50)
print("[起始索引 : 结束索引 : 步长]")
# 省略 start 默认从 0 开始,省略 end 默认到最后,step 是每隔多少取一个
print('省略 start 默认从 0 开始,\n'
      '省略 end 默认到最后,step 是每隔多少取一个')

print("布尔索引")
print(t6[torch.tensor([True,False,True,False,True]),:])
# 行mask必须要和行的长度相同，列mask必须要和列的长度相同
print("[行mask,列mask]")

print("第二行大于五的列数据")
print(t6[1,t6[1,:]>5])

print("第三列大于5的行数据")
print(t6[t6[:,2]>5,2])

# 多为索引
t7=torch.randint(1,10,(1,2,3))
print(t7[0,:])
print(t7[:,0:])
print(t7[:,:,0])












