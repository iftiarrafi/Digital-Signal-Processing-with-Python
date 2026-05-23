import numpy as nm
a=nm.array([1,2,3,4])
print(a) #[1 2 3 4]
print(a.ndim) #1
b = nm.array([[1,2] , [4,5] , [4,7]])
c = nm.array([[[1,2] , [3,4]] , [[6,7] , [8,9]]])
#print(c.shape)
# print(b) 
# print(b.ndim)
# print(b.shape)
# print(b.dtype)

arr = nm.array([[1,2,3] , [6,7,8]] , dtype='int16')
print(arr.dtype)

print(arr[1,2])
print(arr[1,:])
print(arr[:,2])