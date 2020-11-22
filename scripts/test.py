from numpy import empty


m1 = empty((3,2))
m1[:,:] = [[22, 22], [33, 33], [44,44]]


print("\n\nM1: ")
print(m1)



m2 = empty((3,3,2))
m2[1,:,:] = m1[True,:,:]

print("M2: ")
print(m2)