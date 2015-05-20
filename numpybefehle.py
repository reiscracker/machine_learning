#Quelle http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf
# 

import numpy as np

a = np.array([0, 1, 2, 3, 4, 5], float)

a

# type: standard python function
type(a)

a.dtype

# ndarray:
# n-dimensional arrays of homogeneous data
# Many operations performed using ndarray objects execute in compiled code for performance

len(a)

a.ndim
a.shape

2 in a 

# sclicing
a[:2]
a[2:6:3] # [start:end:step] 
a[2:6]
# step default 1  
a[::2] # default values start=0, end is the last
a[3]

a[0] = 5.
a
a[0] = 0.

# 

###############################

# multidimensional arrays 
b = a.reshape((3,2))
b.shape
b.ndim

c = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], float)

c
c[:]
c[1:]
c[:1]
c[:,2]
c[2,:]
c[1:,:2]
c[1:,1:3]

c
c[2,1]
c[2]
c[2][1]



b[1][0] = 77.
b



# Beachte !!
a
# true copy
c = a.reshape((3,2)).copy()
c[0][0] = -99
c
a

a*2

a**2

# vergleich zu normalen python lists
[0,1,2,3] * 2
a.tolist() * 2
list(a) ** 2

a.tostring()



d = a.copy()
d.fill(9.)
d
d = np.array(range(6), float).reshape((2, 3))
d.flatten()

#concatenation
np.concatenate((a, d))
a.shape
d.shape
np.concatenate((a, d.flatten()))


d = (a+1).reshape((2,3))
np.concatenate((a.reshape((2,3)), d))
np.concatenate((a.reshape((2,3)), d), axis = 0)
np.concatenate((a.reshape((2,3)), d), axis = 1)

# increasing axis
d = np.array([1, 2, 3], dtype=int)
d
d[:,np.newaxis]

# special arrays
np.zeros(7, dtype=int)
np.ones(8)
np.zeros_like(a)
np.ones_like(a)
np.identity(4, dtype=float)

#The eye function returns matrices with ones along the kth diagonal
np.eye(4, k=1, dtype=float)

##
np.arange(0, 10, 0.1)
np.linspace(0, 2 * np.pi , 100)

####

# arrays als indices
a[np.array([2,2,4])]
a>4
a[a>4]
a[np.array([True, False,  True, False, True,  False], dtype=bool)]

# Boolean arrays can be used as array selectors
a[a>4] = 4

# Methode für Truncation:
a.clip(1,2)

##################
c
# arrays als indices
c[[0,2],...]
c[[0,2],...][...,[1,3]]
# booblean arrays 
c[np.array([False, True, True, False]), ...]
c[np.array([False, True, True, False]), ...][...,np.array([False, True, True, False])]

#all columns which contain a number % 6 = 0
ind = np.apply_along_axis(lambda x: np.any(x % 6 == 0), 0, c)
c[..., ind]


#############################

I = np.eye(3)
I

x = I[0,[0,1]]
#x contains copied values
x[1] = 1
x
I
# I is still the same

x = I[0, 0:2]
# this is called a view
x[1] = 1
I
# I was modified



#################################
# array mathematics

a = np.array([1,2,3], float)
b = np.array([5,2,6], float)

# element wise operators
a + b 
a - b
a * b # not scalar product
b / a
a % b
b ** a

# 
a = np.array([[1,2], [3,4]], float)
b = np.array([[2,0], [1,3]], float)

a * b # not a matrix multiplication

# error if wrong size
a = np.array([1,2,3], float)
b = np.array([4,5], float)
a + b 

# broadcasting 
a = np.array([[1, 2], [3, 4], [5, 6]], float) 
b = np.array([-1, 3], float)

a + b
 
a = np.zeros((2,2), float)
b = np.array([-1., 3.], float) 
a + b
a + b[np.newaxis,:] 

### mathematical functions

#abs, sign, sqrt, log, log10, exp, sin, cos, tan, arcsin, arccos,
#arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh
a = np.array([1, 4, 9], float) 
np.sqrt(a)

a = np.array([1.1, 1.5, 1.9], float)

np.floor(a)
np.ceil(a)
np.rint(a)

np.pi
np.e

# iteration

a = np.array([[1, 2], [3, 4], [5, 6]], float)
for x in a:
    print x
    
a = np.array([[1, 2], [3, 4], [5, 6]], float)

for(x,y) in a:
  print x, ' * ' , y , ' = ' ,  x * y

# array operations

a = np.array([2, 4, 3], float)

a.sum()
a.prod()

a.mean()
a.var()
a.std()

a.min()
a.max()

#The argmin and argmax functions return the array indices of the minimum and maximum values: 
a.argmin()
a.argmax()

a = np.array([[0, 2], [3, -1], [3, 5]], float)
a.mean(axis=0)
a.mean(axis=1)


a.argmin()
np.unravel_index(a.argmax(), a.shape)


a = np.array([6, 2, 5, -1, 0], float)
sorted(a) 

a = np.array([1, 1, 4, 5, 5, 5, 7], float)
np.unique(a) 


a = np.array([[1, 2], [3, 4]], float)
a.diagonal() 


a = np.array([1, 3, 0], float)
b = np.array([0, 3, 2], float)
a > b 

c = np.array([ True, False, False], bool)
any(c)
all(c) 

a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3)

array([ True, False, False], dtype=bool)
b = np.array([True, False, True], bool)
np.logical_not(b)

c = np.array([False, True, False], bool)
np.logical_or(b, c)

a = np.array([1, 3, 0], float)
np.where(a != 0, 1 / a, a) 

# broadcasting for false array
np.where(a > 0, a, 1)

a = np.array([[0, 1], [3, 0]], float)
a.nonzero()

a = np.array([1, np.NaN, np.Inf, 4, 7], float)
a
np.isnan(a)
~np.isnan(a)
np.isfinite(a)
# handling of no existing values
# assume a was read in from a file 
a = np.array([5, np.NaN, 4, 7], float)
a[~np.isnan(a)]
a[~np.isnan(a)].mean()

# multidimensional array selection
a = np.array([[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
b 
c
a
a[b,c] 

#
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int)
a.take(b) # same as a[b]

#
a = np.array([[0, 1], [2, 3]], float)
b = np.array([0, 0, 1], int)
a.take(b, axis=0)  


##
a = np.array([0, 1, 2, 3, 4, 5], float)
b = np.array([9, 8, 7], float)
a.put([0, 3], b)
a 
a.put([0, 3, 1], b)
a

a = np.array([0, 1, 2, 3, 4, 5], float)
a.put([0, 3], 5)
a 

####################
# Vector and matrix multiplication
a = np.array([1, 2, 3], float)
b = np.array([0, 1, 1], float) 

np.dot(a,b)
a.dot(b)


a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
c = np.array([[1, 1], [4, 0]], float) 
np.dot(b, a)
np.dot(a, b)
np.dot(a, c)
np.dot(c, a)

a = np.array([1, 4, 0], float)
b = np.array([2, 2, 1], float)
np.outer(a, b) 
np.inner(a, b)
np.cross(a, b)

a = np.array([[4, 2, 0], [9, 3, 7], [1, 2, 1]],
float) 
np.linalg.det(a)
vals, vecs = np.linalg.eig(a)
vals
vecs
b = np.linalg.inv(a) 
b
a.dot(b)


#
list_matrix = [[1, 3, 4], [2, 3, 5], [5, 7, 9]]
A = np.array(list_matrix)
b = np.array([4, 4, 4])
# Solve for Ax = b
x = np.linalg.solve(A, b)

#############################################################


