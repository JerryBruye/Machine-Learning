import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

######### Load the data ##########

infile = open('faces.csv','r')
img_data = infile.read().strip().split('\n')
#print(img_data)
img = [map(int,a.strip().split(',')) for a in img_data]
#print(img)
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels,(400,4096))

######### Global Variable ##########

image_count = 0

######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.
#   Note: in the given functionm, U should be a vector, not a array. 
#         You can write your own normalize function for normalizing 
#         the colomns of an array.

def normalize(U):
	return U / LA.norm(U) 

######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.

first_face = np.reshape(faces[0],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('First_face')
plt.imshow(first_face,cmap=plt.cm.gray)


########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.
#   Note: There are two ways to order the elements in an array: 
#         column-major order and row-major order. In np.reshape(), 
#         you can switch the order by order='C' for row-major(default), 
#         or by order='F' for column-major. 


#### Your Code Here ####
random = np.random.choice(len(faces))
random_face = np.reshape(faces[random],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('Random_face')
plt.imshow(random_face,cmap=plt.cm.gray)



########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   float64 intermediate and return values are used for integer inputs.

#### Your Code Here ####
mean = np.mean(faces, axis = 0)
mean_face = np.reshape(mean,(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('Mean_face')
plt.imshow(mean_face,cmap=plt.cm.gray)  


np.save("mean.npy",mean)


mean = np.load("mean.npy")      




######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####

#faces_vector = np.reshape(faces,(400*4096,1),order='F')
A =  faces - np.reshape(np.repeat(mean,400),(400,4096),order='F')
#faces = np.reshape(faces,(400,4096),order='F')

random = np.random.choice(len(faces))
random_face = np.reshape(faces[random],(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('Centralized_face')
plt.imshow(random_face,cmap=plt.cm.gray)  

np.save("face.npy",faces)


faces = np.load("face.npy")


######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####

covariance = np.dot(np.matrix(A), np.matrix(A).T)

values = np.linalg.eig(np.array(covariance))[0]

vectors = np.linalg.eig(np.array(covariance))[1]
z = np.matrix(A).T * vectors
z1 = np.array(z)
z2 = []

for i in range(len(z1[0])):
    array = np.array([])
    #print(len(z[0]))
    for ii in range(len(z1)):
        array = np.append(array, z1[ii][i])
    normalized = normalize(array)
    #normalized = array
    for ii in range(len(z1)):
        z1[ii][i] = normalized[ii]
    z2.append(normalized)
z = np.matrix(z1)
#np.linalg.norm(z)

d = {}
for i in range(len(values)):
    d[values[i]] = z2[i]

#print(eigenvalues)
#print()
#print(eigenvectors)

np.save("values.npy",values)
#np.save("d.npy",d)
np.save("vectors.npy",vectors)
np.save("covariance.npy",covariance)


covariance = np.load("covariance.npy")
values = np.load("values.npy")
vectors = np.load("vectors.npy")
#d = np.load("d.npy")


########## Display the first 16 principal components ##################
import copy
s_values = copy.deepcopy(values)

values.sort()


result = []
#print(values[len(values) - 16:len(values)])
for i in values[0:len(values)]:
    result.append(i)
result.reverse()



#print(result)
np.save("a.npy",result)

result = np.load("a.npy")
#print(d[result[398]], result[398])


########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####
a_face = faces[0]

l = np.array([])
n = 2
for i in range(n):
    #print(np.array(result[i].T)[0])
    #print(np.array(result[i].T)[0])
    index = result[i]
    

    l = np.append(l, d[index])

#print("biggest", biggest)
U = np.matrix(l)
U = np.reshape(U, (n,4096))
tu = U.T
#print(U)
#print(len(U))
temp2 = a_face - mean
temp2 = np.reshape(temp2 , (4096,1))

omega = np.dot(U, temp2)
#mean = np.reshape(mean,(4096,1))
x = np.dot(tu,omega)
new_l = np.matrix(mean) + x.T

#new_l = np.array(new_l.T)[0]

#U = np.reshape(,(64,64),order='F')
random_face = np.reshape(new_l,(64,64),order='F')

#print(random_face)
image_count+=1
plt.figure(image_count)
plt.title('two_vectors_face')
plt.imshow(random_face,cmap=plt.cm.gray) 




########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

a_face = faces[99]


values = s_values
for n in [5, 10, 25, 50, 100, 200, 300, 399]:
    l = np.array([])
    for i in range(n):
    #print(np.array(result[i].T)[0])
    #print(np.array(result[i].T)[0])
        index = values[i]
    

        l = np.append(l, d[index])

    U = np.matrix(l)

    U = np.reshape(U, (n,4096))
    tu = U.T
    #print(U)
    #print(len(U))
    temp2 = a_face - mean
    temp2 = np.reshape(temp2 , (4096,1))
    
    omega = np.dot(U, temp2)
    #mean = np.reshape(mean,(4096,1))
    x = np.dot(tu,omega)
    new_l = np.matrix(mean) + x.T

#new_l = np.array(new_l.T)[0]

#U = np.reshape(,(64,64),order='F')
    random_face = np.reshape(new_l,(64,64),order='F')
    
#print(random_face)
    image_count+=1
    plt.figure(image_count)
    plt.title('multiple_vectors_face with PC' + str(n))
    plt.imshow(random_face,cmap=plt.cm.gray) 
    


######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####
sum_values = sum(values)
l1 = []
for i in values:
    l1.append(i/sum_values)
image_count+=1
plt.figure(image_count)
plt.plot(l1)
plt.show()




