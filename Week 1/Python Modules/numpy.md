## Installing and Importing Numpy 
### For Windows/macOS/Linux
``` pip install numpy ```
### Importing NumPy
```import numpy as np```
## Arrays in NumPy
In NumPy, the core data structure is the ndarray (n-dimensional array). It is a powerful, flexible, and efficient way to store and manipulate large data sets.
```import numpy as np

arr = np.array([1, 2, 3]) # 1D array
arr2d = np.array([[1, 2], [3, 4]]) # 2D array
```
### Indexing in Arrays
```
print(arr[1]) #1D Indexing
print(arr[0][1]) #2D Indexing
```
### Output
2<br>
2
### Slicing In Arrays
```import numpy as np

a = np.array([10, 20, 30, 40, 50])

print(a[1:4])     # [20 30 40]
print(a[:3])      # [10 20 30]
print(a[::2])     # [10 30 50]
print(a[-2:])     # [40 50]
 ```
 
 
 ###   Output
 [20 30 40]<br>
[10 20 30]<br>
[10 30 50]<br>
[40 50]
```
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(b[0, :]) #first row
print(b[:, 1]) #second row
print(b[1:3, 0:2]) #submatrix
 ```
###Output<br>
[1 2 3]<br>
[2 5 8]<br>
[[4 5], [7 8]]

## Commonly Used Functions In NumPy

### `reshape()`
Used to change the shape of an array without changing its data.
```
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3)

print(b)
```
### Output
[ [1, 2, 3] , [4, 5, 6] ]

### `arange()`
Returns evenly spaced values within a given range (like Python’s range() but returns an array).
```
import numpy as np

arr = np.arange(0, 10, 2)
print(arr) 
```
### Output
[0, 2, 4, 6, 8]

### `eye()`
Creates an identity matrix (square matrix with 1s on the diagonal and 0s elsewhere).
```
import numpy as np

e = np.eye(3)
print(e)
```
### Output
 [[1, 0, 0], 
  [0, 1, 0,], 
  [0, 0, 1]]

  ### `ndim`
Gives the number of dimensions of the array
```
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x.ndim)

```
### Output
 2

  ### `size`
Returns the total number of elements in the array.
```
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x.size)
```
### Output
 4

   ### `dtype`

Shows the data type of the array elements.
```
import numpy as np

x = np.array([1, 2, 3])
print(x.dtype)
```
### Output
 int64

   ### `itemsize`

Returns the size (in bytes) of one array element.
```
import numpy as np

x = np.array([1, 2, 3])
print(x.itemsize)

```
### Output
 8
### `flatten()` 

Converts a multi-dimensional array into a one-dimensional array.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.flatten()
print(b)
```
### Output
[1, 2, 3, 4]

## NumPy Vectorised Functions
Vectorized functions (also known as universal functions or ufuncs) operate element-wise on arrays without the need for explicit loops. They are highly optimized and make numerical computations faster and cleaner.

###    ` Vectorised Arthematic Operations`
Operations in NumPy are taken care of element-wise
``` 

import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(a+b)
print(a-b)
print(a/b)
print(a*b)

```
### Output
[11 22 33] &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;     # a + b: [1+10, 2+20, 3+30] <br>
[-9 -18 -27]  &nbsp;&nbsp;&nbsp;   # a - b: [1-10, 2-20, 3-30]<br>
[0.1 0.1 0.1]   &nbsp;&nbsp;&nbsp; # a / b: [1/10, 2/20, 3/30]<br>
[10 40 90]      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# a * b: [1*10, 2*20, 3*30]

### `Universal Functions`
``` 
import numpy as np

angles = np.array([0, np.pi/2, np.pi])
sines = np.sin(angles)
print(sines)
```

### Output
[0.0000000e+00, 1.0000000e+00, 1.2246468e-16]

### Common Vectorized (ufunc) Functions in NumPy:
- np.add(x, y) – Element-wise addition

- np.subtract(x, y) – Element-wise subtraction

- np.multiply(x, y) – Element-wise multiplication

- np.divide(x, y) – Element-wise division

- np.sqrt(x) – Square root

- np.exp(x) – Exponential (e^x)

- np.log(x) – Natural logarithm

- np.sin(x) – Sine

- np.cos(x) – Cosine

- np.abs(x) – Absolute value


## Statistical Functions in NumPy
## Mean (np.mean)
Calculates the average
```
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(np.mean(a))
 ```
 ### Output
 3.0

## Median (np.median) 
Calculates the middle value
```
import numpy as np


a = np.array([10, 20, 30, 40, 50])
print(np.median(a))

 ```
 ### Output
 30.0

 ### More Statistical Functions
 - np.std() – Standard deviation

- np.var() – Variance

- np.max() – Maximum value

- np.min() – Minimum value

- np.sum() – Sum of all elements

- np.prod() – Product of all elements

- np.percentile() – Percentile value

- np.cumsum() – Cumulative sum

- np.cumprod() – Cumulative product

- np.argmax() – Index of max value

- np.argmin() – Index of min value

- np.quantile() – Quantile value

## Logical Operations
NumPy supports element-wise logical operations using standard Python operators:

- == → Equal to

- != → Not equal to

- \> → Greater than

- < → Less than

- \>= → Greater than or equal to

- <= → Less than or equal to

- & → Logical AND (use with () around comparisons)

- | → Logical OR

- ~ → Logical NOT

```
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 3, 5])

print((a == b) | (a < b))
 ```

 ### Output
 [ True,  True,  True,  True]

## What is Broadcasting in NumPy?
Broadcasting is a powerful feature in NumPy that allows array operations between arrays of different shapes without explicitly replicating data.

It automatically "stretches" the smaller array across the larger one so that they have compatible shapes.

## Examples
### 1. Add Scalar to Array
```
import numpy as np

a = np.array([1, 2, 3])
b = 5

result = a + b
print(result) 
```
### Output: 
```
[6 7 8]
```
Here, b (a scalar) is broadcast to the shape of a.
### 2. Add 1D to 2D Array
```
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([10, 20, 30])

result = A + B
print(result)
```
### Output:

```
[[11 22 33]
 [14 25 36]]
 ```
Here, B is broadcast across the rows of A.

### 3. Column-wise Broadcasting
```
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[10],
              [20]])
`

result = A + B
print(result)
```
### Output:
```
[[11 12 13]
 [24 25 26]]
 ```
Here, B is broadcast across the columns of A
### Broadcasting Error Example
```
a = np.array([1, 2, 3])
b = np.array([[1], [2]])

a + b  # This will raise an error due to incompatible shapes
```

## Additional Resources
[NumPy Tutorial](https://www.geeksforgeeks.org/python-numpy/)
<br>
[Video Resources](https://youtu.be/9JUAPgtkKpI?si=9q9uT_IEfcR7SZ2U)