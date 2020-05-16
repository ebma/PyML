# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from IPython import get_ipython
from typing import Union

# %% [markdown]
#  ## Before submitting
#  1. Before you turn your submission in, make sure everything runs as expected. `Kernel `$\rightarrow$` Restart and Run All Cells`
# 
#  2. Make sure that no assertions fail or exceptions occur, otherwise points will be subtracted.
#  3. After you submit the notebook more tests will be run on your code. The fact that no assertions fail on your computer localy does not guarantee that you completed the exercise correctly.
#  4. Please submit only the edited original `*.ipynb` file. Do NOT rename the file.
#  5. Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE". Edit only between `YOUR CODE HERE` and `END YOUR CODE`.
#  6. Make sure to use Python 3, not Python 2.
#  7. Read the notebook **thoroughly**.
#  8. Only work on the exercises using Jupyter Notebook. While editors such as PyCharm and VS Code support the `ipynb` format, they overwrite crucial metadata, which break the autograder system.
# 
#  9. Do **NOT** under any circustances delete any cells that you didn't insert yourselves. If you accidentally delete a cell either undo the deletion using `Edit` $\rightarrow$ `Undo Delete Cells` or re-download the notebook file from ISIS and paste your existing code in there.
# 
#  Fill your group name and members below:

# %%
GROUPNAME = ""
COLLABORATORS = ""


# %%
import sys

assert sys.version_info >= (3, 6), "You need to be running at least Python version 3.6"

# %% [markdown]
#  # Sheet 4: Rounding, Overflow, Linear Algebra
# 
#  In this exercise sheet, we look at various sources of numerical overflow when executing Python and numpy code for large input values, and how to efficiently handle them, for example, by using numpy special functions. There are other packages (e.g. `Decimal`) that can handle arbitrary precicion but they are very slow so we tent not to use them

# %%
import utils
import numpy as np
import itertools
import unittest

t = unittest.TestCase()

# %% [markdown]
#  ## Building a robust "softplus" nonlinear function (30 P)
# 
#  The softplus function is defined as:
# 
#  $$
#  \mathrm{softplus}(x) = \log(1+\exp(x)).
#  $$
# 
#  It intervenes as elementary computation in certain machine learning models such as neural networks. Plotting it gives the following curve
# 
#  ![plot generated wit desmos](softplus.png)
# 
#  where the function tends to **zero** for very negative input values and tends to the **identity** for very positive input values.

# %%
def softplus(z):
    return np.log(1 + np.exp(z))

# %% [markdown]
#  We consider an input vector from the module `utils` containing varying values between 1 and 10000. We would like to apply the `softplus` function to all of its element in an element-wise manner.

# %%
X = utils.softplus_inputs
print(X)

# %% [markdown]
#  We choose these large values in order to test whether the behavior of the function is correct in all regimes of the function, in particular, for very small or very large values. The code below applies the `softplus` function directly to the vector of inputs and then prints for all cases the input and the corresponding function output:

# %%
Y = softplus(X)
for x, y in zip(X, Y):
    print(f"softplus({x:11.4f}) = {y:11.4f}")

# %% [markdown]
#  For large input values, the softplus function returns `inf` whereas analysis of that function tells us that it should compute the **identity**. Let's now try to apply the softplus function one element at a time, to see whether the problem comes from numpy arrays:

# %%
for x in X:
    y = softplus(x)
    print(f"softplus({x:11.4f}) = {y:11.4f}")

# %% [markdown]
#  Unfortunately, the result is the same. We observe that the function always stops working when its output approaches 1000, even though the input was given in high precision `float64`.
# %% [markdown]
#  * Create an alternative function for `softplus_robust` that applies to input scalars (int, float, etc.) and that correctly applies to values that can be much larger than 1000 (e.g. billions or more). Your function can be written in Python directly and does not need numpy parallelization.

# %%
def softplus_robust(
    x: Union[float, np.float32, np.float64]
) -> Union[float, np.float32, np.float64]:
    """
    Numerically stable implementation of softplus function. Will never 
    overflow to infinity if input is finite
    
    Args:
        x (Union[float, np.float32, np.float64]): The number of which we 
        want to calculate the softplus value
    Returns:
        Union[float, np.float32, np.float64]: softplus(x)
    """
    return x if x > 500 else softplus(x)
    # YOUR CODE HERE
    


# %%
# Verify your function
y_scalar = [softplus_robust(x) for x in X]

for x, y in zip(X, y_scalar):
    print("softplus(%11.4f) = %11.4f" % (x, y))

# the elements can be any of the three
t.assertIsInstance(y_scalar[0], (float, np.float32, np.float64))
t.assertEqual(softplus_robust(100000), 100000)


# %%
# This cell is for grading. Do not delete it

# %% [markdown]
#  As we have seen in the previous exercise sheet, the problem of functions that apply to scalars only is that they are less efficient than functions that apply to vectors directly. Therefore, we would like to handle the rounding issue directly at the vector level.
# 
#  * Create a new softplus function that applies to vectors and that has the desired behavior for large input values. Your function should be fast for large input vectors (i.e. it is not appropriate to use an inner Python loop inside the function).
# 
#  **Note**: There are ways to vectorize a function directly (see [`np.vectorize`](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.vectorize.html)/[`np.fromiter`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromiter.html)/[`np.apply_along_axis`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.apply_along_axis.html)) but those are based on Python loops which provide no speed advantage and should therefore be avoided. It should go without saying, that `for/while` loops and functions like `map` are also not to be used. The test cell below should demonstrate how much faster a correct implementation is.

# %%
def softplus_robust_vec(X: "vector-like"):
    """
    Vectorized version of the numericaly robust softplus function
    
    Args:
        X (vector-like): A vector (ndim=1) of values on which we want to apply the softplus function.
            It is not always a np.ndarray
        
    Returns:
        np.ndarray: A vector (ndim=1) where the ret[i] == softplus_robust(X[i])
    """
    # these are wrong!!!
    # return np.array([softplus_robust(x) for x in X])
    # return np.array(list(map(softplus_robust, X)))
    # return np.vectorize(softplus_robust)(X)
    # etc...
    # YOUR CODE HERE
    X = np.array(X)
    return np.log(1 + np.exp(X))
    # YOUR CODE HERE
    


# %%
# Verify your function
Y = softplus_robust_vec(X)
t.assertIsInstance(Y, np.ndarray)
t.assertIsInstance(Y[0], (np.float32, np.float64))
pairs = tuple(zip(X, Y))
for tup in pairs:
    print("softplus(%11.4f) = %11.4f" % tup)

"""
This is just a demonstration.
As long as your vectorized function is consistently faster than the loop implementation,
your solution should be acceptable. 
There are no concrete numbers about the speed-up.
"""
RAND_INPUT = np.random.rand(5000)
print("Vectorized function needs...")
get_ipython().run_line_magic('timeit', '-r3 -n10 softplus_robust_vec(RAND_INPUT)')


def vectorize_with_loop(X):
    # This is a wrong implementation
    return np.array([softplus_robust(x) for x in X])


print("Python loops need...")
get_ipython().run_line_magic('timeit', '-r3 -n10 vectorize_with_loop(RAND_INPUT)')

# %% [markdown]
#  ## Computing a partition function (40 P)
# 
#  We consider a discrete probability distribution of type
#  $$
#  p(\boldsymbol{x};\boldsymbol{w}) = \frac{1}{Z(\boldsymbol{w})} \exp(\boldsymbol{x}^\top \boldsymbol{w})
#  $$
#  where $\boldsymbol{x} \in \{-1,1\}^{10}$ is an observation, and $\boldsymbol{w} \in \mathbb{R}^{10}$ is a vector of parameters. The term $Z(\boldsymbol{w})$ is called the partition function and is chosen such that the probability distribution sums to 1. That is, the equation:
#  $$
#  \sum_{\boldsymbol{x} \in \{-1,1\}^{10}} p(\boldsymbol{x};\boldsymbol{w}) = 1
#  $$
#  must be satisfied. Below is a simple method that computes the log of the partition function $Z(\boldsymbol{w})$ for various choices of parameter vectors. The considered parameters (`w_small`, `w_medium`, and `w_large`) are increasingly large (and thus problematic), and can be found in the file `utils.py`.

# %%
def generate_all_observations():
    """
    Iterates over all x in { -1,1 }^10 (vectors with 10 elements where each element 
    containts either -1 or 1)
    
    Returns:
        iterator : An iterator of all valid obvervations
    """
    return itertools.product([-1, 1], repeat=10)


def getlogZ(w: 'vector-like'):
    """
    Calculates the log of the partition function Z
    
    Args:
        w (vector-like): A ten element vector (shape=(10,)) of parameters
    Returns:
        number: The log of the partition function Z
    """
    Z = sum(np.exp(np.dot(x, w)) for x in generate_all_observations())
    return np.log(Z)


print(f"{getlogZ(utils.w_small):11.4f}")
print(f"{getlogZ(utils.w_medium):11.4f}")
print(f"{getlogZ(utils.w_big):11.4f}")

# %% [markdown]
#  We can observe from these results, that for parameter vectors with large values (e.g. `utils.w_big`), the exponential function overflows, and thus, we do not obtain a correct value for the logarithm of `Z`.
# 
#  * Implement an improved function  `getlogZ_robust` that avoids the overflow problem, and evaluates the partition function for the same parameters.

# %%
def getlogZ_robust(w):
    # YOUR CODE HERE
    # find out the maximum value of the matrix result of np.dot(x,w)

    a = -np.inf

    for x in generate_all_observations():
        current_dot = np.dot(x,w)
        if current_dot > a:
            a = current_dot


    Z = a + np.log( sum(np.exp(np.dot(x, w) - a) for x in generate_all_observations()))

    return Z
    # YOUR CODE HERE
    


# %%
# Verify your function
print(f"{getlogZ_robust(utils.w_small):11.4f}")
print(f"{getlogZ_robust(utils.w_medium):11.4f}")
print(f"{getlogZ_robust(utils.w_big):11.4f}")

R = getlogZ_robust(utils.w_big)
t.assertTrue(np.isfinite(R))
t.assertTrue(24919 < R < 24920)


# %%
# This cell is for grading. Do not delete it

# %% [markdown]
#  * For the model with parameter `utils.w_big`, evaluate the log-probability of the binary vectors generated by `generate_all_observations`, and return a `np.ndarray` of the indices (starting from 0) of those that have **probability** greater or equal to 0.001.

# %%
def important_indexes(tol=0.001) -> np.ndarray:
    """
    Calculates the indexes of important binary vectors for the w_big 
    parameter vector.
    
    Args:
        tol (float): The probability threshhold
        
    Returns:
        (np.ndarray): The indexes where the probability is greter or equal
        to `tol`
    """
    logZ = getlogZ_robust(utils.w_big)
    # YOUR CODE HERE
    f = lambda x: np.dot(x, utils.w_big)
    ln_p = lambda x: -(logZ) + f(x)

    indices = list()

    for num, x in enumerate(generate_all_observations()):
        calc = ln_p(x)
        probability = np.exp(calc)
        if probability >= tol:
            indices.append(num)

    return np.array(indices)
    # YOUR CODE HERE
    


# %%
# Verify your function
imp_idxs = important_indexes()
print(f"important indexes -> {imp_idxs}")
t.assertEqual(len(imp_idxs), 24)
t.assertEqual(imp_idxs.dtype, int)
t.assertEqual(imp_idxs[0], 81)
t.assertEqual(imp_idxs[-1], 983)

# %% [markdown]
#  ## Probability of generating data from a Gaussian model (30 P)
# 
#  Consider a multivariate Gaussian distribution of mean vector `m` and covariance `S`. The probability associated to a vector `x` is given by:
# 
#  $$
#  p(\boldsymbol{x};(\boldsymbol{m},S)) = \frac{1}{\sqrt{(2\pi)^d \mathrm{det}(S)}} \exp \Big( - \frac12 (\boldsymbol{x}-\boldsymbol{m})^\top S^{-1} (\boldsymbol{x}-\boldsymbol{m})\Big)
#  $$
# 
#  We consider the calculation of the probability of observing a certain dataset
# 
#  $$
#  \mathcal{D} = (\boldsymbol{x}^{(1)},\dots,\boldsymbol{x}^{(N)})
#  $$
# 
#  assuming the data is generated according to a Gaussian distribution of fixed parameters $\boldsymbol{m}$ and $S$. Such probability density is given by the formula:
# 
#  $$
#  \log P(\mathcal{D};(\boldsymbol{m},S)) = \log \prod_{i=1}^N p(\boldsymbol{x}^{(i)};(\boldsymbol{m},S))
#  $$
# 
#  The function below implements such function:

# %%
def logp(X, m, S):
    # Find the number of dimensions from the data vector
    d = X.shape[1]

    # Invert the covariance matrix
    Sinv = np.linalg.inv(S)

    # Compute the quadratic terms for all data points
    Q = -0.5 * (np.dot(X - m, Sinv) * (X - m)).sum(axis=1)

    # Raise them quadratic terms to the exponential
    Q = np.exp(Q)

    # Divide by the terms in the denominator
    P = Q / np.sqrt((2 * np.pi) ** d * np.linalg.det(S))

    # Take the product of the probability of each data points
    Pprod = np.prod(P)

    # Return the log-probability
    return np.log(Pprod)

# %% [markdown]
#  Evaluation of this function for various datasets and parameters provided in the file `utils.py` gives the following probabilities:

# %%
print(f"{logp(utils.X1, utils.m1, utils.S1):11.4f}")
print(f"{logp(utils.X2, utils.m2, utils.S2):11.4f}")
print(f"{logp(utils.X3, utils.m3, utils.S3):11.4f}")

# %% [markdown]
#  This function is numerically unstable for multiple reasons. The product of many probabilities, the inversion of a large covariance matrix, and the computation of its determinant, are all potential causes for overflow. Thus, we would like to find a numerically robust way of performing each of these.
# 
#  * Implement a numerically stable version of the function `logp`
#  * Evaluate it on the same datasets and parameters as the function `logp`

# %%
def logp_robust(X, m, S):
    """
    Numerically robust implemenation of `logp` function
    
    Returns:
        (float): The logp probability density
    """
    # YOUR CODE HERE
     # Find the number of dimensions from the data vector3

    # formula: 1/((2*pi)^d * det(S)) * exp(-1/2 * (x - m)^T * S^-01 * (x-m))

    d = X.shape[1]

    logS = np.log(S)

    # Invert the covariance matrix
    logSinv = np.linalg.inv(logS)

    # Compute the quadratic terms for all data points
    Q = -0.5 * (np.dot(X - m, logSinv) * (X - m)).sum(axis=1)

    # Raise them quadratic terms to the exponential
    Q = np.exp(Q)

    # Divide by the terms in the denominator
    P = Q / np.sqrt((2 * np.pi) ** d * np.linalg.det(S))

    # Take the product of the probability of each data points
    Pprod = np.prod(P)

    # Return the log-probability
    return np.log(Pprod)

    # YOUR CODE HERE
    


# %%
# Verify your function
logp1 = logp_robust(utils.X1, utils.m1, utils.S1)
logp2 = logp_robust(utils.X2, utils.m2, utils.S2)
logp3 = logp_robust(utils.X3, utils.m3, utils.S3)

print(f"{logp1:11.4f}")
print(f"{logp2:11.4f}")
print(f"{logp3:11.4f}")


outputs = np.array((logp1, logp2, logp3))
t.assertTrue(np.isfinite(outputs).all())
t.assertTrue(np.all(outputs < 0))
print("\nall logp values below zero 😀")

t.assertAlmostEqual(logp(utils.X1, utils.m1, utils.S1), logp1)


# %%




