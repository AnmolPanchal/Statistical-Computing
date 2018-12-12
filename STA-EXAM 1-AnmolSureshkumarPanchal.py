
# coding: utf-8

# # Question 1 - Bootstrap, jackKnife, CI

# In[201]:


import pandas as pd
import numpy as np
import math
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.stats


# In[202]:


data = pd.read_csv(r"C:\Users\anmol\Downloads\mtcars.csv")
## r before your normal string helps it to convert normal string to raw string 


# In[207]:


# Summary statistics for the dataframe

data.describe()


# In[209]:



"""
Covariance
Covariance is one of the fundamental technique to understand the relation between two variable. A positive covariance number between two variables means that they are positively related, while a negative covariance number means the variables are inversely related. The key drawback of covariance is that it does explain us the degree of positive or negative relation between variables
"""
data.cov()


# In[210]:


data.corr()

"""

Correlation
Correlation is another most commonly used technique to determine the relationship between two variables. Correlation will explain wheather variables are positively or inversely related, also number tells us the degree to which the variables tend to move together.

When we say that two items are correlated means that the change in one item effects a change in another item. Correlation is always a range between -1 and 1. For example, If two items have a correlation of .6 (60%) means that change in one item results in positive 60% change to another item.

"""
data.corr()


# In[211]:


print ("DataFrame Index: ", data.index)


# In[212]:


print(data.values)


# In[213]:


# Sort your dataframe
data.sort_values(by =['mpg','Cars'], ascending=[True,True])


# In[214]:


# Resampling from our dataset
from sklearn.utils import resample
boot = resample(data.iloc[:,1:2], replace=False, n_samples=32, random_state=1)


# In[215]:


boot


# In[255]:


import math
import numpy
import numpy.random


def __array_mean_indices(a, indices, func_axis=None, dtype=None):

    if func_axis == None:
        return (numpy.mean(a.flat[indices], dtype=dtype), )
    else:
        return tuple(numpy.mean(numpy.reshape(numpy.take(a, [j,], axis=func_axis), -1)[indices]) for j in range(a.shape[func_axis]))

def __number_measurements(a, func_axis=None):
    """ Calculates the number of measurements of an array from the array and the function axis.
    """
    if func_axis == None:
        return a.size
    else:
        return a.size / a.shape[func_axis]

def identity(x):
    """
    Identity function used as default function in the resampling methods.

    """
    return x

def bootstrap(a, iterations, func=identity, func_axis=None, dtype=None):
   
    # Calculate the number of measurements
    n = __number_measurements(a, func_axis)
    # Evaluate the function on the bootstrap means
    bootstrap_values = [func(*(__array_mean_indices(a, numpy.random.randint(0, high=n, size=n), func_axis=func_axis, dtype=dtype))) for i in range(iterations)]

    # Return the average value and the error of this averaged value
    return numpy.mean(bootstrap_values), math.sqrt(float(iterations)/float(iterations - 1))*numpy.std(bootstrap_values)
    print (numpy.std(bootstrap_values))



# In[256]:


__array_mean_indices(boot.values,[0,31], func_axis=None, dtype=None)


# In[257]:


__number_measurements(boot.values, func_axis=None)


# In[258]:


identity(x)


# In[259]:


bootstrap(boot.values, 100, func=identity, func_axis=None, dtype=None) 


# In[266]:


z = np.mean(boot.values)
v = np.std(boot.values)
print("The sample mean and std deviation is:->",z,v)


# In[289]:


CV = np.sqrt(np.var(boot))/np.mean(boot)
print(CV)
#Another way to obtain coeffiecient of variation is shown below:
b_cov = scipy.stats.variation(boot)
print(b_cov)


# In[264]:


a= np.mean(boot)
N=32
bias =(a - CV)/N
print(bias)


# In[61]:


n=32
se =  np.std(boot) / n
print("Std error of this sample is:", se)


# In[287]:


mean_a, error_a = bootstrap(boot.values, 100)
print(mean_a,error_a)
#error_a is se_hat and se is se_that


# In[281]:


(mean_a > 34, mean_a < 10)


# In[282]:


(error_a > 2.0/math.sqrt(1000 - 1) - 0.01, error_a < 2.0/math.sqrt(1000 - 1) + 0.01)


# In[346]:


# from scipy.special import erfinv
# import numpy as np
# from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats


# In[347]:


test_statistic = np.mean


# In[348]:


test_statistic


# In[349]:


d = boot.values


# In[351]:


import numpy as np
from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats

resamples = jackknife_resampling(d)
resamples


# In[352]:


x = scipy.stats.variation


# In[353]:


def jackknife_resampling(data):
    n = data.shape[0]
    assert n > 0, "data must contain at least one measurement"

    resamples = np.empty([n, n-1])

    for i in range(n):
        resamples[i] = np.delete(data, i)

    return resamples

def jackknife_stats(data, statistic, conf_lvl=0.95):
    stat_data = statistic(data)
    jack_stat = np.apply_along_axis(statistic, 1, resamples)
    mean_jack_stat = np.mean(jack_stat, axis=0)
    # jackknife bias
    bias = (n-1)*(mean_jack_stat - stat_data)

    # jackknife standard error
    std_err = np.sqrt((n-1)*np.mean((jack_stat - mean_jack_stat)*(jack_stat -
                                    mean_jack_stat), axis=0))

    # bias-corrected "jackknifed estimate"
    estimate = stat_data - bias
    # jackknife confidence interval
    assert (conf_lvl > 0 and conf_lvl < 1), "confidence level must be in (0,1)."
    z_score = np.sqrt(2.0)*erfinv(conf_lvl)
    conf_interval = estimate + z_score*np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


# In[354]:


jackknife_stats(resamples,np.std, conf_lvl=0.95)


# In[355]:


jackknife_stats(resamples,np.std, conf_lvl=0.95)


# In[356]:


jackknife_stats(d,x, conf_lvl=0.95)


# In[360]:


plt.hist(d, 25, histtype='step');


# In[361]:


def mean_confidence_interval(sample, confidence=0.95):
    a = 1.0 * np.array(sample)
    n = len(d)
    m, se = np.mean(d), scipy.stats.sem(d)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


# In[362]:


mean_confidence_interval(resamples, confidence=0.95)


# In[466]:


np.percentile(resamples, 0.95)



# In[467]:


scipy.stats.mstats.mquantiles (resamples,0.95)


# In[468]:


scipy.stats.mstats.mquantiles (resamples,0.05)


# # Question 2 - LSSVD

# In[366]:


import pandas as pd
import numpy as np


# In[477]:


data = pd.read_csv(r"C:\Users\anmol\Downloads\charlie1.csv")
X = data[['z1','z2']]
y = data['Data']

y_out = np.array(y[20:])
x_out = np.array(X[20:])

y = y[0:20]
X = X[0:20]

X = np.array(X)
y = np.array(y)


# In[478]:


def Kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2 / ( (sigma ** 2)))

def Gram_Matrix(x):
    K = np.zeros((len(x),len(x)))
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            K[i, j] = Kernel(x[i], x[j], sigma)
            
    return K

def H(x):
    mat = np.zeros((len(x), len(x)))
    mat[0:len(x), 0:len(x)] = Gram_Matrix(x) + np.eye(len(x))/2*C
    return mat

def alpha():
#     a = 0.5*np.dot(np.linalg.inv(H_mat),(k + np.dot((2-np.dot(np.dot(e.T, np.linalg.inv(H_mat)), k))/(np.dot(np.dot(e.T, np.linalg.inv(H_mat)), e)),e)))
    p1 = np.dot(np.dot(np.linalg.inv(H_mat), e.T),k)
    p2 = np.dot(np.dot(np.linalg.inv(H_mat), e.T), e)
    p3 = (2-p1)/p2
    p3 = k + np.dot(p3, e)
    a = 0.5*np.dot(np.linalg.inv(H_mat),p3)
    return a


# In[513]:


e = np.ones(len(X))
k = np.zeros((len(X)))

sigma = 0.125
C = 1


# In[514]:


for j in range(0, len(X)):
    k[j] = Kernel(X[j], X[j], sigma)
    


# In[515]:


H_mat = H(X)
al = alpha()


# In[516]:


def R_square():
    p1 = 0
    p2 = 0
    total = 0
    for s in range(0, len(X)):
        k = Kernel(X[s], X[s], sigma)
        for j in range(0, len(X)):
            p1 = p1 + al[j]*Kernel(X[s], X[j], sigma)
            for l in range(0, len(X)):
                p2 = p2 + al[j]*al[l]*Kernel(X[j], X[l], sigma)
        total = total + (k - 2 * p1 + p2)
    final = total/len(X)
    return final

final = R_square()


# In[517]:


final


# In[518]:


def classification(x):
    t_out = []
    t_in = []
    p = 0
    p1 = 0
    for z in range(0, len(x)):
        k = Kernel(x[z], x[z], sigma)    
        for j in range(0, len(X)):
            p = p + al[j]*Kernel(x, X[j], sigma)
            for l in range(0, len(X)):
                p1 = p1 + al[j]*al[l]*Kernel(X[j], X[l], sigma)
        d = k - 2*p + p1
        if d <= final:
            t_in.append(x[z])
        else:
            t_out.append(x[z])

    return t_out, t_in

t_out, t_in = classification(x_out)


# In[505]:


t_out


# In[506]:


t_in


# In[507]:


import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

clf = svm.OneClassSVM(kernel = 'rbf', gamma = 'auto')
clf.fit(t_out, t_in)


# In[526]:


clf.predict(t_out)


# In[524]:


n_error_outliers = t_out[t_out == 1].size
print("Number of errors = ",n_error_outliers,"/",y_out.size)
#classification rate
rate = n_error_outliers/y_out.size
print("Classification rate = ",100*(1-rate),"%")


# In[525]:


df = pd.DataFrame(t_out)


# In[511]:


import seaborn as sns
sns.pairplot(df)


# In[512]:


l = df.iloc[0:,1:2]
x = np.linspace(0, 10, 10)
y = l
plt.plot(t_out, y_out, 'o', color='black');

print("This shows that all t_out i.e outliers  and y_out = New points are detected as anomaly and shown below at -1,0 ")
print("Rest all points are not shown as they appear to be inside the circle of radius = final =0.47 and are not counted as anomaly i.e why we have t_in as empty set for any -1 value.")


# # Question 3 - Acceptance Rejection Sampling

# In[493]:


import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[494]:


i = 0
k = 0
n = 1000
z = np.random.uniform(0,1,n)
while i<n:
    u = np.random.uniform(0,1,1)
    y = np.random.exponential(scale=0.001,size = 1)
    k = k+1
    if u >= np.sqrt(2/math.pi)*np.exp(-y*2/2):
        i = i
    else:
        z[i] = y*(u < np.sqrt(2/math.pi)*np.exp(-y*2/2))
        i += 1
        
print(i, k)


# In[495]:


# P= P(Y accepted) =1/c
P=i/k
c = 1/P
print("Bounding Constant is c:", c)


# In[496]:


sns.distplot(z, hist = True, kde = True)
plt.show()


# In[497]:


"""
Answers:
a) Calculate the optimal constant C for acceptance rejection as a function of  λ.
"""
print("The expected number of iterations of the algorithm required until an X is successfully generated is exactly the bounding constant C. In particular, we assume that the ratio f(x)/g(x) is bounded by a constant c > 0. And in practice we would want c as close to 1 as possible.")
print("C =", c)

"""
b) What is the best parameterλ∈(0,∞) you could use for the proposals.

"""
print("λ = scaling parameter i.e scale =0.001 , I have observed that smaller the scale value goes more optimal exponential distribution is generated. So in this case out of all scale values I would consider scale = 0.001 as best parameter for our λ.")
print("scale = 0.001")
"""
c)  Using  the  optimal λ,  how  many  of  the  generated  exponentially  dis-tributed proposals do you expect to 
    accept (as a percentage)?
"""
print("The percentage of accepted distributed proposals")
print(100-( (k-i)/k)*100)
    
"""
d)Write  Python  codes  to  generate  positive  normals  using  the  Accept-Reject Algorithm.

"""
print("The positive normal distribution values are plotted as follow: ")
sns.distplot(z, hist = True, kde = True)
plt.show()


# In[498]:


"""
Acceptance-Rejection method
Denote the density of X by f . This method requires a function g that majorizes f ,
g(x) ≥ f (x)
for all x. Now g will not be a density, since
c = {-∞, ∞}g(x)dx ≥ 1.
Assume that c < ∞. Then h(x) = g(x)/c is a density. Algorithm:
1. Generate Y having density h;
2. Generate U from U(0, 1), independent of Y ;
3. If U ≤ f (Y )/g(Y ), then set X = Y ; else go back to step 1.
The random variable X generated by this algorithm has density f .

Validity of the Acceptance-Rejection method
Note
P(X ≤ x) = P(Y ≤ x|Y accepted).
Now,
P(Y ≤ x, Y accepted) ={x,−∞}f (y)/g(y)*h(y)dy =1/c*{x,−∞}f (y)dy,
and thus, letting x → ∞ gives
P(Y accepted) =1/c.
Hence,
P(X ≤ x) =P(Y ≤ x, Y accepted)/P(Y accepted)={x,−∞}f (y)dy.

Source="https://www.win.tue.nl/~marko/2WB05/lecture8.pdf"


c=sqrt(2e/π)≈1.32.
Source ="https://www.scss.tcd.ie/Brett.Houlding/Domain.sites2/sslides5.pdf"

"""

