# Ex-06 Feature Transformation

 # AIM:

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:

 STEP 1:
 
Read the given Data

 STEP 2:
 
Clean the Data Set using Data Cleaning Process

 STEP 3:
 
Apply Feature Transformation techniques to all the features of the data set

 STEP 4:
 
Print the transformed features.


# PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer 
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()

```

# OUTPUT:

<img width="613" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/36f3bb1a-e0eb-403c-8dc8-a1d25817f3b7">


<img width="495" alt="Screenshot 2023-10-14 224339" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/5edc5804-4b7f-4c53-9119-fdf3c6812a98">


<img width="871" alt="Screenshot 2023-10-14 185503" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/1eb05e87-a997-49c9-93d6-4a4151d6f3e8">


<img width="506" alt="Screenshot 2023-10-14 185826klklm" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/843609ab-6c86-405b-a62b-f9990f6b268f">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/c3953e60-cbb0-4f07-a0a7-9a452ee0e89f">


<img width="527" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/e9b5d5bb-1f21-4d99-92ad-1c40bffe65e8">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/deca5470-02c6-40e7-ab47-ed5375e0c2bc">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/3735b6af-3b06-4fac-9411-06d58f2ffe35">


<img width="530" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/96354cfd-37b0-4252-ab7e-a73df3d3e8f1">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/6aa9bbbb-ed07-4852-876c-281134770c98">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/39d8a3e8-d4f6-40cd-84ff-afa6ffe6a982">


<img width="525" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/6d9d9984-b1b8-415a-9adf-fc1ae9e80a12">


<img width="527" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/33f0d518-cd9c-430b-94a0-216eaf3e7c01">


<img width="530" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/06daaf43-f38e-4292-8040-a2897b973a50">


# RESULT:

Thus feature transformation is done for the given dataset.
