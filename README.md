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

df=pd.read_csv("/content/Data_to_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

sm.qqplot(df["Moderate Negative Skew_1"],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```

# OUTPUT:

<img width="613" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/36f3bb1a-e0eb-403c-8dc8-a1d25817f3b7">



<img width="400" alt="Screenshot 2023-10-14 204023" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/0aa78a1d-e050-40a5-a9a9-6b777ebb4706">



<img width="700" alt="Screenshot 2023-10-14 204103" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/48a777bf-c180-45c7-aade-a4af513cb513">



<img width="801" alt="Screenshot 2023-10-14 204202" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/9533d316-1202-444f-9e8b-b15acd7599d8">



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
