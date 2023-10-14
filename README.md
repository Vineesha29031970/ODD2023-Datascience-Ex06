AIM:

To read the given data and perform Feature Transformation process and save the data to a file.

EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM:

STEP 1:

Read the given Data

STEP 2:

Clean the Data Set using Data Cleaning Process

STEP 3:

Apply Feature Transformation techniques to all the features of the data set

STEP 4:

Print the transformed features.


PROGRAM:


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


OUTPUT:


<img width="613" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/36f3bb1a-e0eb-403c-8dc8-a1d25817f3b7">


<img width="425" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/e3748e7b-edf2-4fc9-a708-27b887b5a826">


<img width="871" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/d29f25b5-6c06-4bd4-8af9-636ca9d9c2fd">


<img width="572" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/5030103f-c6ae-40f9-a0ea-c4349ae9b392">


<img width="436" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/5f43b6cc-6dee-46d6-8381-fbb3702e7798">


<img width="526" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/3a97d124-10c2-4293-84fb-812ea033267c">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/b896602e-b09f-4708-b325-a3e8c6d34682">


<img width="526" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/45121450-8390-4a65-900d-5307e2b5a0b0">


<img width="529" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/f4ab344c-3cdf-473e-aee5-f248eec26d1f">


<img width="525" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/2873d647-7a8f-4ee9-8d35-b4f08923c803">


<img width="527" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/b5e0acfb-e5cd-4935-955e-f91d6b147ddd">


<img width="526" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/b2c31c80-b4ed-4ff6-823a-fe08f1e67567">


<img width="530" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/4e43ade2-3720-4161-9511-d561ab24220c">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/c0760461-ddf6-4f36-aa3b-34a780388046">


<img width="530" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/998c953d-a189-4115-a934-5ec741c78f43">


RESULT:

Thus feature transformation is done for the given dataset.
