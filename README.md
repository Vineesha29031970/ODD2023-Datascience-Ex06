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


<img width="871" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/074e18b7-3d8a-4a99-a195-310133ea167b">


<img width="572" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/5030103f-c6ae-40f9-a0ea-c4349ae9b392">


<img width="436" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/5f43b6cc-6dee-46d6-8381-fbb3702e7798">


<img width="424" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/947cc7c2-e4bb-4466-a28e-5ab36a8e487f">


<img width="531" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/f99522f7-741d-4567-bc5a-0ca5d95fa870">


<img width="530" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/144f6694-9859-4a7e-9933-307b7e125186">


<img width="530" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/02d8a8a4-fc8a-4e15-8274-530ecc62603f">


<img width="532" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/2944ae94-efaf-411f-961f-ef6d2f2c2336">


<img width="527" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/4a549577-6594-4193-b276-5a9376334229">


<img width="528" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/bf6ae1ef-1bc4-44ff-8ba8-3bb2d493d86a">


<img width="527" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/1e3d8b99-512d-4937-9df5-41a2766d73b7">


<img width="527" alt="image" src="https://github.com/Vineesha29031970/ODD2023-Datascience-Ex06/assets/133136880/bd7ea83d-f37e-4e50-b287-3a3817e3c1b2">


RESULT:

Thus feature transformation is done for the given dataset.
