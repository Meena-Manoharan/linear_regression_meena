import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

mat_data = pd.read_csv("student-mat.csv")

mat_data



print('Total number of students: ',len(mat_data))

print("Parameter are: ",mat_data.columns)

mat_data.info()

mat_data.describe()

mat_data['G3'].describe()

sns.set(rc={'figure.figsize':(8,6)})
sns.countplot(x="school", hue ="sex", data=mat_data)

sns.countplot(mat_data.age)

sns.countplot(x="school", hue ="studytime", data=mat_data)

sns.countplot(x="age", hue="failures", data=mat_data)

sns.distplot(mat_data.absences)

sns.set_style('whitegrid')
sns.countplot(x='sex',data=mat_data,palette='plasma')

b = sns.kdeplot(mat_data['age'])
b.axes.set_title('Ages of students')
b.set_xlabel('Age')
b.set_ylabel('Count')
plt.show()

sns.lmplot(x ="failures", y ="G3", data = mat_data, order = 2, ci = None)

mat_data['G3'].describe()

GP = mat_data[mat_data.school == 'GP']
MS = mat_data[mat_data.school == 'MS']

sns.distplot(GP.G3, hist=False, label="GP")
sns.distplot(MS.G3, hist=False, label="MS")
plt.show()

"""Encoding categorical variables"""

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mat_data.iloc[:,0]=le.fit_transform(mat_data.iloc[:,0])
mat_data.iloc[:,1]=le.fit_transform(mat_data.iloc[:,1])
mat_data.iloc[:,3]=le.fit_transform(mat_data.iloc[:,3])
mat_data.iloc[:,4]=le.fit_transform(mat_data.iloc[:,4])
mat_data.iloc[:,5]=le.fit_transform(mat_data.iloc[:,5])
mat_data.iloc[:,7]=le.fit_transform(mat_data.iloc[:,7])
mat_data.iloc[:,8]=le.fit_transform(mat_data.iloc[:,8])
mat_data.iloc[:,9]=le.fit_transform(mat_data.iloc[:,9])
mat_data.iloc[:,10]=le.fit_transform(mat_data.iloc[:,10])
mat_data.iloc[:,11]=le.fit_transform(mat_data.iloc[:,11])
mat_data.iloc[:,5]=le.fit_transform(mat_data.iloc[:,5])
mat_data.iloc[:,13]=le.fit_transform(mat_data.iloc[:,13])
mat_data.iloc[:,14]=le.fit_transform(mat_data.iloc[:,14])
mat_data.iloc[:,15]=le.fit_transform(mat_data.iloc[:,15])
mat_data.iloc[:,16]=le.fit_transform(mat_data.iloc[:,16])
mat_data.iloc[:,17]=le.fit_transform(mat_data.iloc[:,17])
mat_data.iloc[:,18]=le.fit_transform(mat_data.iloc[:,18])
mat_data.iloc[:,19]=le.fit_transform(mat_data.iloc[:,19])
mat_data.iloc[:,20]=le.fit_transform(mat_data.iloc[:,20])
mat_data.iloc[:,21]=le.fit_transform(mat_data.iloc[:,21])
mat_data.iloc[:,22]=le.fit_transform(mat_data.iloc[:,22])

mat_data.head()

"""# Finding Correlation between Attributes"""

mat_data.corr()['G3'].sort_values()

mat_data = mat_data.drop(['school', 'G1', 'G2'], axis='columns')

most_correlated = mat_data.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
most_correlated

mat_data = mat_data.loc[:, most_correlated.index]
mat_data.head()

"""# Grade Prediction"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mat_data, mat_data['G3'], test_size = 0.25, random_state=42)

X_train.head()

X_train = X_train.drop('G3', axis='columns')
X_test = X_test.drop('G3', axis='columns')
lr = LinearRegression()
model = lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

predictions = lr.predict(X_test)
print("Predicted Grade: ",predictions,sep = '\n')

plt.plot(y_test, predictions, 'o')
m, b = np.polyfit(y_test,predictions, 1)
plt.plot(y_test, m*y_test + b)
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")