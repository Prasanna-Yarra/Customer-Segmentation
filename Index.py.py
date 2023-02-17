import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

data = pd.read_csv('Customers.csv')
data

data.head()

data.info()

data.describe()

data.corr()

data.isnull().sum()

sns.countplot(x="Gender",data = data) #Count the number of each gender

#Count the range of the age of customers
age10_20 = data.Age[(data.Age <= 20) & (data.Age >= 10)]
age21_30 = data.Age[(data.Age <= 30) & (data.Age >= 21)]
age31_40 = data.Age[(data.Age <= 40) & (data.Age >= 31)]
age41_50 = data.Age[(data.Age <= 50) & (data.Age >= 41)]
age50above = data.Age[data.Age >= 51]

x = ["10-20","21-30","31-40","41-50","50+"]
y = [len(age10_20),len(age21_30),len(age31_40),len(age41_50),len(age50above)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age range")
plt.ylabel("Number of Customer")
plt.show()

#Distribution of age
plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(data['Age'])
plt.title('Distribution of Age', fontsize = 20)
plt.xlabel('Range of Age')
plt.ylabel('Count')

#Number of customers according to their annual income. 

ai0_30 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 0) & (data["Annual Income (k$)"] <= 30)]
ai31_60 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 31) & (data["Annual Income (k$)"] <= 60)]
ai61_90 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 61) & (data["Annual Income (k$)"] <= 90)]
ai91_120 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 91) & (data["Annual Income (k$)"] <= 120)]
ai121_150 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 121) & (data["Annual Income (k$)"] <= 150)]

Money_range = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
number = [len(ai0_30), len(ai31_60), len(ai61_90), len(ai91_120), len(ai121_150)]

plt.figure(figsize=(15,6))
sns.barplot(x=Money_range, y=number)
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()

#Distribution of Annnual Income
plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(data['Annual Income (k$)'])
plt.title('Distribution of Annual Income (k$)', fontsize = 20)
plt.xlabel('Range of Annual Income (k$)')
plt.ylabel('Count')

#Number of customers according to their spending scores.

ss1_20 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 1) & (data["Spending Score (1-100)"] <= 20)]
ss21_40 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 21) & (data["Spending Score (1-100)"] <= 40)]
ss41_60 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 41) & (data["Spending Score (1-100)"] <= 60)]
ss61_80 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 61) & (data["Spending Score (1-100)"] <= 80)]
ss81_100 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 81) & (data["Spending Score (1-100)"] <= 100)]

score_range = ["1-20", "21-40", "41-60", "61-80", "81-100"]
number = [len(ss1_20), len(ss21_40), len(ss41_60), len(ss61_80), len(ss81_100)]

plt.figure(figsize=(15,6))
sns.barplot(x=score_range, y=number)
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()

#Distribution of spending score
plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(data['Spending Score (1-100)'])
plt.title('Distribution of Spending Score (1-100)', fontsize = 20)
plt.xlabel('Range of Spending Score (1-100)')
plt.ylabel('Count')

data1=data.copy()
data1.drop('CustomerID',axis=1,inplace=True)
sns.pairplot(data1) # Gives insight of how columns are related to each other

df1=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
X=df1[["Annual Income (k$)","Spending Score (1-100)"]]

#Scatterplot of the input data

sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',  data = X  ,s = 60 )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()

#Importing KMeans from sklearn
from sklearn.cluster import KMeans

wcss=[]  #the sum of squared distance between each point and the centroid in a cluster
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(12,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

model = (KMeans(n_clusters = 5) )#Taking 5 clusters
model.fit(X)#Fitting the input data

label = model.labels_
X["label"]=label
centroids = model.cluster_centers_

print(label)  #Labels of each point, means to which cluster they belong
print(centroids) #centroids of each cluster in both column

df1

plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label",  
                 palette=['green','orange','brown','dodgerblue','yellow'], legend='full',data = df1  ,s = 60 )
for i in range(5):
    plt.scatter(centroids[i][0],centroids[i][1],c='black')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()

cust1=df1[df1["label"]==0]
print('Number of customer in 1st group=', len(cust1))
print('They are -', cust1["CustomerID"].values)
print("--------------------------------------------")
cust2=df1[df1["label"]==1]
print('Number of customer in 2nd group=', len(cust2))
print('They are -', cust2["CustomerID"].values)
print("--------------------------------------------")
cust3=df1[df1["label"]==2]
print('Number of customer in 3rd group=', len(cust3))
print('They are -', cust3["CustomerID"].values)
print("--------------------------------------------")
cust4=df1[df1["label"]==3]
print('Number of customer in 4th group=', len(cust4))
print('They are -', cust4["CustomerID"].values)
print("--------------------------------------------")
cust5=df1[df1["label"]==4]
print('Number of customer in 5 group=', len(cust5))
print('They are -', cust5["CustomerID"].values)
print("--------------------------------------------")

df2=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
X2=df2[["Age","Spending Score (1-100)"]]

#Scatterplot of the input data

sns.scatterplot(x ="Age",y = "Spending Score (1-100)" ,  data = X2  ,s = 60 )
plt.ylabel("Spending Score (1-100)")
plt.xlabel("Age") 
plt.title('Age vs Spending Score (1-100)')
plt.show()

#Importing KMeans from sklearn
from sklearn.cluster import KMeans
wcss=[]  #the sum of squared distance between each point and the centroid in a cluster
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X2)
    wcss.append(km.inertia_)

plt.figure(figsize=(12,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

model = (KMeans(n_clusters = 4) )#Taking 4 clusters
model.fit(X2)#Fitting the input data

label = model.labels_
df2["label"]=label
centroids = model.cluster_centers_

print(label)  #Labels of each point, means to which cluster they belong
print(centroids) #centroids of each cluster in both column
df2

plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Age',y = 'Spending Score (1-100)',hue="label",  
                 palette=['green','orange','brown','yellow'], legend='full',data = df2  ,s = 60 )
for i in range(4):
    plt.scatter(centroids[i][0],centroids[i][1],c='black')
plt.xlabel('Age') 
plt.ylabel('Spending Score (1-100)')
plt.title('Age vs Spending Score (1-100)')
plt.show()

cust1=df2[df2["label"]==0]
print('Number of customer in 1st group=', len(cust1))
print('They are -', cust1["CustomerID"].values)
print("--------------------------------------------")
cust2=df2[df2["label"]==1]
print('Number of customer in 2nd group=', len(cust2))
print('They are -', cust2["CustomerID"].values)
print("--------------------------------------------")
cust3=df2[df2["label"]==2]
print('Number of customer in 3rd group=', len(cust3))
print('They are -', cust3["CustomerID"].values)
print("--------------------------------------------")
cust4=df2[df2["label"]==3]
print('Number of customer in 4th group=', len(cust4))
print('They are -', cust4["CustomerID"].values)
print("--------------------------------------------")

df3=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
X3=df3[["Gender","Age","Spending Score (1-100)"]]

dummies=pd.get_dummies(X3["Gender"])
dummies
X3=pd.concat([X3,dummies],axis='columns')

X3.drop('Gender',axis='columns')

X3['Gender'] = X3['Male']
X3.drop(['Male','Female'],axis='columns')

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X3['Gender'],X3['Age'],X3['Spending Score (1-100)'])
ax.view_init(35, 185)
ax.set_xlabel("Gender")
ax.set_ylabel("Age")
ax.set_zlabel("Spending Score (1-100)")
plt.show()

from sklearn.cluster import KMeans
wcss=[]  #the sum of squared distance between each point and the centroid in a cluster
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X3)
    wcss.append(km.inertia_)

plt.figure(figsize=(12,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

model = (KMeans(n_clusters = 4) )#Taking 5 clusters
model.fit(X3)#Fitting the input data

label = model.labels_
X3["label"]=label
centroids = model.cluster_centers_

print(label)  #Labels of each point, means to which cluster they belong
print(centroids) #centroids of each cluster in both column
X3

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X3["Gender"][X3.label == 0], X3["Age"][X3.label == 0],  X3["Spending Score (1-100)"][X3.label == 0], c='purple', s=60)
ax.scatter(X3["Gender"][X3.label == 1], X3["Age"][X3.label == 1],  X3["Spending Score (1-100)"][X3.label == 1], c='red', s=60)
ax.scatter(X3["Gender"][X3.label == 2], X3["Age"][X3.label == 2],  X3["Spending Score (1-100)"][X3.label == 2], c='blue', s=60)
ax.scatter(X3["Gender"][X3.label == 3], X3["Age"][X3.label == 3],  X3["Spending Score (1-100)"][X3.label == 3], c='green', s=60)

ax.view_init(35, 185)

plt.xlabel("Gender")
plt.ylabel("Age)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

df4=data[["CustomerID","Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
X4=data[["Age","Annual Income (k$)","Spending Score (1-100)"]]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X4['Age'],X4['Annual Income (k$)'],X4['Spending Score (1-100)'])
ax.view_init(35, 185)

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.show()

from sklearn.cluster import KMeans
wcss=[]  #the sum of squared distance between each point and the centroid in a cluster
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X4)
    wcss.append(km.inertia_)

plt.figure(figsize=(12,6))    
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()



model1 = (KMeans(n_clusters = 5) )#Taking 5 clusters
model1.fit(X4)#Fitting the input data

labels = model.labels_
df4["label"]=labels
centroids = model.cluster_centers_
print(labels)  #Labels of each point, means to which cluster they belong
print(centroids) #centroids of each cluster in both column
df4



#3D Plot as we did the clustering on the basis of 3 input features
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df4.Age[df4.label == 0], df4["Annual Income (k$)"][df4.label == 0], df4["Spending Score (1-100)"][df4.label == 0], c='purple', s=60)
ax.scatter(df4.Age[df4.label == 1], df4["Annual Income (k$)"][df4.label == 1], df4["Spending Score (1-100)"][df4.label == 1], c='red', s=60)
ax.scatter(df4.Age[df4.label == 2], df4["Annual Income (k$)"][df4.label == 2], df4["Spending Score (1-100)"][df4.label == 2], c='blue', s=60)
ax.scatter(df4.Age[df4.label == 3], df4["Annual Income (k$)"][df4.label == 3], df4["Spending Score (1-100)"][df4.label == 3], c='green', s=60)
#ax.scatter(df4.Age[df4.label == 4], df4["Annual Income (k$)"][df4.label == 4], df4["Spending Score (1-100)"][df4.label == 4], c='yellow', s=60)
ax.view_init(35, 185)

plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

cust1=df4[df4["label"]==1]
print('Number of customer in 1st group=', len(cust1))
print('They are -', cust1["CustomerID"].values)
print("--------------------------------------------")
cust2=df4[df4["label"]==2]
print('Number of customer in 2nd group=', len(cust2))
print('They are -', cust2["CustomerID"].values)
print("--------------------------------------------")
cust3=df4[df4["label"]==0]
print('Number of customer in 3rd group=', len(cust3))
print('They are -', cust3["CustomerID"].values)
print("--------------------------------------------")
cust4=df4[df4["label"]==3]
print('Number of customer in 4th group=', len(cust4))
print('They are -', cust4["CustomerID"].values)
print("--------------------------------------------")