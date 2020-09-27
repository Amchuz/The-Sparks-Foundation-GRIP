"""
Code is developed on Visual Studio Code (Ubuntu OS)
Created by: Prify Philip
"""

from os import system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def main(data):
    print(" ")
    print("Before going into our task do you want to check out the data ? ")
    print(" ")
    print("y- Yes")
    print("n-No")
    choice = input("Select your choice : ")
    choice = choice.lower()

    if(('y' in choice) or ('yes' in choice)):
        print(" ")
        print(" ")
        print("Ok ! Let's explore the data") 
        print('\nOur dataset looks like : \n',data.head())
        print(" ")
        print('\nThe shape of the data is: ',data.shape)
        print(" ")
        print("\nWhat about the datatypes: \n",data.dtypes)
        print(" ")
        print("\nThe whole data can be described as : \n",data.describe())
        print(" ")
        print("\nChecking for null values: \n",data.isnull().sum())
        print(" ")
        print("Checked if we had any null values")
        data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        print(" ")
        print(" ")
    print("Here we are using Kmeans clustering. That means, we need to find the optimum number of clusters K")
    print("Well.. how do we find that ?")
    print("Elbow method to the rescue !")
    x = data.iloc[:, [0, 1, 2, 3]].values

    from sklearn.cluster import KMeans
    wcss = []

    for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
            kmeans.fit(x)
            wcss.append(kmeans.inertia_)
    
    print("Plotting the result on a line graph")
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') # Within cluster sum of squares
    plt.show()
    print(" ")
    print(" ")
    print("From the graph we are choosing k=3")   
    print("Applying kmeans to the dataset ")
    kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    y_kmeans = kmeans.fit_predict(x)   

    print(" ")
    print(" ")
    print("Now let's visualize the clusters - On the first two columns")
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
    print("Plotting the centroids of the clusters")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
    plt.legend()
    plt.show()
    print(" ")
    print(" ")
    print("Yippee... We learned to use Kmeans clustering")
    print("I really had fun")
    print("Hope you enjoyed it too")
    print("Bye")

if __name__=="__main__":
    print("Hello there,")
    print("My name is Prify Philip")
    import sklearn.datasets as datasets
    import pandas as pd
    iris=datasets.load_iris()
    data=pd.DataFrame(iris.data, columns=iris.feature_names)
    main(data)
