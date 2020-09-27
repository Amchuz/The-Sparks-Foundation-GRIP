"""
Code is developed on Visual Studio Code (Ubuntu OS)
Created by: Prify Philip
"""

from os import system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
def main(data):
    y = data['Species']
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
        print(" ")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.externals.six import StringIO  
    from sklearn.tree import export_graphviz
    import pydotplus
    print(" ")
    print(" ")
    print("Ok, now let's train the model and make predictions using it")
    print(" ")
    print(" ")
    print("\nDivide the data into attributes(inputs) and labels(outputs)")
    x = data.iloc[:, [0, 1, 2, 3]].values 
    le = LabelEncoder()
    data['Species'] = le.fit_transform(data['Species'])
    y = data['Species'].values 
    print("\nAttributes:\n",x)
    print("\nLables :",y)
    print("Next step is to split this data into training and test sets.")   

    print(" ")
    print(" ")
    print("\nTrain-Test-Split : ")
    print("The test size in default is 20. Would you like to change it ?")
    print("y- Yes")
    print("n- No")
    size=input("\nYour choice : ")
    if(('y' in size) or ('yes' in size)):
        tsize = int(input("Specify the test size you want : "))
        tsize /= 100
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=tsize,random_state=0)
        print("Splitted with test size ",tsize)
    else :
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        print("Splitted with default test size")
    print(" ")
    print(" ")
    print("\nLets explore the splitted data : ")
    print("X_Train data  : ", x_train.shape)
    print("X_Test data   : ", x_test.shape)
    print("Y_Train data  : ", y_train.shape)
    print("Y_Test data   : ", y_test.shape)
    print(y_test)

    print(" ")
    print(" ")   
    print("Training the Algorithm")
    print("We are going to train the model")
    print("Which training method do you want ?")
    print("g - Gini")
    print("e - Entropy")
    method=input("\nYour choice : ")
    method=method.lower()
    if(('g' in method) or ('gini' in method)):
        dtmodel=DecisionTreeClassifier(criterion="gini",random_state=0,max_depth=3,min_samples_leaf=5)
        dtmodel.fit(x_train,y_train)
    elif(('e' in method) or ('entropy' in method)):
        dtmodel=DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=3,min_samples_leaf=5)
        dtmodel.fit(x_train,y_train)
    else:
        print("Wrong Choice")
        return()
    print(" ") 
    print(" ")
    print("Testing the Algorithm")
    y_pred=dtmodel.predict(x_test)
    print("Predicted values:")
    print(y_pred)
    print("Completed")
    print("Accuracy:",accuracy_score(y_test,y_pred)*100)
    print("Report:",classification_report(y_test,y_pred))
    print("Confusion Matrix : ")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(dtmodel.score(x_test, y_test))
    plt.title(all_sample_title, size = 15)
    plt.show()

    print(" ")
    print(" ")
    print("Now, let's visualize the Decision Tree to understand it better.")
    df=data.copy()
    df=df.drop('Species',axis=1)
    dot_data = StringIO()
    export_graphviz(dtmodel, out_file=dot_data, feature_names=df.columns, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png("dtree.png")
    im = Image.open(r"dtree.png")  
    im.show()
    print(" ")
    print(" ")

    Flag=True
    while Flag==True:
        print("\nWould you like to give try another input ?")
        print("y - Yes")
        print("n - No")
        sp=input("Your answer : ")
        sp=sp.lower()
        if(('y' in sp) or ('yes' in sp)):
            spe=[]
            slen=float(input("Give Sepal Length in cm : "))
            spe.append(slen)
            swid=float(input("Give Sepal width in cm : "))
            spe.append(swid)
            plen=float(input("Give Petal Length in cm : "))
            spe.append(plen)
            pwid=float(input("Give Petal width in cm : "))
            spe.append(pwid)
            y_pred = dtmodel.predict([spe])
            print(" ")
            print(" ")
            print("Species according to encoding : ")
            print("0 - Iris-setosa")
            print("1 - Iris-versicolor")
            print("2 - Iris-virginica")
            print(" ")
            print(" ")
            print("The predicted species is ", y_pred)
            print("Were you expecting the same ?")
            print("OK")
            print("Do you want to try again ?")
            print("y - Yes")
            print("n - No")
            ans = input("Your choice : ")
            if(('y' in ans) or ('yes' in ans)):
                Flag = True
            else:
                Flag = False
        else:
            Flag = False

    print(" ")
    print(" ")
    print("Yippee... We learned to use Decision Tree")
    print("I really had fun")
    print("Hope you enjoyed it too")
    print("Bye")


if __name__=="__main__":
    print("Hello there,")
    print("My name is Prify Philip")
    import sklearn.datasets as datasets
    import pandas as pd
    data = pd.read_csv('Iris.csv', index_col = 0)
    main(data)