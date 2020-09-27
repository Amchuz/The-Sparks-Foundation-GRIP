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
        print("Now let's view some graphs")
        print("Visualization of our dataset")
        data.plot(x='Hours', y='Scores', style='o')  
        plt.title('Hours vs Score')  
        plt.xlabel('Hours Studied')  
        plt.ylabel('Percentage Score')  
        plt.show()
        print("Hours and Scores have a really good correlation between them.")
    print(" ")
    print(" ")
    print("Let's use Linear Regression for the data")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    print("\nDivide the data into attributes(inputs) and labels(outputs)")
    x = data.iloc[:, :-1].values  
    y = data.iloc[:, 1].values
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
    size=size.lower()
    if(('y' in size) or ('yes' in size)):
        tsize = int(input("Specify the test size you want : "))
        tsize /= 100
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = tsize, random_state = 0)
        print("Splitted with test size ",tsize)
    else :
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        print("Splitted with default test size")

    print(" ")
    print("\nLets explore the splitted data : ")
    print("X_Train data  : ", x_train.shape)
    print("X_Test data   : ", x_test.shape)
    print("Y_Train data  : ", y_train.shape)
    print("Y_Test data   : ", y_test.shape)

    print(" ")
    print(" ")   
    print("Training the Algorithm")
    print("We are going to train using Linear Regression")
    print(" ")
    lr = LinearRegression()  
    lr.fit(x_train, y_train) 
    print("Training completed")
    line = lr.coef_*x+lr.intercept_
    print(" ")
    print("Let's visualize the result : ")
    plt.scatter(x, y)
    plt.plot(x, line)
    plt.title('Hours vs Score')  
    plt.xlabel('Hours Studied')  
    plt.ylabel('Percentage Score')  
    plt.show()
        
    print(" ") 
    print(" ")
    print("Testing the Algorithm")
    y_pred = lr.predict(x_test)
    print(" ")
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df
    print("Completed")
    print(" ")
    print(" ")
    print("Let's check out some evaluation metrics")
    print('Mean Absolute Error     : ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error      : ', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(" ")
    print(" ")
    Flag=True
    while Flag==True:
        print("\nWould you like to give try different hour ?")
        print("y - Yes")
        print("n - No")
        time=input("Your answer : ")
        time=time.lower()
        if(('y' in time) or ('yes' in time)):
            hour=float(input("How many hours did the person study : "))
            y_pred = lr.predict([[hour]])
            print("The predicted score is ", y_pred)
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
    print("Yippee... We learned to use Linear Regression")
    print("I really had fun")
    print("Hope you enjoyed it too")
    print("Bye")

if __name__=="__main__":
    print("Hello there,")
    print("My name is Prify Philip")
    print("Do you have a dataset to work with ?")
    print("By the way, the name of the dataset I have provided is data.")
    data = input("Enter the name of your dataset : ")
    data = pd.read_csv(data)
    main(data)
