'''
Main file that contains all the model training
Tried out many different sklearn models but found Descision Trees, MLPClassifiers, and LabelPropagation to work best
The accuracy for most was below 50%
With some skewed results due to the uneven distribution of test data for each classification
You can try to input some of the attributes of a car to see if it is a good deal or not
Otherwise you can be shown the classification report
The models are somewhat optimized for the best accuracy
'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import classification_report, confusion_matrix

from yellowbrick.classifier import ClassificationReport

def main():
    #Some Preprocessing
    training_data = pd.read_csv('data/training_set.csv')
    testing_data = pd.read_csv('data/testing_set.csv')

    x_test = testing_data.drop('DealType', axis=1)
    y_test = testing_data['DealType']

    x_train = training_data.drop('DealType', axis=1)
    y_train = training_data['DealType']

    #Train Classification models
    #classifier = DecisionTreeClassifier()
    classweight = {0: 2.36875, 1: 1, 2:5.36320754717}
    classifier = DecisionTreeClassifier(criterion='log_loss', class_weight=classweight)
    classifier.fit(x_train, y_train)

    #classifierTwo = LabelPropagation()
    classifierTwo = LabelPropagation(kernel='knn', max_iter=2000)
    classifierTwo.fit(x_train, y_train)

    #classifierThree = MLPClassifier()    
    classifierThree = MLPClassifier(random_state=10)
    classifierThree.fit(x_train, y_train)

    #Predict Against testing set to determine accuracy of predictions
    y_pred = classifier.predict(x_test)

    print(classification_report(y_test, y_pred))

    y_pred = classifierTwo.predict(x_test)

    print(classification_report(y_test, y_pred))

    y_pred = classifierThree.predict(x_test)

    print(classification_report(y_test, y_pred))

    classes = ['Great', 'Good', 'Fair']
    visualizer = ClassificationReport(classifier, classes=classes, support=True)

    visualizer.fit(x_train, y_train)        
    visualizer.score(x_test, y_test)         
    visualizer.show()                       

    visualizer = ClassificationReport(classifierTwo, classes=classes, support=True)

    visualizer.fit(x_train, y_train)        
    visualizer.score(x_test, y_test)         
    visualizer.show()    

    visualizer = ClassificationReport(classifierThree, classes=classes, support=True)

    visualizer.fit(x_train, y_train)        
    visualizer.score(x_test, y_test)         
    visualizer.show()    
    
if __name__ == '__main__':
    main()