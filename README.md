# Classification Modeling Using SkLearn

## Problem
I wanted to make a model that if fed some input's predicts the car's deal type (great, good, fair). I also sought to determine the best-suited, high-accuracy model for this particular multiclass classification model.

## How I solved it
I used the pre-existing models in the <a href="https://scikit-learn.org/stable/">sklearn library</a> to basically do multiclass classification using the data provided.
Since sklearn does not allow string inputs to its models, I had to do some preprocessing on the data.
Most of the preprocessing was done in <a href="src/preprocessing.py">preprocessing.py</a>. 
I got removed any columns that contained strings using the following code.

    new_data.drop(['Make','Model', 'Used/New','SellerType', 'SellerName', 'StreetName', 'State', 'ExteriorColor', 'InteriorColor', 'Drivetrain', 'FuelType', 'Transmission', 'Engine', 'VIN', 'Stock#'],axis=1,inplace=True)
 
> Columns that were axed: <br/>
 Make , Model ,  Used/New , SellerType ,  SellerName ,  StreetName ,  State ,  ExteriorColor ,  InteriorColor ,  Drivetrain ,  FuelType ,  Transmission ,  Engine ,  VIN ,  Stock# 

I also went through the Zipcodes Column and cleared any non numerical value.

    new_data = cars_data[cars_data['DealType'].str.contains('NA') == False]
    new_data = new_data[cars_data['Price'].str.contains('Not Priced') == False]
    new_data = new_data[cars_data['Zipcode'].str.contains('City') == False]
    new_data = new_data[cars_data['Zipcode'].str.contains('B') == False]
    new_data = new_data[cars_data['Zipcode'].str.contains('Loop') == False]
    new_data = new_data[cars_data['Zipcode'].str.contains('Fox') == False]
    new_data = new_data[cars_data['Zipcode'].str.contains('Falls') == False]
    new_data = new_data[cars_data['Zipcode'].str.contains('Leesburg') == False]
    new_data = new_data[cars_data['Zipcode'].str.contains('Smithville') == False]

After doing all of this, I seperated 80% of the data to be training data for the models, and left the rest 20% for testing the models.
I could have used the sklearn prepocessing to seperate the data but at the time I did not know of it. 
While doing all of this, I also replaced the DealType values with numerical representations. 

    DealTypeMapping = {'Great': 0, 'Good': 1, 'Fair': 2}
    new_data.replace({'DealType': DealTypeMapping}, inplace=True)

All of this was done using the <a href="https://pandas.pydata.org/">Pandas</a> library.

> I think now it is best to mention the fact that I have not worked with sklearn before and this show's in my lack of understanding of its structure and models.<br/>
I took this oppurtuinity to learn more about sklearn and machine learning.</br>

What I had left after all the preprocessing was a really nice numerical data set of car listings to determine if a listing was great, good, or fair deal.

After all the preprocessing was done, and the preprocessed info was saved in new csv files in the <a href="/data">data folder</a>, I moved on to training the multiclass classification models. This training was done in the <a href="/src/training.py">training.py file</a>. The main purpose was to see which model performed better in predicting the classification after being trained. I used three models from sklearn that natively supported multiclass classificatin: **DecisionTreeClassifier**, **MLPClassifier**, and **LabelPropagation**.

The base models for DecisionTreeClassifer, and LabelPropagation performed poorly interms of accuracy of their predictions. See figures 1.1, and 1.2 for a visualization of the results.  

<p align="center">
  <img src="/data/figures/Figure_1.1.png" />
  </br>
  <b>Figure 1.1</b>
</p>
<p align="center">
  <img src="/data/figures/Figure_1.2.png" />
  </br>
  <b>Figure 1.1</b>
</p>

DesicionTreeClassifier had an overall lack in accuracy due to no changes/optimizations to its basic structure. The accuracy always lands around at 55% with the base model. For LabelPropagation, it was a bit of a different story, the model itself has really high accuracy (100%) for two of the classes, while one severely lack behind with a accuracy of only 28%. However, the recall distribution is what concerns me. 

> Recall is the ratio of true positives and true negatives.

So for it to be really close to zero for two of the classes, and then really high recall for one class is odd. So I don't know if its just some pardoxical result or if it is a viable result.

For MLP Classifier, the state of the model itself changes every single training session. Without setting a seed for the Model, you will get inconsistent results. The two figures show this in effect.

<p align="center">
  <img src="/data/figures/Figure_1.3.png" />
  </br>
  <b>Figure 1.3</b>
</p>
<p align="center">
  <img src="/data/figures/MLPClassifier.png" />
  </br>
  <b>Figure 1.4</b>
</p>

> Figure 1.3 is a visualization of the model without a set random state. Figure 1.4 is a visualization with a set random state.

So with some changes to the model by setting some key parameters I improved the accuracy of the DecisionTreeClassifier. I changed the criterion parameter to log_loss, which improved upon the defualt option, however not by much. Then I also tried to change the class_weight to something the values all classes equally, however, I still did not see much of a change in accuracy, which was now a consitent 57%. The following figure was what I was left with.

<p align="center">
  <img src="/data/figures/DecisionTree.png" />
  </br>
  <b>Figure 1.5</b>
</p>

Since I found no notable success with the DecisionTreeClassifier, I moved on to label propagation changing the kernel and max_iter params, to see if I could get a more evenly distributed model, something not as paradoxical as before. Figure 1.6 showcases this model.

<p align="center">
  <img src="/data/figures/LabelPropagation.png" />
  </br>
  <b>Figure 1.5</b>
</p>

> When tested with some of my own test cases, the previous model performed way better in terms of accuracy of its results. So arguably maybe the previous version was better than this one.

The only changes I made to the MLPClassifier was to add a set random_state to the model, so that its results were replicable. Figure 1.3 already showcases this.

These were all the results for the different Models. However, there was something perticular about most of the trained models. For most the Good rating always did better than the Great, and Fair ratings. I think this was probably due to the uneven distribution of Good, Great, and Fair in the training and testing sets. Maybe if I had equaly large training and testing data for the Great, and Fair classes, I would have better accuracy across the different Models used. I couldn't try out my hypothesis this time around, since I did not have access to some more data close to what was listed on Kradle.

> On another note the visualizations shown in this ReadMe were made using the <a href="https://www.scikit-yb.org/">YellowBricks library</a>.

I came into this whole coding challenge knowing very little about machine learning and classification with sklearn, however, while trying to solve my problems I figured out some key information regarding the library and classification tasks themselves that I might use in future models.

## Conclusion
Decision Trees are probably the best suited model for this classification task if we do not take into account the paradoxical Label propagation model. The MLP Classifier could have attained a higher accuracy if I could have found a good random state where it beatout the Decision Tree. The Label Propagtion (with param changes) wasn't far behind the Decision Tree, maybe I could have tuned it to work better than the Decision Tree. It could be said that tuning these models might result in higher accuracy, however, I think maybe a better distributed data set would have also increased the overall accuracy quite a bit.

If you would like to try and have the models predict your own data, go to the <a href="/data/ownpred.csv">ownpred.csv file</a> and add your data. Then go to the <a href="/src/training.py">training.py file</a> and change the following code:

    testing_data = pd.read_csv('data/testing_set.csv')

    x_test = testing_data.drop('DealType', axis=1)
    y_test = testing_data['DealType']

To this:
    
    testing_data = pd.read_csv('data/ownpred.csv')

    x_test = testing_data.drop('DealType', axis=1)
    y_test = testing_data['DealType']


Then by viewing the terminal output you can determine which model got the prediction right.

## Resources

<a href="https://pandas.pydata.org/">Pandas</a><br/>
<a href="https://scikit-learn.org/stable/">Sklearn library</a><br/>
<a href="https://www.scikit-yb.org/">YellowBricks library</a><br/>
<a href="https://en.wikipedia.org/wiki/Multiclass_classification">Classification Wiki</a><br/>
<a href="https://machinelearningmastery.com/types-of-classification-in-machine-learning/">Classification Article</a><br/>