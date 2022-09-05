'''
Main file for preprocessing csv data to work with sklearn
SKLearn Models only takes numerical inputs for their training
Got rid of columns with no reasonable or helpful imapact on prediction
Removed columns with unique strings and converted those with repeating ones to numerical format
Divided the data into training and testing sets, could have done this using sklearn preprocessing but was not aware when the code was written
'''

import pandas as pd

cars_data = pd.read_csv("data/cars_raw.csv", encoding='utf-8')
print(cars_data.head)

TOTAL_ROWS, TOTAL_COLUMNS = cars_data.shape
DealTypeMapping = {'Great': 0, 'Good': 1, 'Fair': 2}
PriceMapping = {'$':''}

new_data = cars_data[cars_data['DealType'].str.contains('NA') == False]
new_data = new_data[cars_data['Price'].str.contains('Not Priced') == False]
new_data = new_data[cars_data['Zipcode'].str.contains('City') == False]
new_data = new_data[cars_data['Zipcode'].str.contains('B') == False]
new_data = new_data[cars_data['Zipcode'].str.contains('Loop') == False]
new_data = new_data[cars_data['Zipcode'].str.contains('Fox') == False]
new_data = new_data[cars_data['Zipcode'].str.contains('Falls') == False]
new_data = new_data[cars_data['Zipcode'].str.contains('Leesburg') == False]
new_data = new_data[cars_data['Zipcode'].str.contains('Smithville') == False]
new_data.drop(['Make','Model', 'Used/New','SellerType', 'SellerName', 'StreetName', 'State', 'ExteriorColor', 'InteriorColor', 'Drivetrain', 'FuelType', 'Transmission', 'Engine', 'VIN', 'Stock#'],axis=1,inplace=True)
new_data.replace({'DealType': DealTypeMapping, 'Price':PriceMapping}, inplace=True)
new_data.reset_index(inplace=True)
print(new_data.head)

new_data.to_csv(r'C:\\Zaid\\Fun Projects -- Code\\ACM_Research_CarsData\\data\\cars_wihout_NA.csv', index=False)
NEW_TOTAL_ROWS, NEW_TOTAL_COLUMNS = new_data.shape 

new_set = new_data.drop('index', axis=1)
print(new_set)
train_set = int(round(NEW_TOTAL_ROWS * 0.80, -1))
training_set = new_set.head(train_set)
training_set.to_csv(r'C:\\Zaid\\Fun Projects -- Code\\ACM_Research_CarsData\\data\\training_set.csv', index=False)


test_set = int(round(NEW_TOTAL_ROWS * 0.20, -1))
testing_set = new_set.tail(test_set-1)
testing_set.to_csv(r'C:\\Zaid\\Fun Projects -- Code\\ACM_Research_CarsData\\data\\testing_set.csv', index=False)

