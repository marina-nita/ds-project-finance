#importing important Libraries


#installing the ptitprince so we can use Rain Cloud to understand our data better


#reading the CSV by using the path (we worked on Google Colab because it was easier for us as a team to work in the same notebook)
path = "/content/drive/MyDrive/DS Project/Invoice_dataset.csv"

#Data frame for finding the wrong dates

data

data


#Droping the Column UniqieId since it is not important in our Case and dosn't play a role in what we are going to do 

data

#Finding Columns that have missing Data
print(miss_col)

#Calculating how much of Data is missing for each Column in percent.
print((data['Region'].isnull().mean())*100)
print((data['Amount'].isnull().mean())*100)
print((data['Disputed'].isnull().mean())*100)
print((data['InvoiceType'].isnull().mean())*100)

#Using the Histrogram Vizualization to understand the distribution of the Amount Variable(column)

#Using Vizualization to understand better value count of Region (it's frequence of values)

#Using Vizualization to understand better value count of Disputed

#Using Vizualization to understand better value count of Invoice Type

##Using Vizualization to understand better value count of Delay


#transformig the second df to time value so i can find the wrong values of date
df3['OrderDate'] = pd.to_datetime(df3['OrderDate'],'coerce')
df3['InvoiceDate'] = pd.to_datetime(df3['InvoiceDate'],'coerce')
df3['DueDate'] = pd.to_datetime(df3['DueDate'],'coerce')

#finding the null values of dates after transforming
df3[df3['OrderDate'].isna() == True]
df3[df3['InvoiceDate'].isna() == True]
df3[df3['DueDate'].isna() == True]


#visualizing the wrong dates
data.iloc[[26,129,908,1017,1338,1376]]
data.iloc[[474,635,776,1006,1089,1189,1286,1683]]
data.iloc[[227,482,794,1317,1563,1705]]

#replacing the wrong dates in first dataframe


data["InvoiceDate"] =  pd.to_datetime(data["InvoiceDate"], format="%d/%m/%Y")
data["DueDate"] =  pd.to_datetime(data["DueDate"], format="%d/%m/%Y")
data["OrderDate"] =  pd.to_datetime(data["OrderDate"], format="%d/%m/%Y")

# Filtering Dates outside 2011 and 2013
data = data.loc[(data['InvoiceDate'] >= '2011-01-01') & (data['InvoiceDate'] < '2014-01-01')]

#making new columns with the diffrence betwend date coolumns
data['InvoiceOrderDiff'] = ((data.InvoiceDate - data.OrderDate)/np.timedelta64(1, 'D'))
data['DueInvoiceDiff'] = ((data.DueDate - data.InvoiceDate)/np.timedelta64(1, 'D'))
data['DueOrderDiff'] = ((data.DueDate - data.OrderDate)/np.timedelta64(1, 'D'))

#making new columns with the day/month/year of the 3 date columns
data['OrderDay'] = data['OrderDate'].apply(lambda time: time.day)
data['OrderMonth'] = data['OrderDate'].apply(lambda time: time.month)
data['OrderYear'] = data['OrderDate'].apply(lambda time: time.year)
data['InvoiceDay'] = data['InvoiceDate'].apply(lambda time: time.day)
data['InvoiceMonth'] = data['InvoiceDate'].apply(lambda time: time.month)
data['InvoiceYear'] = data['InvoiceDate'].apply(lambda time: time.year)
data['DueDay'] = data['DueDate'].apply(lambda time: time.day)
data['DueMonth'] = data['DueDate'].apply(lambda time: time.month)
data['DueYear'] = data['DueDate'].apply(lambda time: time.year)

#Describing the Numerical Column (Amount) so we can understand it's characteristics 

#Simple BoxPlot Vizualization so we can see if Amount has Outlier(External Values)

#Using Viz. RainCloud so we can understand the Distribution of Amount(it's mean and where do most of values fall) and where do the Outliers fall before Handeling the them.
                  data = data,
                  width_viol = 0.8,
                  orient = 'h',
                  move = 0)

# filtering data, so taking the Amount without NaN values so we can Handle(Droping) the Outliers (since there are low numbers of Rows) 
# displaying data only with Amount = NaN
bool_series = pd.isnull(data["Amount"])
data[bool_series]


#checking if this new dataframe has NaN values
doseithavemissingvalues = pd.isnull(missingval_Amount['Amount'])
missingval_Amount[doseithavemissingvalues]

arrayofAmount = missingval_Amount.to_numpy()

#Creating an empty List and appending the values of Amount without the NaN values
arrayAmount = []

for i in arrayofAmount:
      x = np.take(i,indices = 4)
      arrayAmount.append(x)
print(arrayAmount)

#Removing Outliers 
listt1 = []
for i in arrayAmount:
  if  i > 115:
    listt1.append(i)
  if i < 6:
    listt1.append(i)
print(listt1)

#Droping the Values by using the list
for i in listt1:


#RainCloud after Handeling the Outliers of Amount
data
                  data = data,
                  width_viol = 0.8,
                  orient = 'h',
                  move = 0)

##Handeling Region Missing Values and Typing Problems (through AVG of Amount of each Region we can fill the Missing values of Amount)

#Vizualizing Region 
data['Region'].value_counts()


# Cleaning the data of typing Errors (handeling tping errors)

# filtering data, so taking the Region without NaN values so we can Handle Missing Values
bool_series = pd.isnull(data["Region"])
data[bool_series]

# Length of missing Regions
len(data[bool_series])

#Converting the DataFrame above into an array(dictionary) with only the values of Amount
arr = data[bool_series].to_numpy()
#print(arr)

#amarr = np.take(arr,indices = 4)
amarr = []

for i in arr:
      x = np.take(i,indices = 4)
      amarr.append(x)
print(amarr)
#print(len(data[bool_series]))

#Taking only the rows with Region ---> North America
NA1 = data[data['Region'] == 'North America']

NA1

amountNA = NA1['Amount'].to_numpy()

amountNA

#Average of Amount in North America
NorthAmerica_AVG_Value = round(np.mean(amountNA), 2)
print(NorthAmerica_AVG_Value)

SA = data[data['Region'] == 'South America']

amountSA = SA['Amount'].to_numpy()

# AVG of Amount in South America rounded
SouthAmerica_AVG_Value = round(np.mean(amountSA), 2)
print(SouthAmerica_AVG_Value)

AA = data[data['Region'] == 'Africa&Asia']

amountAA = AA['Amount'].to_numpy()

AfricaAsia_AVG_Value = round(np.mean(amountAA), 2)
print(AfricaAsia_AVG_Value)

Eu = data[data['Region'] == 'Europe']

amountEu = Eu['Amount'].to_numpy()

Europe_AVG_Value = round(np.mean(amountEu), 2)
print(Europe_AVG_Value)

CA = data[data['Region'] == 'Central America']

amountCA = CA['Amount'].to_numpy()

CAmerica_AVG_Value = round(np.mean(amountCA), 2)
print(CAmerica_AVG_Value)

# After we calculated the AVG for each Region we use it as a way to fill the NaN Regions using Amount values(because the NaN Regions have Amount values) (the filling is approximetly)
missingReg = []
for i in amarr:
  if i <= (CAmerica_AVG_Value + 5):
    missingReg.append('Central America') 

  elif i <= (SouthAmerica_AVG_Value + 5): 
    missingReg.append('South America')

  elif i <= (AfricaAsia_AVG_Value + 5):
    missingReg.append('Africa&Asia')

  elif i <= (NorthAmerica_AVG_Value + 5): 
    missingReg.append('North America')

  else:
    missingReg.append('Europe')

print(missingReg)

len(missingReg)

arrayofRegion = np.array(missingReg)
data.loc[data['Region'].isnull(), 'Region'] = arrayofRegion

data

# After the changes we see the General Information about data


# filtering data, so taking the Amount with NaN values so we can Handle Missing Values
missing_rows_Amount = pd.isnull(data["Amount"])
data[missing_rows_Amount]

len(data[missing_rows_Amount])

# Transforming the Data Frame into an array 
array_of_MissingAmount_Values = data[missing_rows_Amount].to_numpy()
#print(array_of_MissingAmount_Values)

array_Region_MissingAmount_Values = []

for i in array_of_MissingAmount_Values:
      z = np.take(i,indices = 0)
      array_Region_MissingAmount_Values.append(z)
print(array_Region_MissingAmount_Values)
#print(len(data[bool_series]))

#Filling the Missing values in Amount with the AVG of the Regions
missingAmount = []
for i in array_Region_MissingAmount_Values: 
  if i == 'Central America':
    missingAmount.append(CAmerica_AVG_Value)
  elif i == 'South America': 
    missingAmount.append(SouthAmerica_AVG_Value)
  elif i == 'Arica&Asia':
    missingAmount.append(AfricaAsia_AVG_Value)
  elif i == 'North America': 
    missingAmount.append(NorthAmerica_AVG_Value)
  else:
    missingAmount.append(Europe_AVG_Value)

print(missingAmount)

# Fillint the missing values of Amount with the AGV in data 
arrayofAmount = np.array(missingAmount)
data.loc[data['Amount'].isnull(), 'Amount'] = arrayofAmount












#Visualization for the Disputed feature in every region
CrosstabResult=pd.crosstab(index=data['Region'],columns=data['Disputed'])
print(CrosstabResult)

missing = pd.isnull(data['Disputed'])

#Convert delay and disputed in 0 and 1

#fill the Disputed missing values

#check if the values are filled
data['Disputed'].isnull().sum()


#Visualization for the InvoiceType feature in every region
CrosstabResult=pd.crosstab(index=data['Region'],columns=data['InvoiceType'])
print(CrosstabResult)

missing1 = pd.isnull(data['InvoiceType'])
data[missing1]

data_noMissingInvoice['InvoiceType'].isnull().sum()

#Group by Region
grouped1 = data_noMissingInvoice.groupby('Region')

# Calculate the mode of the 'InvoiceType' column for each group
modes1 = grouped1['InvoiceType'].apply(lambda x: x.mode())
modes1

#sorted regions vector use for the next part
regions=data['Region'].unique()
regions=sorted(regions)
regions

for i in range (0,len(regions)):      

#convert categorical InvoiceType into numerical

#check if there are still different rows in the dataframe

#transforming into One Hot Encoding the Region coulmn since we have categorical values  
encoded_columns = pd.get_dummies(data.Region)

data


data['Delay'].value_counts()


#transforming the Delay column in 0 , 1 sice its a categorical value and we only have true and False

data


data


#Set the target column and split into test and train
y = data['Delay']


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Spot Check Algorithms to see the differences between them
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms

# Make predictions for a Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

#display the confussion matrix
conf_matrx = confusion_matrix(y_test, predictions)
conf_matrx

#visualize the confusion matrix


#number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5, stop = 20, num = 10)]
#number of features to consider at every split
max_features = ['auto', 'sqrt']
#maximum number of leaves in the tree
max_depth = [5, 10]
#minimum number of samples required to split a node
min_samples_split = [2,3,4]
#minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
#Method of selecting samples for training each tree
bootstrap = [True, False]

#Create the parameter grid
param_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf,
              'bootstrap' : bootstrap}
print(param_grid)

#train the model without parameters
rf_Model = RandomForestClassifier()

#search through the parameter grid randomly; it is faster than grid searching
rf_RandomGrid = RandomizedSearchCV(estimator = rf_Model, param_distributions= param_grid, cv = 10, verbose = 2, n_jobs = 4)

#train the model using hyperparameter tuning
rf_model=rf_RandomGrid.fit(X_train, y_train)

#best paramters
rf_RandomGrid.best_params_

print('Train Accuracy : ', rf_RandomGrid.score(X_train, y_train))
print('Test Accuracy : ', rf_RandomGrid.score(X_test, y_test))

#predictions
predictions_rf = rf_model.predict(X_test)

# Evaluate predictions with F1-score
print(accuracy_score(y_test, predictions_rf))
print(confusion_matrix(y_test, predictions_rf))
print(classification_report(y_test, predictions_rf))

# Create the parameter grid
param_grid_svm = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}

# Using GridSearch as an alternate method for this model
svm_grid = GridSearchCV(SVC(), param_grid_svm, refit = True, verbose = 3)

# Train the model using hyperparameter tuning
svm_model = svm_grid.fit(X_train, y_train)

# Best parameters
svm_grid.best_params_

predictions_SVC = svm_model.predict(X_test)

# Evaluate predictions
print(accuracy_score(y_test, predictions_SVC))
print(confusion_matrix(y_test, predictions_SVC))
print(classification_report(y_test, predictions_SVC))

# Obtained Accuracy
print('Train Accuracy : ', svm_grid.score(X_train, y_train))
print('Test Accuracy : ', svm_grid.score(X_test, y_test))