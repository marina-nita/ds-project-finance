import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.collections as clt
import math
import datetime
from sklearn import preprocessing
import ptitprince as pt
from google.colab import drive
#transforming the dates from string to datetype in the first dataframe
#getting from this dataframe only the values of Amount as an Array 
#droping from this Data Frame the Missing Values of Amount so we can Calculate the AVG of Region using Values of Amount (in North America)
#droping from this Data Frame the Missing Values of Amount so we can Calculate the AVG of Region using Values of Amount (in South America)
# Taking Region valkues from rows that have missing values in Amount
#fill each missing InvoiceType from each region with the corresponding mode
from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
