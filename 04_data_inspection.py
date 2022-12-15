#Since we don't know the Data characteristics we use .info() to get a general view of our data.
data.info()
miss_col = [var for var in data.columns if data[var].isnull().mean()>0]
data.head()
data.info()
data.head()
data.head()
data['Amount'].describe()
data.info()
data.info()
data.info()
data.info()
data['Delay'].head(10)
