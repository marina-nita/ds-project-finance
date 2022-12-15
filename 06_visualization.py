data.hist()
data['Region'].value_counts().plot.bar()
data['Disputed'].value_counts().plot.bar()
data['InvoiceType'].value_counts().plot.bar()
data['Delay'].value_counts().plot.bar()
boxplot = data.boxplot(column=['Amount'], return_type='axes')
type(boxplot)
plt.figure(figsize=(20,10))
ax = pt.RainCloud(y = 'Amount',
                  width_box = 0.4,
plt.title('Model Evaluation')
plt.xlabel('ampount')
plt.ylabel('distribution')
plt.show()
boxplot = data.boxplot(column=['Amount'], return_type='axes')
type(boxplot)
plt.figure(figsize=(20,10))
ax = pt.RainCloud(y = 'Amount',
                  width_box = 0.4,
plt.title('Model Evaluation')
plt.xlabel('ampount')
plt.ylabel('distribution')
plt.show()
data['Region'].value_counts().plot.bar()
data['Region'].value_counts().plot.bar()
# Create a catplot using the "Disputed" column as the x-axis and the "Delay" column as the hue
sns.catplot(x="Disputed", hue="Delay", data=data[['Disputed', 'Delay']], kind="count", height=4, aspect=1.5)
# Show the plot
plt.show()
# Create a boxplot for Disputed and Amount
sns.boxplot(x="Disputed", y="Amount", data=data[['Disputed', 'Amount']])
# Show the plot
plt.show()
# Create a boxplot for Disputed and InvoiceOrderDiff
sns.boxplot(x="Disputed", y="InvoiceOrderDiff", data=data[['Disputed', 'InvoiceOrderDiff']])
# Show the plot
plt.ylim(0,120)
plt.show()
# Create a boxplot for Disputed and DueInvoiceDiff
sns.boxplot(x="Disputed", y="DueInvoiceDiff", data=data[['Disputed', 'DueInvoiceDiff']])
# Show the plot
plt.show()
# Create a boxplot for Disputed and DueOrderDiff
sns.boxplot(x="Disputed", y="DueOrderDiff", data=data[['Disputed', 'DueOrderDiff']])
# Show the plot
plt.show()
CrosstabResult.plot.bar()
CrosstabResult.plot.bar()
plt.rcParams['figure.figsize'] = 20, 10
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
sns.heatmap(conf_matrx,annot = True)
plt.show()
