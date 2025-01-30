import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


data = pd.read_csv('dataset.csv')


print(data.columns)


data = pd.get_dummies(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'], drop_first=True)


x = data.drop('Play Tennis', axis=1)
y = data['Play Tennis']


dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(x, y)


tree_rules = export_text(dtree, feature_names=list(x.columns))
print("Decision tree rules:")
print(tree_rules)


new_sample = pd.DataFrame({
    'Outlook_overcast': [0], 
    'Outlook_rain': [1], 
    'Outlook_sunny': [0], 
    'Temperature_cool': [0], 
    'Temperature_hot': [1], 
    'Temperature_mild': [0], 
    'Humidity_high': [0], 
    'Humidity_normal': [1], 
    'Wind_True': [0]
})


new_sample = new_sample.reindex(columns=x.columns, fill_value=0)


prediction = dtree.predict(new_sample)
print("Prediction for the new sample:", prediction[0])

