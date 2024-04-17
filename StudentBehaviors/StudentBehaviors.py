import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('overdrawn.csv')

# convert categorical data to int representations of unique categories
for col in df.columns:
    labels, uniques = pd.factorize(df[col])
    df[col] = labels
    
X = df.drop(columns='Overdrawn')
y = df['Overdrawn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Change DaysDrink into categorical data
conditions = [
    (df['DaysDrink'] < 7), #catergory 1
    (df['DaysDrink'] >= 14), #catergory 2
    (df['DaysDrink'] >= 7) & (df['DaysDrink'] < 14) #catergory 3
]

categories = [0, 2, 1]

# Apply the coniditions to create the categorical data
df['DaysDrink'] = np.select(conditions, categories)

query1 = [20, 0, 10], [25, 1, 5], [19, 0, 20], [22, 1, 15], [21, 0, 20]
#query1 = [20, 0, 1], [25, 1, 0], [19, 0, 2], [22, 1, 2], [21, 0, 2]
predict1 = dtree.predict_proba(query1 )
num = 1
for x in predict1:
    if(x[0] > 0.5):
        print('Prediction ' + str(num) + ': Will the student overdraw a checking account? Yes')
    else:
        print('Prediction ' + str(num) + ': Will the student overdraw a checking account? No')
    num+=1

import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=('Age', 'Sex', 'DayDrink'),
                                class_names=('0','1'),
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render('overdrawn_dt', view=True)