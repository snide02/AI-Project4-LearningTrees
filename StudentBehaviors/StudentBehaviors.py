import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

bal_df = pd.read_csv('overdrawn.csv')

# convert categorical data to int representations of unique categories
for col in bal_df.columns:
    labels, uniques = pd.factorize(bal_df[col])
    bal_df[col] = labels
    
X = bal_df.drop(columns='Overdrawn')
y = bal_df['Overdrawn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=('Age', 'Sex', 'DayDrink'),
                                class_names=('0','1'),
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render('overdrawn_dt', view=True)