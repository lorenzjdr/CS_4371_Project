import time
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import seaborn as sn
from sklearn.metrics import confusion_matrix
from tkinter import *
from tkinter import filedialog
import csv


# load dataframe
df = pd.read_csv('Dataset/environmentMonitoring.csv/environmentMonitoring.csv')
print(df.head())

# rename dataset columns to match what the model 
df = df.rename(columns={
    'frame.len': 'Length',
    'tcp.dstport': 'DestinationPort',
    'tcp.flags': 'FlagValue',
    'label': 'Label',
    'class': 'Device_name',
    'ip.src': 'SourceIP',
    'ip.dst': 'DestinationIP',
})

# Drop columns the model doesn't use (same as original) ---
df = df.drop(columns=['SourceIP', 'DestinationIP', 'Device_name'], errors='ignore')
df = df.dropna()

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {'0x0010': 10, '0x0018': 18, '0x0012': 12, '0x0014': 14, '0x0011': 11, '0x0002': 2}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            #print('unique:', unique_elements)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                print(x)
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
#     #print(df.head())
#     start_time = time.time()

#     #Model Train
#     data = pd.read_csv('train.csv')
#     data = shuffle(data)
#     #X = df[['Length']]
#     y = df['Label']

# used the features the researchers used
X = df[['DestinationPort', 'Length', 'FlagValue']]
y = df['FlagValue'] # This needs to be fixed; the new data only has 0 as labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

model = RandomForestClassifier()
trained_model = model.fit(X_train, y_train)

# --- Save the model ---
model_path = 'RandomForest_environmentMonitoring_profile.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(trained_model, f)

print(f"Model saved to {model_path}")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# models = {}

# #     # Random Forest
# models['Random Forest'] = RandomForestClassifier()

# # accuracy, precision, recall = {}, {}, {}
# # for key in models.keys():
# #     print(key)
# #     # Fit the classifier model
# #     if os.path.exists('./' + key.replace(" ", '') + str(i) +'_' + 'profile' + '.pkl'):
# #         os.remove('./' + key.replace(" ", '') + str(i) + '_' + 'profile' + '.pkl')
# #     trained_model = models[key].fit(X_train, y_train)
# #     with open(key.replace(" ", '') + str(i) + '_' + 'profile' + '.pkl', 'wb') as f:
# #         pickle.dump(trained_model, f)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#clf = RandomForestClassifier(n_estimators=10)

'''from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=100)

# Support Vector Machines
from sklearn.svm import LinearSVC
clf = LinearSVC()

# Decision Trees
clf = DecisionTreeClassifier()'''

#clf = clf.fit(X_train,y_train)

#Accuracy Score Calculation
'''#print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on testing set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#confidence score
y_predict = (clf.predict_proba(X_test))
with open('confidence.txt', 'w') as a:
    a.write(str(y_predict))

    #print ('confidence score:', (y_predict))

#Confusion Matrix
y_pred = clf.predict(X_test)
confusion_matrix =confusion_matrix(y_test, y_pred)
print("---% seconds ---" % (time.time() -start_time))
plt.figure(figsize=(5,5))
sn.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()'''

