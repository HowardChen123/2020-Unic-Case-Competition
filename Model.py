import pandas as pd

df = pd.read_excel("Joined_data.xlsx")
df = df.iloc[:, :-1]
df = df.dropna()

# encode the customer response column
for i in range(len(df)):
    if df['Customer Response'].values[i] == 'Won':
        df['Customer Response'].values[i] = 1
    else:
        df['Customer Response'].values[i] = 0
        
df = df.rename(columns={'Contract  Status':'Contract Status'})    

# Cateogrical Encoding --------------------------------------------------
import category_encoders as ce
cat_features = ['Contract Status']
target_enc = ce.TargetEncoder(cols=cat_features)

features = ['Contract Status', 'Product Family',
            'FTS Rate', 'Forecast', 'Market Share', 'Number of Competitors', 
            'WAC Price']

y = df['Customer Response'].astype('int64')
X = df[features]

from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

# Fit the encoder using the categorical features and target
target_enc.fit(train_X[cat_features], train_y)

# Transform the features, rename the columns with _target suffix, and join to dataframe
train_X = train_X.join(target_enc.transform(train_X[cat_features]).add_suffix('_target'))
val_X = val_X.join(target_enc.transform(val_X[cat_features]).add_suffix('_target'))

# drop the original columns that are encoded
train_X = train_X.drop(columns=['Contract Status'])
val_X = val_X.drop(columns=['Contract Status'])

train_X = train_X.rename(columns={'Contract Status_target':'Contract Status'})
val_X = val_X.rename(columns={'Contract Status_target':'Contract Status'}) 

train_X[features] = train_X[features].astype('float')
val_X[features] = val_X[features].astype('float')

# -----------------------------------------------------------------------
import lightgbm as lgb
from sklearn import metrics
dtrain = lgb.Dataset(train_X, label=train_y)
dvalid = lgb.Dataset(val_X, label=val_y)

param = {'num_leaves': 32, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}

bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                    early_stopping_rounds=30)

valid_pred = bst.predict(val_X)
valid_score = metrics.roc_auc_score(val_y, valid_pred)
print(f"Validation AUC score: {valid_score:.4f}")

import matplotlib.pyplot as plt

from lightgbm import plot_importance
from lightgbm import plot_split_value_histogram
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(bst, ax = ax)
fig, ax = plt.subplots(figsize=(10, 8))
plot_split_value_histogram(bst, 'Forecast', ax = ax)
plt.show()

ax = lgb.plot_tree(bst, tree_index=3, figsize=(200, 200), show_info=['split_gain'])


"""
--------------------------------------------------------------------------
--------------------------------------------------------------------------
--------------------------------------------------------------------------
"""
# Fitting classifier to the Training set
# Create your classifier here

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'sag', multi_class = 'multinomial',
                                random_state = 0, max_iter=100)
classifier.fit(train_X, train_y)
# Predicting the Test set results
y_pred = classifier.predict(val_X)

# print the training scores
print("training score : %.3f (%s)" % (classifier.score(train_X, train_y), 'multinomial'))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(val_y, y_pred)

-	probability:logistic_model.predict_proba(data)

from scipy.special import expit

sigmoid_function = expit(train_X['Forecast'] * classifier.coef_[0][0]+
                         classifier.intercept_[0]).ravel()

plt.plot(train_X['Forecast'], sigmoid_function)
plt.scatter(train_X['Forecast'], train_y, c=train_y, cmap='rainbow', edgecolors='b')

"""
--------------------------------------------------------------------------
--------------------------------------------------------------------------
--------------------------------------------------------------------------
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
val_X = sc.transform(val_X)

# Build ANN
from keras.models import Sequential
from keras.layers import Dense

# initialize the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(12, input_dim=8, activation='relu'))
classifier.add(Dense(8, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fitting the ANN
classifier.fit(train_X, train_y, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(val_X)

scores = classifier.evaluate(val_X, val_y)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))

from ann_visualizer.visualize import ann_viz

ann_viz(classifier, title="Bid win rate")












































