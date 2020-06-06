import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
import numpy as np
import matplotlib.pyplot as plt

#               -----GAME-----
#1 Any live cell with fewer than two live neighbours dies, as if by underpopulation.
#2 Any live cell with two or three live neighbours lives on to the next generation.
#3 Any live cell with more than three live neighbours dies, as if by overpopulation.
#4 Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

#----------------------------------------------
#    [i-1,j-1]   |    [i-1,j]    |    [i-1,j+1]
# -------------------------------------
#    [i,j-1]     |    [i,j]      |    [i,j+1]
# ------------------------------------
#    [i+1,j-1]   |    [i+1,j]    |    [i+1,j+1]
# ---------------------------------------------

################################################################

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_pred = pd.concat([test['delta'], test.loc[:,'stop.1':'stop.400']], axis=1, sort=False).values

X =  pd.concat([train['delta'], train.loc[:,'stop.1':'stop.400']], axis=1, sort=False).values
y = train.loc[:,'start.1':'start.400'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)



# Predicting a new result
y_pred = regressor.predict(X_test[1,:].reshape(1,-1))
y_pred = regressor.predict(X_pred)

y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0


index_names = train.loc[:,'start.1':'start.400']
index_names = index_names.columns

pred_dic = dict(zip(index_names,y_pred))
id = test['id']
df=pd.DataFrame(data=y_pred[0:,0:], columns=[i for i in index_names])
df = pd.concat([id, df], axis=1, sort=False)
df.to_csv('sub.csv',index = False)




# --- plot the states -----
col_num = 1232
start = train.ix[col_num,'start.1':'start.400'].values.reshape(20,20)
stop = train.ix[col_num,'stop.1':'stop.400'].values.reshape(20,20)

start = X_test[1,1:].reshape(20,20)
stop = y_pred.reshape(20,20)

plt.figure(1)
for i in range(0,21):
    plt.plot([i,i],[0,20],'k')
    plt.plot([0, 20], [i, i], 'k')
for i in range(0,20):
    for j in range(0, 20):
        if start[i,j] == 0:
            plt.plot(j+0.5,i+0.5,'or') #0
         else:
            plt.plot(j+0.5,i+0.5,'^g') #1
plt.title('Final state')
plt.figure(2)
for i in range(0,21):
    plt.plot([i,i],[0,20],'k')
    plt.plot([0, 20], [i, i], 'k')
for i in range(0,20):
    for j in range(0, 20):
        if stop[i,j] == 0:
            plt.plot(j+0.5,i+0.5,'or') #0
         else:
            plt.plot(j+0.5,i+0.5,'^g') #1
plt.title('Initial (predicted) state')

plt.show() ################


sns.relplot(x="start.1", y="start.2", data=train)

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg

(ggplot(mpg)         # defining what data to use
 + aes(x='class')    # defining what variable to use
 + geom_bar(size=20) # defining the type of plot to use
)

(ggplot(mpg)
 + aes(x='displ', y='hwy', color='class')
 + geom_point()
 + labs(title='Engine Displacement vs. Highway Miles per Gallon', x='Engine Displacement, in Litres', y='Highway Miles per Gallon')
)