import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def save_data(X_train, X_test, y_train, y_test, name):
    X_train.to_csv('./splits/' + name + '_X_train.csv', index=False)
    X_test.to_csv('./splits/' + name + '_X_test.csv', index=False)
    y_train.to_csv('./splits/' + name + '_y_train.csv', header=True, index=False)
    y_test.to_csv('./splits/' + name + '_y_test.csv', header=True, index=False)


name = 'two_features'



rows = 7500
xs = 8 
np.random.seed(0)
alpha = .6

x1 = np.random.uniform(-1, 1, rows)
x2 = np.random.uniform(-1, 1, rows)
y = (np.logical_xor((np.sign(x1)+1), (np.sign(x2)+1)) * (np.abs(np.abs(x1 - x2 )-1)  < .1)).astype(int)
#note: in my paper, I expreess the math of y differently, but the two are equivilent.

print(np.mean(y))

df1 = pd.DataFrame({'y': y,
                    'x1': x1,
                    'x2': x2
                    })

#df1.to_csv('candidate_datasets/own/two_features.csv', index=False)
X_train, X_test, y_train, y_test = train_test_split(df1[['x1', 'x2']],
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)
save_data(X_train, X_test, y_train, y_test, name)

n_classes = np.unique(y).size
markers = 'xo'
colors = ['purple', 'yellow']
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
for k, label in enumerate(np.unique(y)):
    plot_mask = (y == label)
    plot_mask = plot_mask.astype(bool)
    plt.scatter(x1[plot_mask],
        x2[plot_mask],
        marker=markers[k],
        c=colors[k],  #'gray',
        edgecolor='k', alpha=alpha)
plt.xlabel('x1')
plt.ylabel('x2')
plt.suptitle('The Two Deterministic Features of the Synthetic Data')
plt.savefig('./plots/data.png') 


