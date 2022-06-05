import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('evaluate_data.csv')

X, y = df[['age']], df['income_if_<=50k']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
n_samples, n_features = X.shape
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
torch.from_file
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test= y_test.view(y_test.shape[0], 1)
class LogisticRegresion(nn.Module):
    def __init__(self, n_input_featuers):
        super(LogisticRegresion, self).__init__()
        self.linear = nn.Linear(n_input_featuers, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegresion(n_features)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1500
for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
result = open("result_pytorch.txt",'w')
result.write(f'acc:{acc:.4f}')