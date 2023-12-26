import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

def read_data():
   path = r'./NIRS_dataset.csv'
   data1 = pd.read_csv(path, index_col='No')
   data = data1.iloc[:, 1:].values
   target = data1.iloc[:, 0].values

   return data, target
   

def Autocorrelation(data):
   data_out = data*data

   return data_out


def SG(data, window_length, polyorder):
   [m, n] = data.shape

   data_out = np.zeros((m, n))
   for i in range(m):
      data_out[i, :] = signal.savgol_filter(data[i, :], window_length, polyorder)

   return data_out


def diff_data(data, nstep):
   data_out = np.diff(data, nstep)

   return data_out

def Standardization(data):
   scaler = StandardScaler()
   scaler.fit(data)
   data_out = scaler.transform(data)

   return data_out

def pca_data(data, n_components):

   pca = PCA(n_components)
   reduced_x = pca.fit_transform(data)

   return reduced_x


def MSC(data):
   mean = np.mean(data, axis=0)

   [m, n] = data.shape
   msc_x = np.zeros((m, n))

   for i in range(m):
      y = data[i, :]
      lin = LinearRegression()
      lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
      k = lin.coef_
      b = lin.intercept_
      msc_x[i, :] = (y - b) / k
   return msc_x


def SNV(data):
   mean = np.mean(data, axis=1)
   std = np.std(data, axis=1)

   [m, n] = data.shape
   snv_x = np.zeros((m, n))

   for i in range(m):
      snv_x[i, :] = (data[i, :] - mean[i]) / std[i]

   return snv_x


if __name__=='__main__':
   data, target = read_data()

   data = Autocorrelation(data)
   data = Standardization(data)
   data = MSC(data)
   data = SNV(data)
   data = SG(data, window_length=5, polyorder=3)
   data = diff_data(data, nstep=1)
   data = pca_data(data, n_components=5)
   x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

   y_con = np.concatenate((y_train, y_test))
   label_encoder = preprocessing.LabelEncoder()
   y_encoded = label_encoder.fit_transform(y_con)
   y_train_encoded = label_encoder.transform(y_train)
   y_test_encoded = label_encoder.transform(y_test)

   # GaussianNB
   gnb = GaussianNB()
   # fit and predict
   y_pred = gnb.fit(x_train, y_train).predict(x_test)
   accuracy = accuracy_score(y_test, y_pred)
   print("GaussianNB: Classification Accuracy: %0.2f%%." % (accuracy * 100))
   
   # KNN
   knn = KNeighborsClassifier(algorithm='kd_tree')
   # fit and predict
   y_pred = knn.fit(x_train, y_train).predict(x_test)
   accuracy = accuracy_score(y_test, y_pred)
   print("KNN: Classification Accuracy: %0.2f%%." % (accuracy * 100))
   
   # CART
   clf = tree.DecisionTreeClassifier()  # criterion = ’gini’,'entropy'
   # fit and predict
   y_pred = clf.fit(x_train, y_train).predict(x_test)
   accuracy = accuracy_score(y_test, y_pred)
   print("CART: Classification Accuracy: %0.2f%%." % (accuracy * 100))
   
   # SVM
   svc = svm.SVC()
   # fit and predict
   y_pred = svc.fit(x_train, y_train).predict(x_test)
   accuracy = accuracy_score(y_test, y_pred)
   print("SVM: Classification Accuracy: %0.2f%%." % (accuracy * 100))

   # Linear Regression
   regressor = LinearRegression()
   y_pred = regressor.fit(x_train, y_train_encoded).predict(x_test)
   y_pred[y_pred < 0] = 0
   y_pred[y_pred > 4] = 4
   y_pred = np.round(y_pred)
   y_pred = y_pred.astype(int)
   accuracy = accuracy_score(y_test_encoded, y_pred)
   print("LR1: Classification Accuracy: %0.2f%%." % (accuracy * 100))

   # MLM
   mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=1)
   y_pred = mlp_classifier.fit(x_train, y_train).predict(x_test)
   accuracy = accuracy_score(y_test, y_pred)
   print("MLP: Classification Accuracy: %0.2f%%." % (accuracy * 100))

   # Backpropagation Neural Network
   mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=1)
   y_pred = mlp_regressor.fit(x_train, y_train_encoded).predict(x_test)
   # print(y_pred)
   y_pred[y_pred < 0] = 0
   y_pred[y_pred > 4] = 4
   y_pred = np.round(y_pred)
   y_pred = y_pred.astype(int)
   accuracy = accuracy_score(y_test_encoded, y_pred)
   print("BPNN: Classification Accuracy: %0.2f%%." % (accuracy * 100))

   # CNN
   X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
   X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
   y_train_encoded = to_categorical(y_train_encoded, num_classes=5)

   model = Sequential()
   model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
   model.add(MaxPooling1D(pool_size=2))
   model.add(Flatten())
   model.add(Dense(64, activation='relu'))
   model.add(Dense(5, activation='softmax'))
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

   y_pred_probs = model.predict(X_test)
   y_pred = np.argmax(y_pred_probs, axis=1)

   accuracy = accuracy_score(y_test_encoded, y_pred)
   print("CNN: Classification Accuracy: %0.2f%%." % (accuracy * 100))

   # DNN
   model = Sequential()
   model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
   model.add(Dense(64, activation='relu'))
   model.add(Dense(5, activation='softmax'))
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

   y_pred_probs = model.predict(X_test)
   y_pred = np.argmax(y_pred_probs, axis=1)

   accuracy = accuracy_score(y_test_encoded, y_pred)
   print("DNN: Classification Accuracy: %0.2f%%." % (accuracy * 100))
