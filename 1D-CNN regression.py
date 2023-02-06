### This code is copied from Google Colab on 2023/02//06 for backup in github.
### One 1D-CNN regression model.

######################################################
#
# Install, Import, Read Files
#
######################################################
# !pip install spectral
# !pip install keras
# !pip install sklearn

from google.colab import drive
# from spectral import imshow, view_cube
# from spectral.algorithms import spatial
# import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import math
import keras

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from sklearn.model_selection import *
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from scipy.spatial import distance
from spectral import imshow, view_cube

# To save model
import pickle
import matplotlib.pyplot as plt
from scipy import stats

drive.mount('/content/gdrive')

import scipy.io
mat_file = '/content/gdrive/MyDrive/Colab Data/Data for CNN Regression/WP_DATA_9DAYS-ver2.mat'
mat = scipy.io.loadmat(mat_file)

model_storage_path = '/content/gdrive/MyDrive/Colab Data/1D-CNN (Models & Results)/Simple Data/Regression Models/'
line_graphs_storage_path = '/content/gdrive/MyDrive/Colab Data/1D-CNN (Models & Results)/Simple Data/Regression Line Graphs/'
bar_graphs_storage_path = '/content/gdrive/MyDrive/Colab Data/1D-CNN (Models & Results)/Simple Data/Regression Bar Graphs/'
images_storage_path = '/content/gdrive/MyDrive/Colab Data/1D-CNN (Models & Results)/Simple Data/Regression Images/'

x_np = mat['spectra']
y_np = mat['WP']

# x_mean = np.mean(x_np, axis=0)
# x_std = np.std(x_np, axis=0)
# x_np = (x_np - x_mean)/x_std
total_data = np.hstack((x_np, y_np))


######################################################
#
# Generating Train and Test Data
#
######################################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def generating_train_test_data():
  # scaler = MinMaxScaler()
  # norm_total_data = scaler.fit_transform(total_data)
  # norm_total_data[norm_total_data[:, 111] == 1] = 0.999
  # norm_total_data[:, :111] = total_data[:, :111] # uncomment for SNV
  norm_total_data = total_data
  np.random.shuffle(norm_total_data)
    
  X_train, X_test, y_train, y_test = train_test_split(norm_total_data[:, 0:111], norm_total_data[:, 111], test_size=0.3)
  unique, counts = np.unique(y_train, return_counts=True)
  return X_train, X_test, y_train, y_test, unique, counts

X_train, X_test, y_train, y_test, unique, counts = generating_train_test_data()


######################################################
#
# Model Training
#
######################################################
def build_conv1D_model():
  model = keras.Sequential(name="model_conv1D")
  model.add(keras.layers.Input(shape=(X_train.shape[1],1)))

  model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', name="Conv1D_1"))
  model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_1"))
  model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
  model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_2"))
  model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', name="Conv1D_3"))
  model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2, name="MaxPooling1D_3"))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(512, activation='relu', name="Dense_1"))
  model.add(keras.layers.Dense(128, activation='relu', name="Dense_2"))
  model.add(keras.layers.Dense(1, activation='linear', name="Dense_3"))
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
  history = model.fit(X_train, y_train, batch_size=256, epochs=100, validation_split=0.3) 

  # Plot
  # summarize history for accuracy
  plt.plot(history.history['mae'])
  plt.plot(history.history['val_mae'])
  plt.title('Regression Model Mean Absolute Error (MAE)')
  plt.ylabel('MAE')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(model_storage_path + 'Train Test Graphs/' + 'Regression MAE' + '.png')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Regression Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(model_storage_path + 'Train Test Graphs/' + 'Regression Loss' + '.png')
  plt.show()

  # Save Model
  filename = model_storage_path + "_regression_100.pkl" # type day epoch

  with open(filename, 'wb') as file:  
    pickle.dump(model, file)


  return model

### Model Training
model_conv1D = build_conv1D_model()
history = model_conv1D.fit(X_train, y_train, batch_size=256, epochs=200, validation_split=0.3, verbose=False) 
model_conv1D.evaluate(X_test, y_test)

def plot_history(history, y_axis):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel(y_axis)
  plt.plot(history.epoch, np.array(history.history[y_axis]), 
           label='Train')
  plt.plot(history.epoch, np.array(history.history[f'val_{y_axis}']),
           label = 'Val')
  plt.grid()
  plt.legend()
  plt.ylim([0,max(history.history[f'val_{y_axis}'])])
  
plot_history(history, 'mae')
plot_history(history, 'loss')


######################################################
#
# Erroorbar Graph
#
######################################################
pred_mean_wp = []

for group in range(groups): # 0 is sd, 1 is ww
  day_mean_wp = []  
  for day in range(days):
    plant_mean_wp = []  
    for plant in range(plants):
      mean_wp = np.mean(pred_wps[group][day][plant])
      plant_mean_wp.append(mean_wp)
    day_mean_wp.append(plant_mean_wp)
  pred_mean_wp.append(day_mean_wp)

sd_mean = np.mean(pred_mean_wp[0], axis= 1)
ww_mean = np.mean(pred_mean_wp[1], axis= 1)

sd_err = []
ww_err = []

for group in range(groups):
  for index, wp in enumerate(pred_mean_wp[group]):
    if (group == 0):
      temp_mean = sd_mean[index]
    else:
      temp_mean = ww_mean[index]

    # MinMax for errorbar
    # max_wp = np.max(wp) - temp_mean
    # min_wp = temp_mean - np.min(wp)
    # err_wp = [min_wp, max_wp]

    # Std for errorbar
    std_err = np.std(wp)
    err_wp = [std_err, std_err]
    if (group == 0):
      sd_err.append(err_wp)
    else:
      ww_err.append(err_wp)

for p in range(9):
  mean_sd = np.mean(pred_mean_wp[0][p])
  mean_ww = np.mean(pred_mean_wp[1][p])
  p_val = stats.ttest_ind(a=pred_mean_wp[1][p], b=pred_mean_wp[0][p], equal_var=True)
  print('Mean healthy WP on day ' + str(p + 1) + ' : ' + str(mean_ww))
  print('Mean drought stressed WP on day ' + str(p + 1) + ' : ' + str(mean_sd))
  print('WP p-value on day ' + str(p + 1) + ' : ' + str(p_val))

plt.title('Water Potential Prediction Model')
plt.xlabel('Day')
plt.ylabel('Mean Water Potential for All Plants')
plt.errorbar(np.linspace(1, 9, 9), ww_mean, yerr = np.transpose(ww_err), color = 'blue', label = 'well-watered',  fmt='-o')
plt.errorbar(np.linspace(1, 9, 9), sd_mean, yerr = np.transpose(sd_err), color = 'red', label = 'drought',  fmt='-o')
plt.legend()
plt.savefig(line_graphs_storage_path + 'water_potential_prediction.png')

p_values = []
for tmp_day in range(9):
  p_val = stats.ttest_ind(a=pred_mean_wp[0][tmp_day], b=pred_mean_wp[1][tmp_day], equal_var=True)
  # p_val = stats.f_oneway(pred_mean_wp[0][tmp_day], pred_mean_wp[1][tmp_day])
  print('Water Potential Prediction Model Day: ' + str(tmp_day + 1) + ' P-value: ' + str(p_val))
  p_values.append(p_val)


######################################################
#
# Single Plant Histogram
#
######################################################
bin_list = np.linspace(-1.5, 0.5, 100)

# for day in range(days):
#   hist_data = pred_wps[1][day][0]
#   plt.hist(hist_data, bins =bin_list, weights=np.ones(len(hist_data)) / len(hist_data))
#   plt.title('Day ' + str(day+1))
#   plt.show()
for plant_pot in range(10):
  early = pred_wps[0][3][plant_pot]
  late = pred_wps[0][8][plant_pot]
  e_mean = np.mean(early)
  l_mean = np.mean(late)
  p_val = stats.ttest_ind(a=early, b=late, equal_var=True)
  print(str(e_mean) + ' : ' + str(l_mean))
  print(p_val)
  plt.hist(early, bins =bin_list, weights=np.ones(len(early)) / len(early),  alpha=0.5, label="data1")
  plt.hist(late, bins =bin_list, weights=np.ones(len(late)) / len(late),  alpha=0.5, label="data2")
  plt.show()

import matplotlib.pyplot as pyplot
mask = [mat['masksd'], mat['maskww']]
# group day plant

pot_no = 7
# SD 7, 8
for g in range(2):
  for mask_day in range(9):
    pred_index = 0
    tmp_pred = pred_wps[g][mask_day][pot_no].flatten()
    tmp_pred = (tmp_pred - tmp_pred.min()) / (tmp_pred.max() - tmp_pred.min())
    tmp_mask = mask[g][mask_day][pot_no].flatten()[0].flatten()

    for t_mask in range(len(tmp_mask)):
      if (tmp_mask[t_mask] == 0):
        continue
      else:
        t_pred = tmp_pred[pred_index]
        tmp_mask[t_mask] = t_pred * 255
        pred_index += 1
        
    tmp_mask.resize((328,510), refcheck=False)

    pyplot.imshow(tmp_mask, cmap='hot')
    tmp_g = 'PD' if g == 0 else 'WW'
    pyplot.title('Group ' + tmp_g + ' Day ' + str(mask_day + 1))
    plt.savefig(images_storage_path + 'Group ' + tmp_g + ' Pot ' + str(pot_no) + ' Day ' + str(mask_day + 1) + '.png')
    pyplot.show()

######################################################
#
# Average Histogram
#
######################################################
# this one compares between treatment groups for all 9 days
bin_list = np.linspace(-2, 0, 100)
p_values = []

import pandas as pd
for plant_day in range(9):
  early = pred_wps[0][plant_day] # 0 is sd
  late = pred_wps[1][plant_day] # 1 is ww
  early = [pix for sub_plant in early for pix in sub_plant]
  late = [pix for sub_plant in late for pix in sub_plant]
  e_mean = np.mean(early)
  l_mean = np.mean(late)

  #####
  hist_sd = np.histogram(early, bins=bin_list, weights=np.ones(len(early)) / len(early))
  hist_ww = np.histogram(late, bins=bin_list,  weights=np.ones(len(late)) / len(late))
  
  mid_sd = [ (hist_sd[1][a] + hist_sd[1][a+1])/2 for a in range(99)]
  mid_ww = [ (hist_ww[1][a] + hist_ww[1][a+1])/2 for a in range(99)]
  tmp_sd = np.multiply(hist_sd[0], mid_sd)
  tmp_ww = np.multiply(hist_ww[0], mid_ww)
  mn_sd = np.mean(tmp_sd)
  mn_ww = np.mean(tmp_ww)
  p_val = stats.ttest_ind(a=tmp_sd, b=tmp_ww, equal_var=True)
  p_values.append({'mean_sd': mn_sd, 'mean_ww': mn_ww, 'p_val': p_val})
  # print('mean WP for drought plants on day ' + str(plant_day + 1) + ' : ' + str(e_mean))
  # print('mean WP for well-watered plants on day ' + str(plant_day + 1) + ' : '  + str(l_mean))
  # print('mean WP p-value on day ' + str(plant_day + 1) + ' : ' + str(p_val))
  # print(' =================================================== ')

  ##### Plot Graph
  plt.hist(early, bins =bin_list, weights=np.ones(len(early)) / len(early),  alpha=0.5, label="drought stressed", color='red')
  plt.hist(late, bins =bin_list, weights=np.ones(len(late)) / len(late),  alpha=0.5, label="healthy", color='blue')

  plt.legend(loc = 'upper right')
  plt.xlabel('Water Potential Value')
  plt.ylabel('Water Potential Percentage')
  plt.title( 'Histogram of Water Potential Day ' + str(plant_day+1))
  plt.savefig(bar_graphs_storage_path + 'histogram_' + str(plant_day+1) + '.png')
  plt.show()

# for p in range(9):
#   print('Mean healthy WP on day ' + str(p + 1) + ' : ' + str(p_values[p]['mean_ww']))
#   print('Mean drought stressed WP on day ' + str(p + 1) + ' : ' + str(p_values[p]['mean_sd']))
#   print('WP p-value on day ' + str(p + 1) + ' : ' + str(p_values[p]['p_val']))