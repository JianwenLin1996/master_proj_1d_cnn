### This code is copied from Google Colab on 2023/02//06 for backup in github.
### One 1D-CNN classification model for each day.

######################################################
#
# Install, Import, Read Files
#
######################################################
# !pip install spectral
# !pip install keras
# !pip install sklearn

from google.colab import drive
from sklearn.model_selection import *
# from spectral import imshow, view_cube
# from spectral.algorithms import spatial
# import spectral.io.envi as envi
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import math
import keras
drive.mount('/content/gdrive')

import scipy.io

# To save model
import pickle

mat_file = '/content/gdrive/MyDrive/Colab Data/Data for CNN Regression/WP_DATA_9DAYS.mat'
mat = scipy.io.loadmat(mat_file)

x_np = mat['spectra']
y_np = mat['WP']

total_data = np.hstack((x_np, y_np))

model_storage_path = '/content/gdrive/MyDrive/Colab Data/1D-CNN (Models & Results)/Simple Data/Models/'
line_graphs_storage_path = '/content/gdrive/MyDrive/Colab Data/1D-CNN (Models & Results)/Simple Data/Line Graphs/'
bar_graphs_storage_path = '/content/gdrive/MyDrive/Colab Data/1D-CNN (Models & Results)/Simple Data/Bar Graphs/'

color=['red', 'orange', 'gold', 'green', 'blue', 'indigo', 'purple',  'lime', 'lightpink', 'black']
label_array = ['WW', 'SD']
wl_array = np.linspace(500, 850, num = 110) # Band range is from 500 - 850 after noisy bands are removed


######################################################
#
# Labelling Spectrum
#
######################################################
def labelling_spectrum(day):
  spectrum = np.array([mat['CdataPxsd'], mat['CdataPxww']])
  # (2, 9, 10) group, day, plant
  total_data = np.empty((1, 112))

  group_no = 2
  plant_no = 3

  for group in range(group_no):
    for plant in range(plant_no):
      tmp = spectrum[group][day][plant]
      pixel_no = tmp.shape[0]
      if(group == 0): # 0 is SD
        labels = np.full((pixel_no, 1), 0)
      else: # 1 is WW
        labels = np.full((pixel_no, 1), 1)
      tmp = np.hstack((tmp, labels))
      total_data = np.vstack((total_data, tmp))
  return total_data

# print('sum' + ' - ' + str(sum))
# total_data = np.empty((sum, 111)) # Exact size is initialize first to not prevent numpy append method which is O(N**2)


########################################################
#
# Generating Train-Test Data
#
########################################################
def generating_train_test_data():
  print(np.shape(total_data))
  np.random.shuffle(total_data)
    
  X_train, X_test, y_train, y_test = train_test_split(total_data[:, 0:111], total_data[:, 111], test_size=0.3)
  unique, counts = np.unique(y_train, return_counts=True)
  print(dict(zip(unique, counts)))
  return X_train, X_test, y_train, y_test, unique, counts


########################################################
#
# Model Training
#
########################################################
def model_training(day):
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
  model.add(keras.layers.Dense(2, activation='softmax', name="Dense_3"))
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  
  history = model.fit(X_train, y_train, batch_size=256, epochs=100, validation_split=0.3) 
  # From past testing using epoch 300, it shows accuracy and loss converges after 50, so set 150 epoch will be good enough
  model.evaluate(X_test, y_test)

  # Plot
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model Day-'+ str(day+1) +' Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  # plt.savefig(model_storage_path + 'Train Test Graphs/' + ' ' + str(day+1) + ' Accuracy' + '.png')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Day-'+ str(day+1) +' loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  # plt.savefig(model_storage_path + 'Train Test Graphs/' + ' ' + str(day+1) + ' Loss' + '.png')
  plt.show()

  # Save Model
  # filename = model_storage_path + "_" + str(day) + "_100.pkl" # type day epoch

  # with open(filename, 'wb') as file:  
    # pickle.dump(model, file)
  # return model


########################################################
#
# Model Training
#
########################################################
def test(total_data, dy):
  ww_spec = []
  pd_spec = []
  for d in total_data:
    if (d[-1] == 1):
      ww_spec.append(d)
    elif (d[-1] == 0):
      pd_spec.append(d)

  ext_ww = ww_spec[:2000]
  ext_pd = pd_spec[:2000]

  for i in range(100):
    plt.plot(np.linspace(0,111,111), ext_ww[i][:111], color='b')
    plt.plot(np.linspace(0,111,111), ext_pd[i][:111], color='r')
  plt.title(dy)
  plt.show()

# Day 0 is purposely made to fail learn, DO NOT train the model again
for d in range(0, 9): # initially (2,9) day will be zero index, so here 2-8 means 3-9
  total_data = labelling_spectrum(d)
  test(total_data, d)
  # X_train, X_test, y_train, y_test, unique, counts = generating_train_test_data()  
  # model_training(d)


########################################################
#
# Get P-values
#
########################################################
def get_p (ww, sd, model):
  percentage_difference_ww = ww[:, 0] - ww[:, 1]
  percentage_difference_sd = sd[:, 0] - sd[:, 1]
  interval = 9
  for i in range(0,7):
    globals()[f"percentage_difference_ww_{str(i+1)}"] = percentage_difference_ww[interval*(i):interval*(i+1)]
    globals()[f"percentage_difference_sd_{str(i+1)}"] = percentage_difference_sd[interval*(i):interval*(i+1)]
    
  percentage_difference_ww = np.vstack((percentage_difference_ww_1, percentage_difference_ww_2, percentage_difference_ww_3, percentage_difference_ww_4, percentage_difference_ww_5, percentage_difference_ww_6, percentage_difference_ww_7))
  percentage_difference_sd = np.vstack((percentage_difference_sd_1, percentage_difference_sd_2, percentage_difference_sd_3, percentage_difference_sd_4, percentage_difference_sd_5, percentage_difference_sd_6, percentage_difference_sd_7))
  
  for tmp_d in range(9):
    d_ww = percentage_difference_ww[:, tmp_d]
    d_sd = percentage_difference_sd[:, tmp_d]
    p_val = stats.ttest_ind(a=d_ww, b=d_sd, equal_var=True)
    print('Model ' + str(model + 1) + ': Day: ' + str(tmp_d + 1) + ' P-value: ' + str(p_val))

########################################################
#
# Get P-values
#
########################################################
def get_diff(all_y_values_percentage):
  percentage_difference_ww = all_y_values_percentage[:, 0] - all_y_values_percentage[:, 1]
  interval = 9
  for i in range(0,7):
    globals()[f"percentage_difference_ww_{str(i+1)}"] = percentage_difference_ww[interval*(i):interval*(i+1)]
    
  percentage_difference_ww = np.vstack((percentage_difference_ww_1, percentage_difference_ww_2, percentage_difference_ww_3, percentage_difference_ww_4, percentage_difference_ww_5, percentage_difference_ww_6, percentage_difference_ww_7))
  percentage_difference_ww_mean = np.mean(percentage_difference_ww, axis = 0)
  percentage_difference_ww_err = percentage_difference_ww - percentage_difference_ww_mean
  
  percentage_difference_ww_err_max = np.amax(percentage_difference_ww_err, axis = 0)
  percentage_difference_ww_err_min = np.absolute(np.amin(percentage_difference_ww_err, axis = 0))
  
  return [percentage_difference_ww_err_min, percentage_difference_ww_err_max], percentage_difference_ww_mean


########################################################
#
# Plot Bar Graphs
#
########################################################
def average_plot_bar(all_y_values, plot_model_day, group):
  if (group == 0):
    group_str= 'Progressive Drought'
  else:
    group_str= 'Well-Watered'

  tmp_list_sd = []
  tmp_list_ww = []
  for m in range(7):# 7 plants
    tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0] # 9 days
    tmp_list_sd.append(tmp)
    tmp_list_ww.append(tmp)
      
  for k in range(7): # 7 plants
    tmp_list_sd[k] = all_y_values[(k*9) : (k*9)+9, 0]
    tmp_list_ww[k] = all_y_values[(k*9) : (k*9)+9, 1]
  mean_y_values_sd = np.mean(tmp_list_sd, axis=0)
  mean_y_values_ww = np.mean(tmp_list_ww, axis=0)
  print(tmp_list_sd)
  print(mean_y_values_ww)
  print(mean_y_values_sd)

  for index , day_values in enumerate(mean_y_values_sd):  
    # plt.bar(x_values+ (index/10), day_values/np.sum(day_values), color =color[index], width = 0.3)   
    tmp_mean = [mean_y_values_sd[index], mean_y_values_ww[index]]
    if (index == 0):
      plt.bar(index + 1, tmp_mean[0]/np.sum(tmp_mean), color='darkred', width = 0.3, label = 'PD') 
      plt.bar(index + 1.4, tmp_mean[1]/np.sum(tmp_mean), color='darkblue', width = 0.3, label = 'WW') 
    else:
      plt.bar(index + 1, tmp_mean[0]/np.sum(tmp_mean), color='darkred', width = 0.3) 
      plt.bar(index + 1.4, tmp_mean[1]/np.sum(tmp_mean), color='darkblue', width = 0.3) 

  plt.legend(loc = 'upper right')
  plt.ylim(0, 1.3)
  plt.xlabel('Day')
  plt.ylabel('Average Pixel Percentage')
  plt.title( 'Model Day- ' + str(plot_model_day + 1) + ' ' + group_str +' Average Classification Result Across 9 Days')
  plt.savefig(bar_graphs_storage_path + 'Average Model ' + str(plot_model_day + 1) + ' ' + group_str + ' Classification' + '.png')
  plt.show()

def plot_bar(all_y_values, plant, plot_model_day, group):
  if (group == 0):
    group_str= 'Progressive Drought'
  else:
    group_str= 'Well-Watered'

  for index , day_values in enumerate(all_y_values):  
    # plt.bar(x_values+ (index/10), day_values/np.sum(day_values), color =color[index], width = 0.3)   
    if (index == 0):
      plt.bar(index + 1, day_values[0]/np.sum(day_values), color='darkred', width = 0.3, label = 'PD') 
      plt.bar(index + 1.4, day_values[1]/np.sum(day_values), color='darkblue', width = 0.3, label = 'WW') 
    else:
      plt.bar(index + 1, day_values[0]/np.sum(day_values), color='darkred', width = 0.3) 
      plt.bar(index + 1.4, day_values[1]/np.sum(day_values), color='darkblue', width = 0.3) 

  plt.legend(loc = 'upper right')
  plt.ylim(0, 1.3)
  plt.xlabel('Day')
  plt.ylabel('Pixel Percentage')
  plt.title( 'Model Day-' + str(plot_model_day + 1) + ': ' + group_str +' Plant ' + str(plant + 1) + ' Classification Result Across 9 Days')
  plt.savefig(bar_graphs_storage_path + 'Model ' + str(plot_model_day + 1) + ' ' + group_str + ' Plant ' + str(plant + 1) + ' Classification' + '.png')
  plt.show()
  

########################################################
#
# Load Model and Save Graphs
#
########################################################
# from tempfile import template
def load_model_comparing_save_line(model_day):
  group_no = 2
  plant_no = 10
  day_no = 9

  all_y_values = np.zeros(((plant_no-3) * 9,group_no))
  all_y_values_percentage_ww = np.zeros(((plant_no-3) * 9, group_no))
  all_y_values_percentage_sd = np.zeros(((plant_no-3) * 9, group_no))
  # Load Model
  load_filename = model_storage_path + "_" + str(model_day) + "_100.pkl" # type model_day epoch
  with open(load_filename, 'rb') as file:  
    loaded_model = pickle.load(file)
  
  all_y_values_index = 0
  for plant in range(3, plant_no):
    print('plant:   ' + str(plant))
    for day in range(day_no):
      spec = spectrum[0][day][plant]
      predictions = loaded_model.predict(spec)
      predicted_index = np.empty((predictions.shape[0], 1))

      for index, prediction in enumerate(predictions):
        predicted_index[index] = np.argmax(prediction)
      
      predicted_index = predicted_index.flatten()

      ### Histogram code below
      unique, counts = np.unique(predicted_index, return_counts=True)
      temp_dict = dict(zip(unique, counts))

      x_values = np.linspace(1, group_no, num = group_no)
      y_values = np.empty((group_no))
      for k in range(0, group_no): # inclusive for start, exclusive for end
        if (k in unique):
          y_values[k] = temp_dict[k]
        else:
          y_values[k] = 0
          
      all_y_values[all_y_values_index] = y_values
      total_count = predicted_index.size
      all_y_values_percentage_sd[all_y_values_index] = y_values/total_count
      all_y_values_index += 1

    plot_bar(all_y_values_percentage_sd[all_y_values_index-9 : all_y_values_index], plant, model_day, 0)
  average_plot_bar(all_y_values_percentage_sd, model_day, 0)
  err_ww, percentage_difference_ww_mean = get_diff(all_y_values_percentage_sd)

  all_y_values_index = 0
  for plant in range(3, plant_no):
    print('plant:   ' + str(plant))
    for day in range(day_no):
      spec = spectrum[1][day][plant]
      predictions = loaded_model.predict(spec)
      predicted_index = np.empty((predictions.shape[0], 1))

      for index, prediction in enumerate(predictions):
        predicted_index[index] = np.argmax(prediction)
      
      predicted_index = predicted_index.flatten()

      ### Histogram code below
      unique, counts = np.unique(predicted_index, return_counts=True)
      temp_dict = dict(zip(unique, counts))
      print(temp_dict)

      x_values = np.linspace(1, group_no, num = group_no)
      y_values = np.empty((group_no))
      for k in range(0, group_no): # inclusive for start, exclusive for end
        if (k in unique):
          y_values[k] = temp_dict[k]
        else:
          y_values[k] = 0
      print(all_y_values_index)
      all_y_values[all_y_values_index] = y_values
      total_count = predicted_index.size
      all_y_values_percentage_ww[all_y_values_index] = y_values/total_count
      all_y_values_index += 1

    plot_bar(all_y_values_percentage_ww[all_y_values_index-9 : all_y_values_index], plant, model_day, 1)
  average_plot_bar(all_y_values_percentage_ww, model_day, 1)
  err_sd, percentage_difference_sd_mean = get_diff(all_y_values_percentage_ww)

  get_p(all_y_values_percentage_ww, all_y_values_percentage_sd, model_day)

  plt.errorbar(np.linspace(1,9,num = 9), percentage_difference_sd_mean, yerr = err_sd, color = 'red', label = 'PD',  fmt='-o')
  plt.errorbar(np.linspace(1,9,num = 9), percentage_difference_ww_mean, yerr = err_ww, color = 'blue', label = 'WW',  fmt='-o')
  plt.legend()
  plt.title('Model Day-'+ str(model_day+1))
  plt.xlabel('Day')
  plt.ylabel('Percentage Difference')
  plt.savefig(line_graphs_storage_path + ' ' + str(model_day+1) + '.png')
  plt.show()


########################################################
#
# Main Code
#
########################################################
spectrum = np.array([mat['CdataPxsd'], mat['CdataPxww']])

for d in range(0, 1): # initially (0,9) day will be zero index, so here 2-8 means 3-9
  load_model_comparing_save_line(d)