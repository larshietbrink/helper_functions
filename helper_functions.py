# Small helper functions to use in data science projects
# Created by Lars Hietbrink

# Let's create a function to plot our loss curves...
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

# Create TensorBoard callback
def create_tensorboard_callback(dir_name: str, experiment_name: str):
  """
  Returns a tensorboard callback object that saves logs to specified directory and automatically creates a logfile name.

  Args:
    dir_name: the name where the log files should be stored (string).
    experiment_name: the name of the experiment (string).
  """

  # Set the log file directory and file name and create a tensorboard callback
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

  # Print result and return callback object
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Plot the validation and training curves:
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  
  Returns:
    Two plots:
      1. Training loss vs. Validation loss
      2. Training accuracy vs. Validation accuracy
  """
  
  # Get the loss values (both training and validation)
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  # Get the accuracy values (both training and validation)
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  # Set the number of epochs found in the history object
  epochs = range(len(history.history["loss"]))

  # Plot loss
  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label="training_accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()