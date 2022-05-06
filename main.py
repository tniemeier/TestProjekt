import tensorflow as tf
import dataStuff
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)

BATCH_SIZE = 2
EPOCHS = 5


# function to build a simple fully connected neural network model
def get_basic_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model


def main():
    # get the cleaned dataframes
    df, target_df = dataStuff.do_data_stuff()

    # divide dataset into train, test and validation data (20% is test set, 16% is validation set and 64% is training set)
    x_train, x_test, y_train, y_test = train_test_split(df, target_df, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # init a dictonary to store the performance per rating model
    skill = {}

    # init for each rating a model and train and evaluate on it and store the performance result in the skill dictonary
    for rating in dataStuff.ratings:
        # convert dataframes to tensors
        tensor = tf.convert_to_tensor(x_train)
        target_tensor = tf.convert_to_tensor(y_train[rating])

        test_tensor = tf.convert_to_tensor(x_test)
        test_target_tensor = tf.convert_to_tensor(y_test[rating])

        val_tensor = tf.convert_to_tensor(x_val)
        val_target_tensor = tf.convert_to_tensor(y_val[rating])

        # create model, train it and evaluate it
        model = get_basic_model()
        history = model.fit(tensor, target_tensor, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(val_tensor, val_target_tensor))
        skill[rating] = model.evaluate(test_tensor,  test_target_tensor)

    # Code to plot training to validation accuracy
    """
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    """

    #TO-DO: aggregate results and show them graphically, skill is thought to use to plot it


if __name__ == "__main__":
    main()
