import tensorflow as tf
import dataStuff
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)

BATCH_SIZE = 2
EPOCHS = 150
RUNS = 3


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


def do_plotting(labels, val_std, val_mean):
    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, val_std, width, label='std')

    fig2, ax2 = plt.subplots()
    rects2 = ax2.bar(x, val_mean, width, label='mean')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by attributes in std')
    ax.set_xticks(x)
    ax.tick_params(labelrotation=90)
    ax.set_xticklabels(labels)

    ax2.set_ylabel('Scores')
    ax2.set_title('Scores by attributes in mean')
    ax2.set_xticks(x)
    ax2.tick_params(labelrotation=90)
    ax2.set_xticklabels(labels)

    fig.tight_layout()
    fig2.tight_layout()
    plt.show()

def main():
    # get the cleaned dataframes
    df, target_df = dataStuff.do_data_stuff()

    # divide dataset into train, test and validation data (20% is test set, 16% is validation set and 64% is training set)
    x_train, x_test, y_train, y_test = train_test_split(df, target_df, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # init a dictonary to store the performance per rating model
    val_accuracy = {}

    # init for each rating a model and train and evaluate on it and store the performance result in the skill dictonary
    for i in range(0, RUNS):
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

            if rating in val_accuracy:
                val_accuracy[rating].append(model.evaluate(test_tensor, test_target_tensor)[1])
            else:
                val_accuracy[rating] = [model.evaluate(test_tensor, test_target_tensor)[1]]

    val_accuracy_mean = {}
    val_accuracy_std = {}
    for key in val_accuracy:
        val_accuracy_mean[key] = np.mean(val_accuracy[key])
        val_accuracy_std[key] = np.std(val_accuracy[key])

    labels = list(val_accuracy.keys())
    val_std = list(val_accuracy_std.values())
    val_mean = list(val_accuracy_mean.values())

    print('val std:', val_std)
    print('val mean:', val_mean)

    do_plotting(labels, val_std, val_mean)

    with open('results.txt', 'w') as f:
        f.write('val std:')
        f.write(str(val_std))
        f.write('\n')
        f.write('val mean:')
        f.write(str(val_mean))



if __name__ == "__main__":
    main()
