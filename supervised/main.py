import csv
from matplotlib import pyplot as plt
import alexnet
from time import time
import tensorflow as tf
import fer2013 

def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def main(argv):
    steps = 5000
    batch_size = 7178
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # fer2013.parser('../fer2013.csv')

    train_x, train_y, test_x, test_y = fer2013.load_data()
    print(train_x.shape)
    print(test_x.shape)
    my_feature_columns = [tf.feature_column.numeric_column(key='img',shape=[48,48,1])]
    
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        n_classes=7)

    s = time()
    classifier.train(
        input_fn=lambda:train_input_fn({'img':train_x}, train_y, batch_size),
        steps=steps)
    e = time() - s
    
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn({'img':test_x}, test_y, batch_size))
    train_result = classifier.evaluate(input_fn=lambda:eval_input_fn({'img':train_x}, train_y, batch_size))

    print('\nTrain set accuracy: {accuracy:0.3f}'.format(**train_result))
    print('Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print('Evaluated in {} seconds\n'.format(round(e,2)))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    
