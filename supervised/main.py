import csv
from matplotlib import pyplot as plt
import alexnet
from time import time
import tensorflow as tf
import fer2013 
import sys

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

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(argv):
    steps = 20000
    batch_size = 50
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
#     fer2013.parser('../fer2013.csv')

    train_x, train_y, test_x, test_y = fer2013.load_data()
    # if alexnet, make sure length of sets is divisible by batch_size
    train_x, train_y = train_x[:-(len(train_x) % batch_size)], train_y[:-(len(train_y) % batch_size)]
    test_x, test_y = test_x[:-(len(test_x) % batch_size)], test_y[:-(len(test_y) % batch_size)]

    my_feature_columns = [tf.feature_column.numeric_column(key='img',shape=[48,48,1])]
    # Build 2 hidden layer DNN with 10, 10 units respectively.
#     classifier = tf.estimator.DNNClassifier(
#         feature_columns=my_feature_columns,
#         # Two hidden layers of 10 nodes each.
#         hidden_units=[10, 10],
#         n_classes=7)

    classifier = tf.estimator.Estimator(
        model_fn=alexnet.my_alexnet,
        params={
            'feature_columns': my_feature_columns,
            'batch_size':batch_size,
            'n_classes': 7,
        })

    s = time()
    classifier.train(
        input_fn=lambda:train_input_fn({'img':train_x}, train_y, batch_size),
        steps=steps)
    t = time() - s
    
    s = time()
    train_result = classifier.evaluate(input_fn=lambda:eval_input_fn({'img':train_x}, train_y, batch_size))
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn({'img':test_x}, test_y, batch_size))
    e = time() - s
    
    print('\nTrain set accuracy: {accuracy:0.3f}'.format(**train_result))
    print('Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print('Trained in {} seconds'.format(round(t,2)))
    print('Evaluated in {} seconds\n'.format(round(e,2)))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
