import tensorflow as tf
import numpy as np

def alexnet_layer(tensor_in, n_filters, filter_shape, pool_size, activation=tf.nn.tanh,
                  padding='VALID', norm_depth_radius=4, dropout=None):
    conv = learn.ops.conv2d(tensor_in,
                            n_filters=n_filters,
                            filter_shape=filter_shape,
                            activation=activation,
                            padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding=padding)
    norm = tf.nn.lrn(pool, depth_radius=norm_depth_radius, alpha=0.001 / 9.0, beta=0.75)
    if dropout:
        norm = learn.ops.dropout(norm, dropout)
    return norm


def alex_conv_pool_layer(tensor_in, n_filters, kernel_size, stride, pool_size, pool_stride,
                         activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=n_filters,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=activation_fn,
                                           padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_stride, padding=padding)
    return pool


def alex_3_convs_pool_layer(tensor_in, activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=256,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding=padding)
    return pool

def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat

def dense_layer(tensor_in, layers, activation_fn=tf.nn.tanh, keep_prob=None):
    if not keep_prob:
        return tf.contrib.layers.stack(
            tensor_in, tf.contrib.layers.fully_connected, layers, activation_fn=activation_fn)

    tensor_out = tensor_in
    for layer in layers:
        tensor_out = tf.contrib.layers.fully_connected(tensor_out, layer,
                                                       activation_fn=activation_fn)
        tensor_out = tf.contrib.layers.dropout(tensor_out, keep_prob=keep_prob)

    return tensor_out

def alexnet_model(features, labels, mode, params):
    #X, y, image_size=(-1, IMAGE_SIZE, IMAGE_SIZE, 3)):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.reshape(net, (params['batch_size'], 48, 48, 1))
    
    with tf.variable_scope('layer1'):
        layer1 = alex_conv_pool_layer(net, 96, [11, 11], 4, (1, 3, 3, 1), (1, 2, 2, 1))

    with tf.variable_scope('layer2'):
        layer2 = alex_conv_pool_layer(layer1, 256, [5, 5], 2, (1, 3, 3, 1), (1, 2, 2, 1))

    with tf.variable_scope('layer3'):
        layer3 = alex_3_convs_pool_layer(layer2)
        layer3_flat = flatten_convolution(layer3)
        
    logits = dense_layer(layer3_flat, [4096, 4096, params['n_classes']], activation_fn=tf.nn.tanh, keep_prob=0.2)
    
    # prediction is confidence
    
    predicted_classes = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)    
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
        
    assert mode == tf.estimator.ModeKeys.TRAIN
            
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
        learning_rate=0.1)
    
    return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)


def my_alexnet(features, labels, mode, params):
    training = False
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True
        
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.reshape(net, (params['batch_size'], 48, 48, 1))
    conv = tf.contrib.layers.convolution2d(net,
                                           num_outputs=64,
                                           kernel_size=[5,5],
                                           stride=1,
                                           activation_fn=tf.nn.relu,
                                           padding='SAME')
    pool = tf.nn.max_pool(conv, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
    conv = tf.contrib.layers.convolution2d(pool,
                                           num_outputs=64,
                                           kernel_size=[5,5],
                                           stride=1,
                                           activation_fn=tf.nn.relu,
                                           padding='SAME')
    pool = tf.nn.max_pool(conv, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
    conv = tf.contrib.layers.convolution2d(pool,
                                           num_outputs=128,
                                           kernel_size=[4,4],
                                           stride=1,
                                           activation_fn=tf.nn.relu,
                                           padding='SAME')
    conv_flat = flatten_convolution(conv)
#     dropout = tf.layers.dropout(conv_flat, rate=0.3, training=training)
#     net = tf.layers.dense(dropout, 3072)
#     logits = tf.layers.dense(net, params['n_classes'])
    dropout = tf.contrib.layers.dropout(conv_flat, keep_prob=0.3, is_training=training)
    net = tf.contrib.layers.fully_connected(dropout, 3072)
    logits = tf.contrib.layers.fully_connected(net, params['n_classes'])
    
    predicted_classes = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)    
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    
    print('Accuracy',accuracy)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
        
    assert mode == tf.estimator.ModeKeys.TRAIN
            
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(), optimizer='Momentum', learning_rate=0.001)
    
    return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
