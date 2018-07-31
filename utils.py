import sys
from six.moves import xrange
import numpy as np
import tensorflow as tf
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.privacy_accountant.tf import accountant
from differential_privacy.dp_sgd.dp_optimizer import sanitizer

def MnistInput(mnist_data_file, batch_size, randomize, FLAGS):
    """Create operations to read the MNIST input file.

    Args:
      mnist_data_file: Path of a file containing the MNIST images to process.
      batch_size: size of the mini batches to generate.
      randomize: If true, randomize the dataset.

    Returns:
      images: A tensor with the formatted image data. shape [batch_size, 28*28]
      labels: A tensor with the labels for each image.  shape [batch_size]
    """
    file_queue = tf.train.string_input_producer([mnist_data_file])
    reader = tf.TFRecordReader()
    _, value = reader.read(file_queue)
    example = tf.parse_single_example(
        value,
        features={"image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
                  "image/class/label": tf.FixedLenFeature([1], tf.int64)})

    image = tf.cast(tf.image.decode_png(example["image/encoded"], channels=1),
                    tf.float32)
    image = tf.reshape(image, [FLAGS.image_size * FLAGS.image_size])
    image /= 255
    label = tf.cast(example["image/class/label"], dtype=tf.int32)
    label = tf.reshape(label, [])

    if randomize:
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            capacity=(batch_size * 100),
            min_after_dequeue=(batch_size * 10))
    else:
        images, labels = tf.train.batch([image, label], batch_size=batch_size)

    return images, labels

def Eval(mnist_data_file, network_parameters, num_testing_images,
         randomize, load_path, FLAGS, save_mistakes=False):
    """Evaluate model for a number of steps.

    Args:
      mnist_data_file: Path of a file containing the MNIST images to process.
      network_parameters: parameters for defining and training the network.
      num_testing_images: the number of images we will evaluate on.
      randomize: if false, randomize; otherwise, read the testing images
        sequentially.
      load_path: path where to load trained parameters from.
      save_mistakes: save the mistakes if True.

    Returns:
      The evaluation accuracy as a float.
    """
    batch_size = 100
    # Like for training, we need a session for executing the TensorFlow graph.
    with tf.Graph().as_default(), tf.Session() as sess:
        # Create the basic Mnist model.
        images, labels = MnistInput(mnist_data_file, batch_size, randomize, FLAGS)
        logits, _, _ = utils.BuildNetwork(images, network_parameters)
        softmax = tf.nn.softmax(logits)

        # Load the variables.
        ckpt_state = tf.train.get_checkpoint_state(load_path)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            raise ValueError("No model checkpoint to eval at %s\n" % load_path)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        coord = tf.train.Coordinator()
        _ = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_examples = 0
        correct_predictions = 0
        image_index = 0
        mistakes = []
        for _ in xrange((num_testing_images + batch_size - 1) // batch_size):
            predictions, label_values = sess.run([softmax, labels])

            # Count how many were predicted correctly.
            for prediction, label_value in zip(predictions, label_values):
                total_examples += 1
                if np.argmax(prediction) == label_value:
                    correct_predictions += 1
                elif save_mistakes:
                    mistakes.append({"index": image_index,
                                     "label": label_value,
                                     "pred": np.argmax(prediction)})
                image_index += 1

    return (correct_predictions / total_examples,
            mistakes if save_mistakes else None)

def Validate(priv_accountant, sess, FLAGS):
    # Fetch spent privacy budget from accountant
    if priv_accountant and FLAGS.eps > 0:
        spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=FLAGS.eps)
        for spent_eps, spent_delta in spent_eps_deltas:
            sys.stderr.write("spent privacy: eps %.4f delta %.5g\n" % (spent_eps, spent_delta))

    train_accuracy, _ = Eval(FLAGS.training_data_path, FLAGS.network_parameters,
                             num_testing_images=FLAGS.num_testing_images,
                             randomize=True, load_path=FLAGS.save_path, FLAGS=FLAGS)
    
    eval_accuracy, _ = Eval(FLAGS.eval_data_path, FLAGS.network_parameters,
                            num_testing_images=FLAGS.num_testing_images,
                            randomize=False, load_path=FLAGS.save_path, FLAGS=FLAGS)
    
    sys.stderr.write("train_accuracy: %.2f\n" % train_accuracy)
    sys.stderr.write("eval_accuracy: %.2f\n" % eval_accuracy)

    return train_accuracy, eval_accuracy

def BoundGradients(training_params, priv_accountant, network_parameters, batch_size):
    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
          priv_accountant,
          [network_parameters.default_gradient_l2norm_bound / batch_size, True]
    )
    for var in training_params:
        if "gradient_l2norm_bound" in training_params[var]:
            l2bound = training_params[var]["gradient_l2norm_bound"] / batch_size
            gaussian_sanitizer.set_option(var, sanitizer.ClipOption(l2bound, True))
    return gaussian_sanitizer


def CreateModel(sess, mnist_train_file, FLAGS):
    """
        This function creates a simple feed-forward neural network.
    """
    batch_size = FLAGS.batch_size
    network_parameters = FLAGS.network_parameters
    # Create the basic Mnist model.
    images, labels = MnistInput(mnist_train_file, FLAGS.batch_size, FLAGS.randomize, FLAGS=FLAGS)
    logits, projection, training_params = utils.BuildNetwork(images, network_parameters)

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(labels, 10))
    # The actual cost is the average across the examples.
    cost = tf.reduce_sum(cost, [0]) / batch_size
    
    return cost, logits, training_params



def InitializeGraph(sess, mnist_train_file, FLAGS, optimizer):
    """
    This function constructs a simple feed-forward neural network, and privacy accountant.
    It initializes all of the variables in the graph and returns the cost function,
    along with a gradient descent operation.
    """
    batch_size = FLAGS.batch_size
    network_parameters = FLAGS.network_parameters

    cost, logits, training_params = CreateModel(sess, mnist_train_file, FLAGS)

    # Add global_step
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    with_privacy = FLAGS.eps > 0
    gd_op = optimizer.minimize(cost, global_step=global_step)
    coord = tf.train.Coordinator()
    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    # We need to maintain the intialization sequence.
    for v in tf.trainable_variables():
        sess.run(tf.variables_initializer([v]))
    sess.run(tf.global_variables_initializer())
    
    return cost, gd_op, priv_accountant


