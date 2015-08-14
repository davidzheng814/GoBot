import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys, os

batch_size = 128
num_epochs = 10

num_channels = 8
board_size = 19

data_folder = '/Users/dzd123/Documents/Summer 2015/GoBot/training_data/'

num_filters_list = [64, 64, 64, 64, 48, 48, 32, 32]
filter_size_list = [7, 7, 5, 5, 5, 5, 5, 5]

def load_dataset():
    X_train, y_train, X_val, y_val, X_test, y_test = (None, None, None, None, None, None)
    files = [filename for filename in os.listdir(data_folder) if filename.endswith('.npz')]

    num_test = int(.08*len(files))
    num_val = int(.04*len(files))

    #numpy arrays of int8 type
    for i, filename in enumerate(files):

        with np.load(data_folder+filename) as data:
            X = data['inputs']
            y = data['targets']

            X = X.astype('float32')
            y = y.astype('int8')

            if i == 0:
                X_test = X
                y_test = y
            elif i < num_test:
                X_test = np.append(X_test, X, axis=0)
                y_test = np.append(y_test, y, axis=0)
            elif i == num_test:
                X_val = X
                y_val = y
            elif i < num_test + num_val:
                X_val = np.append(X_val, X, axis=0)
                y_val = np.append(y_val, y, axis=0)
            elif i == num_test + num_val:
                X_train = X
                y_train = y
            else:
                X_train = np.append(X_train, X, axis=0)
                y_train = np.append(y_train, y, axis=0)


    return X_train, y_train, X_val, y_val, X_test, y_test

def build_network(input_var):
    network = lasagne.layers.InputLayer(shape=(batch_size, num_channels, board_size, board_size), input_var=input_var)
    # network = lasagne.layers.DropoutLayer(network, p=0.2)

    for num_filters, filter_size in zip(num_filters_list, filter_size_list):
        network = lasagne.layers.Conv2DLayer(network, num_filters, filter_size, pad='full',
            non_linearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(network, num_units=board_size*board_size, nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches(inputs, targets, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def restrict(results, restriction_var):
    '''
        Inputs:
            results: TensorVariable (batch_size, 361) [softmax output]
            restriction_var: TensorVariable (batch_size, 19, 19) [1 if legal, 0 otherwise]
        Output:
            TensorVariable (batch_size, 361)
    '''
    restriction_var = T.flatten(restriction_var, outdim=2)

    results = results*restriction_var

    results_sum = T.sum(axis=1, keepdims=True)

    return results/results_sum

def main():
    print "Loading Data..."
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    print "Loading Data Complete"

    print "Architecturing Network..."
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    restriction_var = T.tensor3('restrictions')

    network = build_network(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = restrict(test_prediction, restriction_var)

    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    print "Network Architecture Complete"

    print "Compiling..."
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var, restriction_var], [test_loss, test_acc])

    print "Start Training..."

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets, inputs[:,6,:,:])
            val_err += err
            val_acc += acc
            val_batches += 1

        print "Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time)
        print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
        print "  validation loss:\t\t{:.6f}".format(val_err / val_batches)
        print "  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100)

    print "Training Complete"

    print "Start testing..."

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    print "Final results:"
    print "  test loss:\t\t\t{:.6f}".format(test_err / test_batches)
    print "  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100)

    np.savez('model.npz', lasagne.layers.get_all_param_values(network))

main()