import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys, os
import time

'''
    Hyper-parameters Below
'''
batch_size = 128
num_epochs = 10

num_channels = 8
board_size = 19

# Folder containing all datapoints as npz files
data_folder = '/mnt/training_data_split/training_data_split/'

# None if neural network weights are set to random
# Else, filename of neural network's parameters
params_file = 'model.npz'

# A list of the number of filters in each convolutional layer of the neural network
num_filters_list = [64, 64, 64, 64, 48, 48, 32, 32]
# A list of the size of each filter in each convolutional layer
filter_size_list = [9, 7, 5, 5, 5, 5, 5, 5]

# Dropout probability
dropout = 0

# A 3-tuple representing the ratio at which datapoints are divided into training, validation, and testing sets
# Must add up to 1
train_val_test_ratio = (.88, .04, .08)

# None if use all datapoints in folder
# Else, the maximum number of datapoints in the folder to use
max_datasets = None 

'''
    ^^^ Hyper-parameters ^^^
'''

'''
    Source Code Below
'''

files = [filename for filename in os.listdir(data_folder) if filename.endswith('.npz')]
if max_datasets:
    files = files[:max_datasets]

print "Num Files:", len(files)
train_val_test_sizes = (int(train_val_test_ratio[0]*len(files)), int((train_val_test_ratio[0]+train_val_test_ratio[1])*len(files)))
training_files = files[:train_val_test_sizes[0]]
validation_files = files[train_val_test_sizes[0]:train_val_test_sizes[1]]
testing_files = files[train_val_test_sizes[1]:]

files = (training_files, validation_files, testing_files)

def build_network(input_var, batch_size):
    '''
        Create a theano function representing the neural net.
    '''

    network = lasagne.layers.InputLayer(shape=(batch_size, num_channels, board_size, board_size), input_var=input_var)
    # network = lasagne.layers.DropoutLayer(network, p=0.2)

    for num_filters, filter_size in zip(num_filters_list, filter_size_list):
        network = lasagne.layers.Conv2DLayer(network, num_filters, filter_size, pad='full',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal())

        if dropout:
            network = lasagne.layers.DropoutLayer(network, p=dropout)

    network = lasagne.layers.DenseLayer(network, num_units=board_size*board_size, nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches(dataset_ind, shuffle=False):
    '''
        Creates an iterator to generate batches for training/validation/testing sets.

        dataset_ind = 0 if training set, 1 if validation set, 2 if testing set
    '''
    dataset = files[dataset_ind]

    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(dataset) - batch_size + 1, batch_size):
        filenames = [dataset[index] for index in indices[start_idx:start_idx+batch_size]]

        input_batch = []
        target_batch = []
        for filename in filenames:
            with np.load(data_folder+filename) as data:
                input_batch.append(data['input'].astype(theano.config.floatX))
                target_batch.append(data['target'].astype(np.uint16))

        yield np.concatenate(input_batch), np.concatenate(target_batch)

def restrict(results, restriction_var):
    '''
        Normalize a Theano output vector to legal moves.

        Inputs:
            results: TensorVariable (batch_size, 361) [softmax output]
            restriction_var: TensorVariable (batch_size, 19, 19) [1 if legal, 0 otherwise]
        Output:
            TensorVariable (batch_size, 361)
    '''
    flattened = T.flatten(restriction_var, outdim=2)

    results = results*(1. - flattened)

    results_sum = T.sum(results, axis=1, keepdims=True)

    return results/results_sum

def main():
    print "Architecturing Network..."
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    restriction_var = T.tensor3('restrictions')
    learning_rate_var = T.scalar('learningrate')

    network = build_network(input_var, batch_size)
    
    if params_file:
        with np.load(params_file) as params:
            lasagne.layers.set_all_param_values(network, params['arr_0'])

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate_var)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = restrict(test_prediction, restriction_var)

    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    print "Network Architecture Complete"

    print "Compiling..."
    train_fn = theano.function([input_var, target_var, learning_rate_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var, restriction_var], [test_loss, test_acc])

    print "Compilation Complete"

    print "Start Training..."

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i, batch in enumerate(iterate_minibatches(0, shuffle=True)):
            if epoch <= 6:
                learning_rate = .05
            elif epoch <= 8:
                learning_rate = .01
            else:
                learning_rate = .005

            inputs, targets = batch
            train_err += train_fn(inputs, targets, learning_rate)
            train_batches += 1

            if i % 5000 == 0:
                print epoch, i, 0

        val_err = 0
        val_acc = 0
        val_batches = 0
        for i, batch in enumerate(iterate_minibatches(1, shuffle=False)):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets, inputs[:,6,:,:])
            val_err += err
            val_acc += acc
            val_batches += 1

            if i % 5000 == 0:
                print epoch, i, 1

        print "Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time)
        print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
        print "  validation loss:\t\t{:.6f}".format(val_err / val_batches)
        print "  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100)

        np.savez_compressed('model'+str(epoch)+'.npz', lasagne.layers.get_all_param_values(network))

    print "Training Complete"

    print "Start testing..."

    test_err = 0
    test_acc = 0
    test_batches = 0
    for i, batch in enumerate(iterate_minibatches(2, shuffle=False)):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets, inputs[:,6,:,:])
        test_err += err
        test_acc += acc
        test_batches += 1

        if i % 1000 == 0:
            print i, 2

    print "Final results:"
    print "  test loss:\t\t\t{:.6f}".format(test_err / test_batches)
    print "  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100)
    

main()
