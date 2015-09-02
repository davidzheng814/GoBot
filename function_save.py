import cPickle
import theano.tensor as T
import lasagne
import numpy as np
import theano

def pickle(function, filename):
    f = file(filename, 'wb')
    cPickle.dump(function, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def restrict(results, restriction_var):
    '''
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

def save_function(network, input_var, restriction_var, output_file):
    '''
        Input:
            network: trained lasagne network
            output_file: filename of pickled function
    '''

    prediction = lasagne.layers.get_output(network, deterministic=True)
    prediction = restrict(prediction, restriction_var)

    function = theano.function([input_var, restriction_var], [prediction])

    pickle(function, output_file)

