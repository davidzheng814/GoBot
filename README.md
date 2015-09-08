# GoBot
GoBot is a Convolutional Neural Network that plays the board game Go, one of the few classic strategy games in which humans still have an edge on computers.
GoBot uses supervised learning from professional games to develop a strategy of its own. 
GoBot takes in a board-state and outputs a probability vector, which indicates its belief on the best possible responses.

## Setup
To use GoBot, you will need to install `Theano`, `lasagne`, and their dependences. In addition, you may need to do some extra setup if you wish to have Theano train the neural net with your GPU. 
I used a GPU-optimized Ubuntu instance on Amazon Web Services to train my network. The bash commands I used to install necessary libraries and dependencies can be found in `setup.sh`. 

## Usage
### Create Training Data
GoBot provides a program to convert Go SGF files into numpy training data for GoBot to use.
Run `python create_training_data.py [SGF_FOLDER] [OUTPUT_FOLDER]` to convert a folder of SGF files into numpy files.

Each numpy file holds a single board-state and the correct response at that state. Although unnecessary, if you wish to parse the data file yourself, the format is as follows:
`data['input']` holds a (1,8,19,19) shape numpy array, representing the current gamestate. 
`data['target']` holds a numpy scalar from [0, 360], representing the correct response. 

### Training
To train GoBot, open `GoBot.py`. Set the hyper-parameters to the neural network as desired. Make sure to change `data_folder` to the path of the folder containing the training data created in the last step. Also, set `params_file` to `None` if you wish to training the neural network from scratch. 
Run `python GoBot.py` to start training. Consider running `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python GoBot.py` to set some GPU-optimized Theano settings in addition. The program will output a numpy file holding the nnet parameters after each epoch. 

### Saving as Pickled Function
If you wish to save a working GoBot neural network Theano function, run `python function_save.py [PARAMS-FILE] [OUTPUT-FILE]`, where `PARAMS-FILE` contains the path to the nnet params outputted after training. 

## Play Against GoBot
If you wish to play against GoBot, visit my other Github project [here](https://github.com/stevenhao/yeah). 
