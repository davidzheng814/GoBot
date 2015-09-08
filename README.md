# GoBot
GoBot is a Convolutional Neural Network that plays the board game Go, one of the few classic strategy games in which humans still have an edge on computers.
GoBot takes in a board-state and outputs a probability vector, which indicates its belief on the best possible responses.

## Setup
To use GoBot, you will need to install `Theano`, `lasagne`, and their dependences. In addition, you may need to do some extra setup if you wish to have Theano train the neural net with your GPU. 
I used a GPU-optimized Ubuntu instance on Amazon Web Services to train my network. The bash commands I used to install necessary libraries and dependencies can be found in `setup.sh`. 
