import numpy as np
import os

input_folder = '/Users/dzd123/Documents/Summer 2015/GoBot/training_data/'
output_folder = '/Users/dzd123/Documents/Summer 2015/GoBot/training_data_split/'

def split_training_data():
    for filename in os.listdir(input_folder):
        if not filename.endswith('.npz'):
            continue

        with np.load(input_folder+filename) as data:
            X = data['inputs']
            y = data['targets']

            for i, (inp, target) in enumerate(zip(X, y)):
                np.savez_compressed(output_folder+filename[:-4]+'_'+str(i), input=inp, target=target)

if __name__ == '__main__':
    split_training_data()