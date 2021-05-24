import os
import sys

from evaluate import evaluate_on_dataset


def test_part1():
	
	os.chdir('part1_MNIST_pointcloud')

	if not os.path.exists('Dataset'):
		os.system('wget https://www.dropbox.com/s/246zhani5mptt7a/Dataset.zip')
		os.system('unzip Dataset.zip')
	path_to_ds = 'Dataset/valid_ds.h5'

	

	accuracy = evaluate_on_dataset(path_to_ds)

	os.chdir('..')

	assert accuracy > 0.85


