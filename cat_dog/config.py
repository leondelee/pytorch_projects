#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Configuration:
    """
    configuration parameters for this project
    """
    num_classes = 2         # how many classes to be predicted
    env = 'default'         # env parameter for visdom
    model_name = 'AlexNet'  # the name of model to be used, which has to be same as the one in 'models/__init__.py'
    batch_size = 1        # number of batches
    load_data_workers = 1   # number of workers for data loading
    train_all_path = '/home/speit/torch/data/cat_dog/training_set/all/'
    test_all_path = '/home/speit/torch/data/cat_dog//test_set/all/'
    train_dog_path = '/home/speit/torch/cat_dog/data_related/training_set/dogs/'
    #train_dog_path = '/home/speit/torch/cat_dog/data_related/demo_data/'
    train_cat_path = '/home/speit/torch/cat_dog/data_related/training_set/cats/'
    test_dog_path = '/home/speit/torch/cat_dog/data_related/test_set/dogs/'
    test_cat_path = '/home/speit/torch/cat_dog/data_related/test_set/cats/'
    use_gpu = False         # whether to use gpu or not
    print_frequency = 20    # print info every print_frequency batches
    result_file = '/result/result.csv'  # the position to store result
    max_epoch = 10          # max training epoch
    learning_rate = 0.01    # learning rate
    learning_decay = 0.95   # when loss value increase, learning_rate = learning_decay * learning_rate
    weight_decay = 1e-4
    time_format = "%m_%d_%H:%M:%S"
    log_dir_path = '/home/speit/torch/cat_dog/log/'
    checkpoint_dir_path = '/home/speit/torch/checkpoints/cat_dog/'

    def parse(self, kwargs):
        import warnings
        for key, value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn("Warning: Do not have attribute %s" % key)
                setattr(self, key, value)

        print("User config:")
        for key, value in self.__class__.__dict__.items():
            if not key.startwith('__'):
                print(key, getattr(self, key))


