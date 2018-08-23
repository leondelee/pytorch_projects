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
    train_all_path = '/home/speit/torch/data/cat_dog/training_set/all/' # the path to training data
    test_all_path = '/home/speit/torch/data/cat_dog//test_set/all/'     # the path to test data
    use_gpu = False         # whether to use gpu or not
    print_frequency = 20    # print info every print_frequency batches
    result_file = '/result/result.csv'  # the position to store result
    max_epoch = 10          # max training epoch
    learning_rate = 0.01    # learning rate
    learning_decay = 0.95   # when loss value increase, learning_rate = learning_decay * learning_rate
    weight_decay = 1e-4
    time_format = "%m_%d_%H:%M:%S"  # time format used in this project for logging and saving model
    log_dir_path = '/home/speit/torch/cat_dog/log/' # the path to log folder
    checkpoint_dir_path = '/home/speit/torch/checkpoints/cat_dog/' # the path to store trained model

    def parse(self, kwargs):
        """
        add additional attributes to Configuration class
        :param kwargs:
        :return:
        """
        import warnings
        for key, value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn("Warning: Do not have attribute %s" % key)
                setattr(self, key, value)

        print("User config:")
        for key, value in self.__class__.__dict__.items():
            if not key.startwith('__'):
                print(key, getattr(self, key))


