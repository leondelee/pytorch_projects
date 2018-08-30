# Some Deep Learning Projects

Author: Liangwei Li 
</br>E-mail: liliangwei@sjtu.edu.cn
<!-- TOC -->

- [Some Deep Learning Projects](#some-deep-learning-projects)
    - [Project 1: Cat vs Dog](#project-1-cat-vs-doghttpsgithubcomleondeleepytorch_projectstreemastercat_dog)

<!-- /TOC -->
## [Project 1: Cat vs Dog](https://github.com/leondelee/pytorch_projects/tree/master/cat_dog)

This project provides a pytorch version solution to a classic Kaggle competition: [Dogs and Cats](https://www.kaggle.com/tongpython/cat-and-dog). Every participant is supposed to train a model for this binary classification: to predict whether a picture represents a cat or a dog. In this project, one version is completed with pytorch-cpu and some other relative packages of python. It can record log and model state automatically at every epoch, which means it can start even if an interuption occurs. 
</br>
</br> Training data as well as test data are included in this project.
</br> 
</br> Mainly, ResNet34 is used as the model to be trained. However, in .models, any possible model can be difined and thus trained. 
</br>
</br> **To begin the training process on your own computer**, some specific dependent packages are required, which are listed in requirements.txt. When the environment is set, your need to make some changes of the parameters "train_all_path, test_all_path, train_dog_path, test_dog_path, train_cat_path, test_cat_path " in config.py, which will be used to define the location of the data set. After this, run main.py and every thing should work normally. 
</br>
</br>To test the proformance of the model, make one change at the end of main.py as follows:

    #train(model, training_all)
    evaluate(model, dev_all)
# To be continued
