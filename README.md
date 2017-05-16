1. Classification for CIFAR10 and CIFAR100 based on embeddings from pre-final layer of Inception V3. The transformed data was created using TensorFlow. This data can be availed from https://drive.google.com/open?id=0B4BnUOdKQQZpSm1aUlpieUx3UE0. 

This should save people without GPUs, about 5 hours of computation time.

2.
The batch normalization instance created using this class can find gradient with respect to parameters (i.e gamma and beta) using gradient with respect to outputs, outputs, and inputs. Accessor and mutator methods for the parameters have been provided, along with a method for assigning parameter penalties, which has currently been set to zero. Parameter initialization has been skipped and may be done based on user's choice, preferrably by using a common implementation for paramter initialization. This code is based on the paper https://arxiv.org/pdf/1502.03167.pdf. 
