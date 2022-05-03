# Dynamic Client Filtration for a Federated Learning Framework
## Built on top of the paper - Ditto: Fair and Robust Federated Learning Through Personalization

This repository contains the code for Group 13's Course Project

Texas Tech University - CS 5388 - Spring 2022

Caleb Darnell, Semih Cal, Hasan Al-Qudah

## Our Work

### Contribution and Summary
Our contribution to the field of federated learning is the dynamic client filtering used to proactively detect
malicious clients from polluting the shared global model. Our solution focuses on defense against data poisoning,
random weights, and boosting weights. By calculating the loss for each client model in the current epoch, we can
identify statistical outliers that are believe to be clients effected by data poisoning (or poorly preforming models).
Then, we employ a simplified aggregation function for the weights of each cnn layer for each of the current clients.
Using the newly created aggregated weight matrix, we are able to again detect statistical outliers. Finally,
we created a filtering solution that looks at the strikes against a client and labels it as malicious or not.
The clients identified as malicious are removed from the aggregation function towards the global model.

### Testing our solution
The success of our solution is measured in a similar fashion as proposed in the Ditto paper. Test accuracy
(represented as total test accuracy, malicious test accuracy, and benign test accuracy) in addition to variance
are our main metrics for determining the success of our solution. However, we have included another metric unique
to our solution we are calling "Corruption Detection Average Accuracy", which measures the accuracy of our filtering
(i.e., how accurate we were at removing corrupted clients before the aggregation function)

### Location
While we needed to make changes throughout the original Repo to get the framework to work for paper, 
the majority of our work is located in `flearn/trainers_MTL/ditto.py`.

The majority of our logic takes place after the local training iterations of the client and before the client
side weights are aggregated to update the global model. Specifically, most of the core logic is around 
lines 163 - 295. Additionally, we made some enhancements to the lambda logic proposed by Ditto's authors.
Our changes can be found around lines 123 - 134.

# Our Selected Paper
The source paper selected for this project can be found here:
> [Ditto: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/abs/2012.04221)

The base of out repository comes from a git clone of the authors solution:
> https://github.com/litian96/ditto
