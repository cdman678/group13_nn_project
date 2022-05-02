import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy
import datetime

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba
from flearn.utils.language_utils import letter_to_vec, word_to_indices


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using global-regularized multi-task learning to Train')

        self.corrupt_input = False  # True = input data for corrupted clients will be altered
        self.corrupt_accuracy = []

        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        # self.inner_opt = tf.train.FtrlOptimizer(params['learning_rate'])

        super(Server, self).__init__(params, learner, dataset)

    def find_percentage_agreement(self, s1, s2):
        match_count = 0  # initialize counter to 0
        for idx, value in enumerate(s1):
            if s2[idx] == value:
                match_count += 1

        percentage_agreement = match_count / len(s1)

        return percentage_agreement

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        print("*"*10)
        print(f"Number of Clients Total: {len(self.clients)}")


        np.random.seed(1234567+self.seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
        print(corrupt_id)
        print("*" * 10)

        if self.dataset == 'shakespeare':
            for c in self.clients:
                c.train_data['y'], c.train_data['x'] = process_y(c.train_data['y']), process_x(c.train_data['x'])
                c.test_data['y'], c.test_data['x'] = process_y(c.test_data['y']), process_x(c.test_data['x'])

        batches = {}
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                if self.corrupt_input:
                    c.train_data['y'] = np.asarray(c.train_data['y'])
                    if self.dataset == 'celeba':
                        c.train_data['y'] = 1 - c.train_data['y']
                    elif self.dataset == 'femnist':
                        c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
                    elif self.dataset == 'shakespeare':
                        c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
                    elif self.dataset == "vehicle":
                        c.train_data['y'] = c.train_data['y'] * -1
                    elif self.dataset == "fmnist":
                        c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

            if self.dataset == 'celeba':
                # due to a different data storage format
                batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters)
            else:
                batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)


        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:
                tmp_models = []
                for idx in range(len(self.clients)):
                    tmp_models.append(self.local_models[idx])

                print("Time: ", datetime.datetime.now())

                num_train, num_correct_train, loss_vector = self.train_error(tmp_models)
                avg_train_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
                num_test, num_correct_test, _ = self.test(tmp_models)
                tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(num_train), avg_train_loss))
                tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
                non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
                tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
                tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
                print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))
                print("Corruption Detection Average accuracy: ", sum(self.corrupt_accuracy) / len(self.corrupt_accuracy))

            # weighted sampling
            indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

            raw_csolns = []
            losses = []
            avg_losses = []
            corrupt_validation = []

            for idx in indices:
                client_losses = []
                w_global_idx = copy.deepcopy(self.global_model)
                c = self.clients[idx]
                for _ in range(self.local_iters):
                    data_batch = next(batches[c])

                    # local
                    self.client_model.set_params(self.local_models[idx])
                    _, grads, _ = c.solve_sgd(data_batch)


                    if self.dynamic_lam:
                        upper_lambda = 40
                        step_size = 1
                        client_side_learning = .05

                        model_tmp = copy.deepcopy(self.local_models[idx])
                        model_best = copy.deepcopy(self.local_models[idx])
                        tmp_loss = 10000
                        # pick a lambda locally based on validation data
                        # x = [x / 10 for x in range(1, 21, 1)]  # increase time complexity to get better lambda
                        # [0.1, 1, 2]

                        for lam_id, candidate_lam in enumerate([x / 10 for x in range(1, upper_lambda, step_size)]):
                        # for lam_id, candidate_lam in enumerate([0.1, 1, 2]):  # Paper's Solution
                            for layer in range(len(grads[1])):
                                eff_grad = grads[1][layer] + candidate_lam * (self.local_models[idx][layer] - self.global_model[layer])
                                model_tmp[layer] = self.local_models[idx][layer] - client_side_learning * eff_grad

                                # Paper's Solution
                                # model_tmp[layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                            c.set_params(model_tmp)
                            l = c.get_val_loss()
                            if l < tmp_loss:
                                tmp_loss = l
                                model_best = copy.deepcopy(model_tmp)

                        self.local_models[idx] = copy.deepcopy(model_best)

                    else:
                        for layer in range(len(grads[1])):
                            eff_grad = grads[1][layer] + self.lam * (self.local_models[idx][layer] - self.global_model[layer])
                            self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                    # global
                    self.client_model.set_params(w_global_idx)
                    loss = c.get_loss()
                    losses.append(loss)
                    client_losses.append(loss)
                    _, grads, _ = c.solve_sgd(data_batch)
                    w_global_idx = self.client_model.get_params()

                # get the difference (global model updates)
                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]

                avg_losses.append(sum(client_losses)/self.local_iters)


                # send the malicious updates
                if idx in corrupt_id:
                    corrupt_validation.append(True)
                    if self.boosting:
                        # scale malicious updates
                        diff = [self.clients_per_round * u for u in diff]
                    elif self.random_updates:
                        # send random updates
                        stdev_ = get_stdev(diff)
                        diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]
                else:
                    corrupt_validation.append(False)

                if self.q == 0:
                    raw_csolns.append(diff)
                else:
                    raw_csolns.append((np.exp(self.q * loss), diff))

            # > Check Variables <

            # print("=="*10)
            # print("Number of Clients per round: ", self.clients_per_round)
            # print("Number of losses: ", len(avg_losses))
            # print("Number of diff: ", len(diff[0]))          # Weights
            # print("Number of csolns: ", len(raw_csolns))  # client's update?
            # print("Losses: ", avg_losses)

            # > FILTER OUT BAD ACCURACIES <

            # Outlier are < Q1−1.5×IQR (or) > Q3+1.5×IQR
            q1 = np.percentile(avg_losses, 25)
            q3 = np.percentile(avg_losses, 75)
            iqr = q3 - q1

            # outlier_bottom = q1 - (iqr*1.5)  # Low loss is a good thing (although could be over fitting)
            outlier_top = q3 + (iqr*1.5)

            loss_outliers = [x for x in avg_losses if x > outlier_top]
            loss_outliers_index = [True if x > outlier_top else False for x in avg_losses]
            # print("Loss outliers", loss_outliers)

            # > FILTER OUT BAD WEIGHTS <

            # The idea here is to flag clients as outliers for each layer
            # Then, if a client was flagged for X percent of the layers, we will label the client as bad
            # Weighted and random attacks are expected to lie outside of the expected range for the majority of layers

            outlier_counter = [0] * self.clients_per_round
            outlier_percent_threshold = .1  # Start with 10% - to move as parameter

            if self.dataset == "vehicle":
                # As Vehicle is an svm and not a cnn, the "layers" are different formats
                num_layers = len(raw_csolns[0][0])

                # Client #, ? (use 0), layer #
                for layer in range(num_layers):
                    layer_values = [raw_csolns[client][0][layer] for client in range(self.clients_per_round)]
                    tmp_q1 = np.percentile(layer_values, 25)
                    tmp_q3 = np.percentile(layer_values, 75)
                    tmp_iqr = tmp_q3 - tmp_q1
                    tmp_outlier_top = tmp_q3 + (tmp_iqr * 1.5)
                    # Identify which clients had outlier values for this layer
                    for clients_weight_index in range(len(layer_values)):
                        if layer_values[clients_weight_index] > tmp_outlier_top:
                            # We add +1 outlier count for the client
                            outlier_counter[clients_weight_index] += 1

            else:
                # To reduce time complexity of analyzing all weights in a deep NN,
                # We can aggregate all weights from each layer for every client
                # Then, we expect clean clients to product a similar absolute value weight
                # This primarily is targeted at boosted attacks, but random attacks are likely to fall as outsiders
                aggregated_weight_matrix = [[0]*len(raw_csolns)] * len(raw_csolns[0])

                for p in range(len(raw_csolns)):  # for each client
                    for tmp_i, v in enumerate(raw_csolns[p]):
                        tmp = sum(np.abs(v.astype(np.float64)))
                        while str(type(tmp)) != "<class 'numpy.float64'>":
                            tmp = sum(np.abs(tmp.astype(np.float64)))
                        aggregated_weight_matrix[tmp_i][p] = tmp  # the i-th layer, for client p, summed

                # At tis point, aggregated_weight_matrix is of length n, where n is the number of layers
                # Each index of aggregated_weight_matrix is of length c, where c is the number of clients
                # From here we can calculate similar to Vehicle
                num_layers = len(aggregated_weight_matrix)

                for sum_layer_weights in aggregated_weight_matrix:
                    tmp_q1 = np.percentile(sum_layer_weights, 25)
                    tmp_q3 = np.percentile(sum_layer_weights, 75)
                    tmp_iqr = tmp_q3 - tmp_q1
                    tmp_outlier_top = tmp_q3 + (tmp_iqr * 1.5)
                    # Identify which clients had outlier values for this layer
                    for clients_weight_index in range(len(sum_layer_weights)):
                        if sum_layer_weights[clients_weight_index] > tmp_outlier_top:
                            # We add +1 outlier count for the client
                            outlier_counter[clients_weight_index] += 1

            # Calculate which clients had x% or more flagged layers
            weight_outliers_index = []
            for client_outlier_sum in outlier_counter:
                if client_outlier_sum >= num_layers * outlier_percent_threshold:
                    weight_outliers_index.append(True)
                else:
                    weight_outliers_index.append(False)

            # > FILTER OUT BAD CLIENTS <

            outliers_index = []
            for o_index in range(len(loss_outliers_index)):
                if (loss_outliers_index[o_index] == True) or (weight_outliers_index[o_index] == True):
                    outliers_index.append(True)
                else:
                    outliers_index.append(False)

            # This is a metric we can use to validate our accuracy at capturing malicious attacks
            # print("out_index: ", outliers_index)
            # print("cor_index: ", corrupt_validation)

            self.corrupt_accuracy.append(self.find_percentage_agreement(outliers_index, corrupt_validation))

            # if outliers_index == corrupt_validation:
            #     print("-Success-"*5)

            # print("==" * 10)

            # > Remove Clients identified as corrupted <
            csolns = []
            for index in range(len(outliers_index)):
                # print(outliers_index[index])
                if not outliers_index[index]:
                    csolns.append(raw_csolns[index])

            # > ========================= <

            # csolns = raw_csolns

            if self.q != 0:
                avg_updates = self.aggregate(csolns)
            else:
                if self.gradient_clipping:
                    csolns = l2_clip(csolns)

                expected_num_mali = int(self.clients_per_round * self.num_corrupted / len(self.clients))

                if self.median:
                    avg_updates = self.median_average(csolns)
                elif self.k_norm:
                    avg_updates = self.k_norm_average(self.clients_per_round - expected_num_mali, csolns)
                elif self.krum:
                    avg_updates = self.krum_average(self.clients_per_round - expected_num_mali - 2, csolns)
                elif self.mkrum:
                    m = self.clients_per_round - expected_num_mali
                    avg_updates = self.mkrum_average(self.clients_per_round - expected_num_mali - 2, m, csolns)
                else:
                    # aggregate the weights (Malicious clients have already been filtered out)
                    avg_updates = self.simple_average(csolns)

            # update the global model with our aggregated and filtered weights
            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]

