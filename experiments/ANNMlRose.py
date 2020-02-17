import numpy as np
import experiments
import learners

class MLRoseExperiment(experiments.BaseExperiment):

    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [], activation = 'sigmoid', 
        #                             algorithm = 'random_hill_climb', 
        #                             max_iters = 1000, bias = True, is_classifier = True, 
        #                             learning_rate = 0.0001, early_stopping = True, 
        #                             clip_max = 5, max_attempts = 100, random_state = 3)

        best_params = None

        print(self._details.ds_name)

        learner = learners.MLRoseLearner()
        if best_params is not None:
            learner.set_params(**best_params)

        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       learner, 'MLRose', 'MLRose', None,
                                       complexity_param=None, seed=self._details.seed,
                                       threads=self._details.threads,
                                       best_params=best_params,
                                       verbose=self._verbose)