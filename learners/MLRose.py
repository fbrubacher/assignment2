import learners
import mlrose


class MLRoseLearner(learners.BaseLearner):
    def __init__(self,
                 verbose=False,
                 **kwargs):
        super().__init__(verbose)
        self._learner = mlrose.NeuralNetwork(hidden_nodes=[], activation='sigmoid',
                                             algorithm='random_hill_climb',
                                             max_iters=1000, bias=True, is_classifier=True,
                                             learning_rate=0.01, early_stopping=True,
                                             clip_max=5, max_attempts=100, random_state=3)

    def learner(self):
        return self._learner
