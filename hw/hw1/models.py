import numpy as np
import classify

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.num_input_features = None
        # self.reference_example = None
        # pass

    def fit(self, X, y):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        pass

    def predict(self, X):
        # TODO: Write code to make predictions.
        num_examples, num_input_features = X.shape
        y = np.empty([num_examples], dtype=np.int)
        # for yi in range(num_examples):
        #     i = 0
        #     j = num_input_features - 1
        #     sum = 0
        #     while i < j:
        #         sum += X[yi, i] - X[yi, j]
        #         i += 1
        #         j -= 1
        #     if sum >= 0:
        #         y[yi] = 1
        #     else:
        #         y[yi] = 0
        h = num_input_features//2
        y = np.array([1 if (X[i,:h].sum() - X[i,-h:].sum() >=0) else 0 for i in range(num_examples)])
        return y
        pass



class Perceptron(Model):

    def __init__(self,n,iter):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.num_input_features = None
        self.num_example = None
        self.w = None
        self.newy = None
        self.n = n
        self.iter = iter
        pass

    def fit(self, X, y):
        # TODO: Write code to fit the model.
        self.num_examples, self.num_input_features = X.shape
        # w = np.empty(self.num_examples)
        self.newy = np.empty(self.num_examples)
        self.w = np.zeros(self.num_input_features)
        X = X.toarray()
        for iterator in range (self.iter):
            for i in range(0, self.num_examples):
                self.newy[i] = np.array([1 if np.dot(self.w, X[i,:]) >= 0 else 0])
                if self.newy[i] != y[i]:
                    self.w += self.n * X[i, :] * (-1 if y[i] == 0 else y[i])
        pass

    def predict(self, X):
        # TODO: Write code to make predictions.
        # newy = np.empty(self.num_examples)
        if self.num_input_features is None:
            raise Exception('fit not been called')
        yhat = np.empty(X.shape[0],dtype=np.int)
        X = X.toarray()
        new_features = min(X.shape[1], self.w.shape[0])
        for i in range(X.shape[0]):
            yhat[i] = 1 if np.dot(self.w[:new_features], X[i,:new_features]) >= 0 else 0

        return yhat
        pass


# TODO: Add other Models as necessary.
