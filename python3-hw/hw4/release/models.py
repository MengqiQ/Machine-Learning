import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y, **kwargs):
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

    def fit(self, X, y, **kwargs):
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


class LambdaMeans(Model):

    def __init__(self, cluster_lambda,clustering_training_iterations ):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.lamdba = cluster_lambda
        self.iter = clustering_training_iterations
        self.u = None
        self.r = None
        self.k = None
        self.y = None
        pass

    def fit(self, X, _, **kwargs):
        """  Fit the lambda means model  """
        assert 'lambda0' in kwargs, 'Need a value for lambda'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        lambda0 = kwargs['lambda0']
        iterations = kwargs['iterations']
        # TODO: Write code to fit the model.  NOTE: labels should not be used here.
        row, column = X.shape
        self.u = np.mean(X, axis = 0)
        # np.vstack(self.u, )
        # u_1 = 1/row * np.sum(X, axis = 0)
        # lambda0 = 1/row * np.sum(np.linalg.norm(X - self.u[0]))
        lambda0 = np.mean(np.linalg.norm(X - self.u[0]))
        self.k = len(self.u)
        for iter in range(self.iter):
            for i in range(row):
                self.k = len(self.u)
                for k in range(self.k):
                    if np.min(np.linalg.norm(X[i,:] - self.u, axis = 1)) <= lambda0:
                        self.r[i,:] = np.zeros(len(self.u))
                        self.r[i,np.argmin(np.linalg.norm(X[i,:] - self.u), axis = 1)] = 1
                    else:
                        # self.k += 1
                        self.r[i,:] = np.zeros(len(self.u)+1)
                        self.r[i,len(self.u)] = 1
                        # self.u[self.k+1] = X[i,:]
                    self.u[k] = np.sum(np.dot(self.r,X[i,:]), axis = 1)/np.sum(self.r, axis = 1)
                np.vstack(self.u, X[i,:])
        self.k = len(self.u)

    def predict(self, X):
        # TODO: Write code to make predictions.
        row, column = X.shape
        for i in range(row):
            for k in range(self.k):
                self.y[i] = np.argmin(np.linalg.norm(X[i,:] - self.u), axis = 1)
        return self.y


class StochasticKMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.u = None
        self.r = None
        self.k = None
        self.y = None
        self.belta = None
        self.p = None
        self.c = 2
        pass

    def fit(self, X, _, **kwargs):
        assert 'num_clusters' in kwargs, 'Need the number of clusters (K)'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        num_clusters = kwargs['num_clusters']
        iterations = kwargs['iterations']
        # TODO: Write code to fit the model.  NOTE: labels should not be used here.
        X = X.toarray()
        row, column = X.shape
        # self.u = np.zeros([num_clusters, column])
        # self.u[0,:] = np.mean(X, axis=0).reshape(1, -1)
        # self.belta = np.zeros(row)
        # self.p = np.zeros([row, num_clusters])
        self.k = num_clusters
        if self.k == 1:
            self.u = np.mean(X, axis=0).reshape(1, -1)
        else:
            max = np.max(X, axis = 0)
            min = np.min(X, axis = 0)
            self.u = np.stack([(1 - i / (self.k - 1)) * min + (i / (self.k - 1)) * max for i in range(self.k)], axis = 0)
        for iter in range(iterations):
            iter += 1
            self.belta = self.c * iter
            self.p = np.zeros([self.k, row])
            for i in range(row):
                    dis = np.linalg.norm(X[i,:] - self.u, axis = 1)
                    d_hat = np.mean(dis)
                    # self.belta[i] = self.c * i
                    # denominator = np.sum(np.exp(- self.belta[i] * self.u[i,k]) / d_hat)
                    p = np.exp(- self.belta * dis / d_hat)
                    self.p[:,i] = p / np.sum(p)
                    # self.u[k,:] = np.sum(np.dot(self.p[i,k], X[i,:])) / np.sum(self.p[i,k])
            self.u = np.dot(self.p, X) / np.sum(self.p, axis = 1).reshape(self.k, -1)

    def predict(self, X):
        # TODO: Write code to make predictions.
        X = X.toarray()
        row, column = X.shape
        feature = min(X.shape[1], self.u.shape[1])
        self.y = np.zeros(row)
        for i in range(row):
            self.y[i] = np.argmin(np.linalg.norm(X[i, :feature] - self.u[:, :feature], axis=1))
        return self.y
        pass

