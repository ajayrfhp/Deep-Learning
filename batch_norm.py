class BatchNormalizedLayer():
    """Layer implementing an batch normalization of its inputs.

    This layer is parameterised by a gamma vector and bias vector.
    """

    def __init__(self, input_dim, output_dim):
        """Initialises a parameterised batch normalization layer.

        Args:
            input_dim (int): Dimension of inputs to the layer.
            output_dim (int): Dimension of the layer outputs.
            Gamma and Biases may be initialized as required.
        """
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        #<Initialization routine for gamma and beta>
        self.cache = [] 

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x`, outputs `y`, gamma `gamma` and biases `beta` the layer
        corresponds to `y = gamma * (zero mean and unit variance x) + beta`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, input_dim).
        """
        mu = np.mean(inputs, axis = 0)
        sigma2 = np.var(inputs,axis=0)
        std = np.sqrt(sigma2 + 1e-8)
        diff = inputs - mu
        uhat = np.divide((inputs-mu),np.sqrt(std))
        self.cache = [std,diff,uhat]
        y = self.gamma * uhat + self.beta  
        return y

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        [std,diff,uhat] = self.cache
        N,D = inputs.shape
        duhat = grads_wrt_outputs * self.gamma
        dvar = -0.5 * np.divide(np.sum(duhat * uhat, axis =0), np.square(std))
        dmu = -1 * np.divide(np.sum(duhat, axis =0), std)
        #The second term in this gradient just sums to zero along every row and has been omitted
        grads_wrt_inputs = np.divide(duhat, std) + 2 * dvar * diff * (1.0 / N)  + dmu * (1.0 / N) 
        return grads_wrt_inputs

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim)

        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_weights, grads_wrt_biases]`.
        """
        grads_wrt_beta = np.sum( grads_wrt_outputs, axis = 0)
        grads_wrt_gamma = np.sum(grads_wrt_outputs * self.cache[2], axis=0)
        return [grads_wrt_gamma, grads_wrt_beta]

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        params_penalty = 0
        return params_penalty
    
    
    
    def params(self):
        """A list of layer parameter values: `[weights, biases]`."""
        return [self.gamma, self.beta]

    
    def params(self, values):
        self.gamma = values[0]
        self.beta = values[1]

    def __repr__(self):
        return 'BatchNormalizedLayer(input_dim={0}, output_dim={1})'.format(
            self.input_dim, self.output_dim)
