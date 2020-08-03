from tensor_genn.layers.base_layer import BaseLayer


class Layer(BaseLayer):

    def __init__(self, name, model, params, vars_init, global_params):
        super(Layer, self).__init__(name, model, params, vars_init, global_params)


    def compile(self, tg_model):
        super(Layer, self).compile(tg_model)

        for connection in self.upstream_connections:
            connection.compile(tg_model)


    def connect(self, sources, connections):
        if len(sources) != len(connections):
            raise ValueError('sources list and connections list length mismatch')

        for source, connection in zip(sources, connections):
            connection.connect(source, self)


    def set_weights(self, weights):
        if len(weights) != len(self.upstream_connections):
            raise ValueError('weight matrix list and upsteam connection list length mismatch')

        for connection, w in zip(self.upstream_connections, weights):
            connection.set_weights(w)

    def get_weights(self):
        return [connection.get_weights() for connection in self.upstream_connections]


class IFLayer(Layer):

    def __init__(self, name, threshold=1.0):
        super(IFLayer, self).__init__(
            name, if_model, {}, {'Vmem': 0.0, 'nSpk': 0}, {'Vthr': threshold}
        )

        self.threshold = threshold


    def set_threshold(self, threshold):
        if not self.tg_model:
            raise RuntimeError('model must be compiled before calling set_threshold')

        for batch_i in range(self.tg_model.batch_size):
            nrn_name = '{}_nrn_{}'.format(self.name, batch_i)
            nrn = self.tg_model.g_model.neuron_populations[nrn_name]
            nrn.extra_global_params['Vthr'].view[:] = threshold

        self.threshold = threshold