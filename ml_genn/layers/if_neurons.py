from pygenn.genn_model import create_custom_neuron_class
from ml_genn.layers.neurons import Neurons

if_model = create_custom_neuron_class(
    'if',
    var_name_types=[('Vmem', 'scalar'), ('nSpk', 'unsigned int')],
    extra_global_params=[('Vthr', 'scalar')],
    sim_code='''
    if ($(t) == 0.0) {
        // Reset state at t = 0
        $(Vmem) = 0.0;
        $(nSpk) = 0;
    }
    $(Vmem) += $(Isyn) * DT;
    ''',
    threshold_condition_code='''
    $(Vmem) >= $(Vthr)
    ''',
    reset_code='''
    $(Vmem) = 0.0;
    $(nSpk) += 1;
    ''',
    is_auto_refractory_required=False,
)

class IFNeurons(Neurons):

    def __init__(self, threshold=1.0):
        super(IFNeurons, self).__init__()
        self.threshold = threshold

    def compile(self, mlg_model, name, n):
        model = if_model
        vars = {'Vmem': 0.0, 'nSpk': 0}
        egp = {'Vthr': self.threshold}

        super(IFNeurons, self).compile(mlg_model, name, n, model, {}, vars, egp)

    def set_threshold(self, threshold):
        self.threshold = threshold

        if self.nrn is not None:
            self.nrn.extra_global_params['Vthr'].view[:] = threshold