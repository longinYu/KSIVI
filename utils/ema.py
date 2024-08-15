import torch


class EMA:
    def __init__(self, beta, model_params):
        self.beta = beta
        # self.model_params = list(model_params)
        self.shadow_params = [p.clone().detach() for p in model_params]
        self.collected_params = []

    def update_params(self, model_parameters):
        for sp, mp in zip(self.shadow_params, model_parameters):
            sp.data = self.beta * sp.data + (1.0 - self.beta) * mp.data

    def apply_shadow(self, model_parameters):
        for sp, mp in zip(self.shadow_params, model_parameters):
            mp.data.copy_(sp.data)
    
    # for inference
    def store(self, model_parameters):
        self.collected_params = [param.clone() for param in model_parameters]

    def restore(self, model_parameters):
        for c_param, param in zip(self.collected_params, model_parameters):
            param.data.copy_(c_param.data)