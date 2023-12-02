import torch


class SGDOptimizer:

    def __init__(self, initial_lr, lr_schedule=None):
        super(SGDOptimizer, self).__init__()
        self.lr = initial_lr
        self.lr_schedule = None
        self.iteration = 0

    # Calculate
    def update_param(self, param, grad):

        # obtain current step size
        if self.lr_schedule is not None:
            cur_lr = self.lr_schedule(self.iteration)
        else:
            cur_lr = self.lr

        # update parameter
        updated_param = torch.add(input=param, other=-grad, alpha=cur_lr)

        return updated_param

    def update_params(self, param_list, grad_list):

        assert len(param_list) == len(grad_list), 'param_list and grad_list need to have same lengths'

        # use gradients to update corresponding parameters
        updated_param_list = []
        for param, grad in zip(param_list, grad_list):
            assert param.shape == grad.shape, 'parameter tensor and gradient tensor need to have same sizes'
            updated_param_list.append(self.update_param(param, grad))

        # update iteration
        self.iteration += 1

        return updated_param_list
