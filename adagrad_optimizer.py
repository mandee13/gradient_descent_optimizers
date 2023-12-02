import torch


class AdaGradOptimizer:

    def __init__(self, initial_lr, eps=1e-7, lr_schedule=None):
        super(AdaGradOptimizer, self).__init__()
        self.lr = initial_lr
        self.eps = eps
        self.lr_schedule = lr_schedule
        self.iteration = 0
        self.grad_accumulate_list = []

    def initialize_grad_accumulate(self, param_list):
        for _, param in enumerate(param_list):
            self.grad_accumulate_list.append(torch.zeros(size=param.shape))

    def update_param(self, param, grad, grad_accumulate):

        # obtain current step size
        if self.lr_schedule is not None:
            cur_lr = self.lr_schedule(self.iteration)
        else:
            cur_lr = self.lr

        # update accumulated gradients
        updated_grad_accumulate = torch.add(grad_accumulate, grad**2)

        # calculate current scale factor
        scale = torch.div(cur_lr, torch.sqrt(updated_grad_accumulate) + self.eps)

        # calculate current change values
        cur_grad = torch.mul(scale, grad)

        updated_param = torch.add(input=param, other=-cur_grad)

        return updated_param, updated_grad_accumulate

    def update_params(self, param_list, grad_list):

        assert len(param_list) == len(grad_list), 'param_list and grad_list need to have same lengths'

        # initialize gradient accumulate list before the first iteration
        if self.iteration == 0:
            self.initialize_grad_accumulate(param_list)

        # use gradients to update corresponding parameters and accumulated gradients
        updated_param_list = []
        updated_grad_accumulate_list = []
        for param, grad, grad_accumulate in zip(param_list, grad_list, self.grad_accumulate_list):
            updated_param, updated_grad_accumulate = self.update_param(param, grad, grad_accumulate)
            updated_param_list.append(updated_param)
            updated_grad_accumulate_list.append(updated_grad_accumulate)

        # update iteration and accumulated gradients
        self.iteration += 1
        self.grad_accumulate_list = updated_grad_accumulate_list

        return updated_param_list
