import torch


class XLAPatchedLinear(torch.autograd.Function):
    """
    Modified from https://pytorch.org/docs/stable/notes/extending.html#example
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        with torch.no_grad():
            return torch._C._nn.linear(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        input_dim = input.dim()
        if input_dim > 2:
            input_flat = input.flatten(start_dim=0, end_dim=-2)
            grad_output_flat = grad_output.flatten(start_dim=0, end_dim=-2)
        else:
            input_flat = input
            grad_output_flat = grad_output

        if ctx.needs_input_grad[0]:
            grad_input_flat = grad_output_flat.mm(weight)
            if input_dim > 2:
                grad_input = grad_input_flat.view(*input.size())
            else:
                grad_input = grad_input_flat
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_flat.t().mm(input_flat)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_flat.sum(0)

        return grad_input, grad_weight, grad_bias


def xla_patched_linear(input, weight, bias=None):
    return XLAPatchedLinear.apply(input, weight, bias)
