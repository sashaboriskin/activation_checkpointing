import torch
from torch.autograd import Function


class ActivationCheckpoint(Function):
    @staticmethod
    def forward(ctx, run_fn, *args, use_rng_state=True):
        ctx.run_fn = run_fn

        ctx.use_rng_state = use_rng_state
        ctx.device_type = "cuda"
        ctx.autocast_kwargs = dict(
            enabled=torch.is_autocast_enabled(ctx.device_type),
            dtype=torch.get_autocast_dtype(ctx.device_type),
            cache_enabled=torch.is_autocast_cache_enabled(),
        )

        # get rng state
        if use_rng_state:
            ctx.fwd_cpu_rng_state = torch.get_rng_state()
            ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()

        # tensor/not_tensor data
        ctx.tensor_idx = []
        ctx.nontensor_args = {}
        tensor_args = []
        for i, a in enumerate(args):
            if torch.is_tensor(a):
                ctx.tensor_idx.append(i)
                tensor_args.append(a)
            else:
                ctx.nontensor_args[i] = a

        # save only tensors for backward
        ctx.save_for_backward(*tensor_args)

        # forward without activations
        with torch.no_grad():
            out = run_fn(*args)
        return out

    @staticmethod
    def backward(ctx, *grad_out):

        saved = list(ctx.saved_tensors)
        args = []
        s_it = 0

        max_idx = max([*ctx.tensor_idx, *ctx.nontensor_args.keys()])
        for i in range(max_idx + 1):
            if i in ctx.nontensor_args:
                args.append(ctx.nontensor_args[i])
            else:
                args.append(saved[s_it])
                s_it += 1

        # set rng state
        if ctx.use_rng_state:
            torch.set_rng_state(ctx.fwd_cpu_rng_state)
            torch.cuda.set_rng_state(ctx.fwd_cuda_rng_state)

        # get the autocast_ctx
        autocast_ctx = torch.amp.autocast(
            device_type=ctx.device_type, **ctx.autocast_kwargs
        )

        # build a new torch graph
        with torch.enable_grad(), autocast_ctx:
            detached = []
            for a in args:
                if torch.is_tensor(a):
                    b = a.detach()
                    b.requires_grad = a.requires_grad
                    detached.append(b)
                else:
                    detached.append(a)

            # make a new forward
            out = (ctx.run_fn(*tuple(detached)),)

        # filter the outputs that require gradients
        outs_req = []
        grads_req = []
        for o, g in zip(out, grad_out):
            if torch.is_tensor(o) and o.requires_grad:
                outs_req.append(o)
                grads_req.append(g)

        # make a new backward
        torch.autograd.backward(outs_req, grads_req)

        # collect grads in the input format
        grad_inputs = []
        for a in detached:
            if torch.is_tensor(a) and a.requires_grad:
                grad_inputs.append(a.grad)
            else:
                grad_inputs.append(None)

        return (None, *grad_inputs)


def checkpoint(fn, *args, use_rng_state=True):
    return ActivationCheckpoint.apply(fn, *args, use_rng_state)
