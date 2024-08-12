import math

def compute_conv_output_size(h_in, w_in, kernel_size, stride, padding, dilation):
    def numerator(dim_in, idx):
        return dim_in + 2 * padding[idx] - dilation[idx] * (kernel_size[idx] - 1) - 1

    def denominator(idx):
        return stride[idx]

    out_dims = {}
    for dim_name, dim_in, idx in [('H_out', h_in, 0), ('W_out', w_in, 1)]:
        expr = numerator(dim_in, idx) / denominator(idx) + 1
        out_dims[dim_name] = math.floor(expr)
    return out_dims