'''Module that contains implementations of all the metrics used.'''
import torch

def compute_accuracy(model_output : torch.Tensor, gt : torch.Tensor) -> float:
    '''
    Computes the accuracy of a prediction w.r.t a gt.

    Args:
        model_output (torch.Tensor): the probability distributions over the
            classes predicted by the model for each sample of the batch. This
            tensor should be of size (BxC), where B is the batch size and C is
            the number of classes in the gt.
        gt (torch.Tensor): of size (Bx1) and containing the gt class for each
            element of the batch.
        
    Returns:
        float: the accuracy.

    '''
    nbr_predictions = model_output.shape[0]
    model_predictions = torch.argmax(model_output, axis=1)
    nbr_correct_predictions = (model_predictions == gt.squeeze()).sum()
    return (nbr_correct_predictions / nbr_predictions).item()
