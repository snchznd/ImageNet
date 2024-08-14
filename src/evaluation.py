from src.metrics import compute_accuracy
import torch
from tqdm import tqdm

def evaluate(
        model,
        evaluation_dataloader,
        criteria=None,
        device=None
        ):
    model.eval()
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_losses = []
    accuracies = []
    with torch.no_grad():
        for samples, targets in tqdm(evaluation_dataloader, colour='green'):
            samples, targets = samples.to(device), targets.to(device)
            predictions = model(samples)
            if criteria:
                #loss = 0#criteria(predictions, targets).item()
                loss = criteria(predictions, targets).item()
                batch_losses.append(loss)
            accuracy = compute_accuracy(predictions, targets)
            accuracies.append(accuracy)
    
    avg_accuracy = sum(accuracies) / len(accuracies)

    if criteria:
        avg_loss = sum(batch_losses) / len(batch_losses)
        return avg_accuracy, avg_loss

    return avg_accuracy

