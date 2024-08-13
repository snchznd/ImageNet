'''This module implement the training loop for a torch model'''
import torch
from tqdm import tqdm

def train(model,
          criteria,
          optimizer,
          train_dataloader,
          epochs,
          scheduler=None,
          device=None,
          early_stopping=False,
          evaluation_dataloader=None,
          evaluation_function=None,
          save_model=False,
          model_save_path=None
          ):
    if not device :
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    perform_evaluation = evaluation_dataloader and evaluation_function
    train_losses = []
    batch_train_losses = []
    eval_losses = []
    batch_eval_losses = []
    for epoch in range(epochs):
        epoch_train_losses = []
        epoch_eval_losses = []
       
        # training
        acc = 0
        for samples, targets in tqdm(train_dataloader, colour='green'):
            samples, targets = samples.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions = model(samples)

            loss = criteria(predictions, targets)
            epoch_train_losses.append(loss.item())

            loss.backward()

            optimizer.step()

            #if acc == 10_000:
            #    break
            #acc += 1
            

        batch_train_losses += epoch_train_losses
        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(epoch_train_loss)

        if scheduler:
            scheduler.step(epoch_train_loss)

        # evaluation
        # ...
    
    return model, train_losses, batch_train_losses