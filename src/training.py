'''This module implement the training loop for a torch model'''
from src.evaluation import evaluate
import torch
from tqdm import tqdm

def train(model,
          criteria,
          optimizer,
          train_dataloader,
          epochs,
          scheduler=None,
          device=None,
          early_stopping=False, # TODO: implement
          evaluation_dataloader=None,
          save_model=False, #TODO: implement
          model_save_path=None # TODO: implement
          ):
    if not device :
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses = []
    batch_train_losses = []
    eval_losses = []
    batch_eval_losses = []
    eval_accuracies = []
    eval_losses = []
    
    for epoch in range(epochs):
        print('\n' + f'epoch: {epoch}')
        model.train()
        epoch_train_losses = []
        epoch_eval_losses = []
       
        # training
        acc = 0
        for samples, targets in tqdm(train_dataloader, colour='blue'):
            samples, targets = samples.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions = model(samples)

            loss = criteria(predictions, targets)
            epoch_train_losses.append(loss.item())

            loss.backward()

            optimizer.step()

            #if acc == 500:
            #    break
            #acc += 1
            

        batch_train_losses += epoch_train_losses
        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(epoch_train_loss)

        # TODO: add train accuracy?

        if scheduler:
            scheduler.step(epoch_train_loss)

        # evaluation
        if evaluation_dataloader:
            eval_accuracy, eval_loss = evaluate(model=model,
                                                evaluation_dataloader=evaluation_dataloader,
                                                criteria=criteria,
                                                device=device)
            eval_accuracies.append(eval_accuracy) 
            eval_losses.append(eval_loss) 

        #prints 
        print(f'  ---> train loss: {epoch_train_loss:.2f}')
        if evaluation_dataloader:
            print(f'  ---> eval accuracy: {eval_accuracy * 100:.2f}%')
            print(f'  ---> eval loss: {eval_loss:.2f}')
    
    return model, train_losses, eval_accuracies, eval_losses  #, batch_train_losses