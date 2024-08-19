'''This module implement the training loop for a torch model'''
from src.procedures.evaluation import evaluate
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(model,
          criteria,
          optimizer,
          train_dataloader,
          epochs,
          scheduler=None,
          device=None,
          evaluation_dataloader=None,
          early_stopping=False, 
          save_final_model=False,
          model_save_path=None, 
          use_tensorboard=False,
          tensorboard_log_dir=None,
          ):
    if not device :
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses = []
    batch_train_losses = []
    eval_losses = []
    eval_accuracies = []
    train_accuracies = []
    eval_losses = []
    
    if use_tensorboard:
        log_dir = tensorboard_log_dir if tensorboard_log_dir else 'runs'
        writer = SummaryWriter(log_dir=log_dir)
    acc = 0
    idx = 0
    best_eval_accuracy = -1
    for epoch in range(epochs):
        print('\n' + f'epoch: {epoch}')
        model.train()
        epoch_train_losses = []
        epoch_eval_losses = []
       
        # training
        for samples, targets in tqdm(train_dataloader, colour='blue'):
            samples, targets = samples.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions = model(samples)

            loss = criteria(predictions, targets)
            epoch_train_losses.append(loss.item())

            #tensorboard logging
            if use_tensorboard:
                writer.add_scalar('Train Loss (batches)', loss.item(), global_step=idx)
            idx += 1

            loss.backward()

            optimizer.step()

            #if acc >= 100:
            #    break
            #acc += 1
            

        batch_train_losses += epoch_train_losses
        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(epoch_train_loss)
        if use_tensorboard:
            writer.add_scalar('Train Loss (epochs)', epoch_train_loss, global_step=epoch)

        if scheduler:
            scheduler.step(epoch_train_loss)

        # evaluation on train split
        train_accuracy = evaluate(model=model,
                                            evaluation_dataloader=train_dataloader,
                                            device=device)
        train_accuracies.append(train_accuracy) 
        if use_tensorboard:
            writer.add_scalar('Train Accuracy [%] (epochs)', train_accuracy * 100, global_step=epoch)

        # evaluation on validation split
        if evaluation_dataloader:
            eval_accuracy, eval_loss = evaluate(model=model,
                                                evaluation_dataloader=evaluation_dataloader,
                                                criteria=criteria,
                                                device=device)
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                if early_stopping and model_save_path:
                    save_file_path = os.path.join(model_save_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_file_path)

            eval_accuracies.append(eval_accuracy) 
            eval_losses.append(eval_loss) 
            if use_tensorboard:
                writer.add_scalar('Validation Loss (epochs)', eval_loss, global_step=epoch)
                writer.add_scalar('Validation Accuracy [%] (epochs)', eval_accuracy * 100, global_step=epoch)

        #prints 
        print(f'  ---> train loss: {epoch_train_loss:.2f}')
        print(f'  ---> train accuracy: {train_accuracy * 100:.2f}%')
        if evaluation_dataloader:
            print(f'  ---> eval accuracy: {eval_accuracy * 100:.2f}%')
            print(f'  ---> eval loss: {eval_loss:.2f}')
    
    if save_final_model and model_save_path:
        save_file_path = os.path.join(model_save_path, 'final_model.pth')
        torch.save(model.state_dict(), save_file_path)

    if use_tensorboard:
        writer.close()
    return model, train_losses, eval_losses, train_accuracies, eval_accuracies  #, batch_train_losses