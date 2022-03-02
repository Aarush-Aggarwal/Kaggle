from multiprocessing.spawn import import_main_path
import torch
import pandas as pd
import numpy as np
from zmq import device
import config
from tqdm import tqdm


def make_predictions(model, loader, output_csv="submission.csv"):
    preds = []
    filenames = []
    model.eval()
    
    for data, label, files in tqdm(loader):
        data = data.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        with torch.no_grad():
            pred = model(data).argmax(1)
            preds.append(pred.cpu().numpy())
            filenames += files
    
    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()


def check_accuracy(loader, model, device="cpu"):
    model.eval()
    
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0
    
    for data, labels, _ in tqdm(loader):
        data = data.to(config.DEVICE)
        data = data.to(config.DEVICE)
        
        with torch.no_grad():
            scores = model(data)
            
            predictions = scores.argmax(1)
            
            num_correct += (predictions == labels).sum()
            num_samples += predictions.shape[0] 
            
            # add to lists
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
        print(f"Got {num_correct} / {num_samples} with accuracy { float(num_correct)/float(num_samples) } * 100:.2f")
        
        model.train()
        
        # return value to be sent to sklearn function in train.py
        return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(all_labels, axis=0, dtype=np.int64)
    
    
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # New Learning Rate for new Checkpoint (so that we do not use Learning Rate of old checkpoint)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr