from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from utils.data import SentimentDataset
from utils.evaluate import get_metrics, metrics_to_string, print_CMatrix
from utils.forward import forward_model

from tqdm import tqdm
from os.path import join
import torch

import torch.nn.functional as F

file_dirpath = join('D:\\','Container','jupyter-notebook','experiment-X03ex')
test_file_path = join(file_dirpath, 'data', 'test_message.tsv')


class BERTModel(object):
    def __init__(self, model_id="", device=None, lr_rate=-1):
        """ Set up device """
        if device is not None:
            self.device = device
        else: raise Exception('No device was given')
        
        """ Set up model """
        if model_id != "":
            self.model, self.tokenizer = init_model(model_id)
        else: raise Exception('Require pretrained model ID')
        
        """ Set up optimizer """
        if lr_rate > 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_rate)
        else: raise Exception('Learning rate must be above 0')
            
    def set_train_mode(self, mode:bool):
        if mode == True: self.model.train()
        else: self.model.eval()
    
    def update(self, loss):    
        # Update model
        self.optimizer.zero_grad() # 1. clears x.grad for every parameter x in the optimizer,
                                   #    so PyTorch doesn't accumulate gradients on subsequent backward passes
        loss.backward()            # 2. computes dloss/dx for every parameter x which has requires_grad=True automatically
        self.optimizer.step()      # 3. updates the value of x using the gradient x.grad

def init_model(model_id):
    tokenizer = BertTokenizer.from_pretrained(model_id)
    config = BertConfig.from_pretrained(model_id)
    config.num_labels = SentimentDataset.num_labels

    model = BertForSequenceClassification.from_pretrained(model_id, config=config)
    return model, tokenizer
        
def evaluate_model(model, data_loader, i2w):
    model.set_train_mode(mode=False)
    
    val_loss_total = 0
    list_hyp, list_label = [], []

    loader_bar = tqdm(data_loader, leave=True, desc="valid_batch", bar_format='{desc}|{bar:30}|{percentage:3.0f}%')
    for i, batch_data in enumerate(loader_bar):    
        loss, batch_hyp, batch_label = forward_model(model.model, batch_data[:-1], i2w=i2w, device=model.device, use_adv_train=False)
        
        # Total loss
        val_loss = loss.item()
        val_loss_total += val_loss
        val_loss_avg = val_loss_total/(i+1)

        # Evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        loader_bar.set_description("[V] {:3d}/{:3d} > valid_loss:{:.4f}".format(i, len(data_loader), val_loss_avg))
        
    # Calculate metrics
    metrics = get_metrics(list_hyp, list_label)
    print("[V] Evaluation --> valid_loss:{:.4f} -- {}".format(val_loss_avg, metrics_to_string(metrics)))
    # Confusion Matrix
    print_CMatrix(test_file_path, i2w, model.model, model.tokenizer)
    
    return val_loss_avg, metrics
        
def train_model(model, train_loader, valid_loader, i2w, n_epochs,
                evaluate_every=1, early_stop=3,
                lr_decay=False, step_size=0.2, gamma=0.1, valid_criterion='F1'):
    
    if lr_decay:
        scheduler = StepLR(model.optimizer, step_size=step_size, gamma=gamma)
    best_val_metric = -1
    count_stop = 0
    
    # Releases all unoccupied cached memory currently held so that those can be used
    if str(model.device) == 'cuda': torch.cuda.empty_cache()
    
    for epoch in range(n_epochs):
        model.set_train_mode(mode=True)          # Set module in training mode, makes model keeps some layers
        torch.set_grad_enabled(True)             # Enable the autograd computation

        tr_loss_total = 0
        list_hyp, list_label = [], []
        
        loader_bar = tqdm(train_loader, leave=True, desc="train_batch", bar_format='{desc}|{bar:30}|{percentage:3.0f}%')
        for i, batch_data in enumerate(loader_bar):
            # Forward model
            """ Iterate data loader in train_loader to get batch_data and
                pass it to the forward function forward_sequence_classification """
            loss, batch_hyp, batch_label = forward_model(model.model, batch_data[:-1], i2w=i2w, device=model.device)
            model.update(loss)

            # Total loss
            tr_loss = loss.item()              # get current loss value
            tr_loss_total += tr_loss           # get sum of all loss value
            tr_loss_avg = tr_loss_total/(i+1)  # calculate average of the loss value

            # Training metrics
            list_hyp += batch_hyp
            list_label += batch_label
            loader_bar.set_description('[T] {:3d}/{:3d} > train_loss:{:.4f}'.format(i, len(train_loader), tr_loss_avg))
        
        # Calculate metrics
        metrics = get_metrics(list_hyp, list_label)
        print('[T] epoch: {:2d} --> train_loss:{:.4f} -- {}'.format(epoch+1, tr_loss_avg, metrics_to_string(metrics)))
        
        # Decay Learning Rate
        if lr_decay: scheduler.step()
            
        # evaluate the model
        if ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics = evaluate_model(model, valid_loader, i2w)
                                                   
            # Early stopping
            print("epoch:{:2d}".format(epoch+1))
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # save model
                torch.save(model.model.state_dict(), "best_model.pth")
                count_stop = 0
                print("{} = {} --> count_stop:{}/{} --> saved new model.".format(best_val_metric, val_metric, count_stop, early_stop))
            else:
                count_stop += 1
                print("{} > {} --> count_stop:{}/{}".format(best_val_metric, val_metric, count_stop, early_stop))
                if count_stop == early_stop: break