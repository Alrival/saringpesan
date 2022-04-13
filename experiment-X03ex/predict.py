from os.path import join, isfile
import torch
from random import seed

from utils.model import BERTModel, init_model, train_model
from utils.data import SentimentDataset, SentimentDataLoader
from utils.evaluate import print_CMatrix
from utils.forward import forward_model
from utils.api import init_server

### Initial settings ###
seed(2143658709)
file_dirpath = join('D:\\','Container','jupyter-notebook','experiment-X03ex')
train_dataset_path = join(file_dirpath,'data','train_message.tsv')
valid_dataset_path = join(file_dirpath,'data','valid_message.tsv')
test_file_path = join(file_dirpath, 'data', 'test_message.tsv')

def generate_model(n_epochs=10,
                   learning_rate=1e-5):
    
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")

    model = BERTModel(model_id="indobenchmark/indobert-base-p2", device=device, lr_rate=learning_rate)
    w2i, i2w = SentimentDataset.label2index, SentimentDataset.index2label
    
    # Prepare Dataset
    train_dataset = SentimentDataset(train_dataset_path, model.tokenizer, lowercase=True)
    valid_dataset = SentimentDataset(valid_dataset_path, model.tokenizer, lowercase=True)

    train_loader = SentimentDataLoader(dataset=train_dataset, max_seq_len=128, batch_size=8, shuffle=True)
    valid_loader = SentimentDataLoader(dataset=valid_dataset, max_seq_len=128, batch_size=8, shuffle=False)
    
    train_model(model, train_loader, valid_loader, i2w, n_epochs)

def main():
    if not isfile('best_model.pth'):
        generate_model()
    
    # Load model
    model, tokenizer = init_model('indobenchmark/indobert-base-p1')
    model.load_state_dict(torch.load('best_model.pth'))
    w2i, i2w = SentimentDataset.label2index, SentimentDataset.index2label
    
    print_CMatrix(test_file_path, i2w, model, tokenizer)    
    init_server(model, tokenizer, i2w, w2i)
    
if __name__ == "__main__":
    main()