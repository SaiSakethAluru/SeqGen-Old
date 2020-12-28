import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.transformer import Transformer
from src.dataset import MyDataset
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score

LABEL_LIST = ['background','objective','methods','results','conclusions']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--num_epochs',type=int,default=20)
    parser.add_argument('--lr',type=float,default=1e-3)
    # parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--max_par_len',type=int,default=20)
    parser.add_argument('--max_seq_len',type=int,default=30)
    parser.add_argument('--train_data',type=str,default='data/train.txt')
    parser.add_argument('--dev_data',type=str,default='data/dev.txt')
    parser.add_argument('--test_data',type=str,default='data/test.txt')
    parser.add_argument('--embedding_path',type=str,default='data/glove.6B.100d.txt')
    parser.add_argument('--embed_size',type=int,default=100)
    parser.add_argument('--forward_expansion',type=int,default=4)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--save_model',type=bool,default=True)
    parser.add_argument('--save_path',type=str,default='models/')
    parser.add_argument('--load_model',type=bool,default=False)
    parser.add_argument('--load_path',type=str,default='model/')
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--test_interval',type=int,default=1)
    args = parser.parse_args()
    return args

def train(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("using gpu: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
        print('using cpu')
        torch.manual_seed(args.seed)
    
    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": False
        }
    dev_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False
        }
    test_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False
        }

    training_set = MyDataset(args.train_data,args.embedding_path,LABEL_LIST,args.max_par_len,args.max_seq_len)
    training_generator = DataLoader(training_set, **training_params)

    dev_set = MyDataset(args.dev_data,args.embedding_path,LABEL_LIST,args.max_par_len,args.max_seq_len)
    dev_generator = DataLoader(dev_set,**dev_params)

    test_set = MyDataset(args.test_data,args.embedding_path,LABEL_LIST,args.max_par_len,args.max_seq_len)
    test_generator = DataLoader(test_set,**test_params)

    src_pad_idx = 0
    trg_pad_idx = 0
    model = Transformer(
        label_list=LABEL_LIST,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=100,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.5,
        device=device,
        max_par_len=args.max_par_len,
        max_seq_len=args.max_seq_len,
        embed_path=args.embedding_path
    )
    model = model.to(device).double()
    
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    
    epoch_losses = []
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        print(f"----------------[Epoch {epoch} / {args.num_epochs}]-----------------------")

        losses = []
        for batch_idx,batch in tqdm(enumerate(training_generator)):
            # print('batch',batch)
            # print('type of batch',type(batch))
            inp_data,target = batch
            # print('inp_data',inp_data)
            # print('type(inp_data)',type(inp_data))
            # print('target',target)
            # print('type(target)',type(target))
            # print('target.shape',target.shape)
            inp_data = inp_data.to(device)
            # print('inp_data.shape',inp_data.shape)
            target = target.to(device)
            # assert False

            output = model(inp_data,target[:,:-1])
            output = output.reshape(-1,output.shape[2])
            target = target[:,1:].reshape(-1)

            print('output.shape',output.shape)
            print('target.shape',target.shape)
            
            optimizer.zero_grad()

            loss = criterion(output,target)
            losses.append(loss.item())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)

            optimizer.step()

        mean_loss = sum(losses)/len(losses)
        scheduler.step(mean_loss)

        print(f"Mean loss for epoch {epoch} is {mean_loss}")
        # Validation
        model.eval()
        for batch_idx,batch in tqdm(enumerate(dev_generator)):
            inp_data,target = batch
            inp_data.to(device)
            target.to(device)
            with torch.no_grad():
                output = model(inp_data,target[:,:-1])
                reshaped_output = output.reshape(-1,output.shape[2])
                reshaped_target = target[:,1:].reshape(-1)
                loss = criterion(output,target).item()
                print(f"Validation loss at epoch {epoch} is {loss}")
            output = torch.softmax(output,dim=-1).argmax(dim=-1)
            f1 = f1_score(target.to('cpu').flatten(),output.to('cpu').flatten(),average='macro')
            print(f'------Macro F1 score on dev set: {f1}------')
            if loss < best_val_loss:
                best_val_loss = loss
                print(f"val loss less than previous best val loss of {best_val_loss}")
                if args.save_model:
                    output_path = args.save_path + '_' + args.seed
                    print(f"Saving model to path {output_path}")
                    torch.save(model,output_path)


        # Testing
        if epoch % args.test_interval == 0:
            model.eval()
            loss_ls = []
            for batch_idx, batch in tqdm(enumerate(test_generator)):
                inp_data,target = batch
                inp_data.to(device)
                target.to(device)
                with torch.no_grad():
                    output = model(inp_data,target[:,:-1])
                output = torch.softmax(output,dim=-1).argmax(dim=-1)
                f1 = f1_score(target.to('cpu').flatten(),output.to('cpu').flatten(),average='macro')
                print(f"------Macro F1 score on test set: {f1}------")



            


if __name__ == "__main__":
    args = get_args()
    train(args)