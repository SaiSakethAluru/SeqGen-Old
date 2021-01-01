import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.transformer import Transformer
from src.dataset import MyDataset
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from torchviz import make_dot
from graphviz import Source

LABEL_LIST = ['background','objective','methods','results','conclusions']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)    ## debug: increase later
    parser.add_argument('--num_epochs',type=int,default=200)
    parser.add_argument('--lr',type=float,default=1e-3)
    # parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--max_par_len',type=int,default=20)    ## debug: 
    parser.add_argument('--max_seq_len',type=int,default=20)    ## debug:
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
        
    else:
        device = torch.device('cpu')
        print('using cpu')
    
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
        num_layers=1,   ## debug
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
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.1, patience=10, verbose=True
    # )
    
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

            # print('model net',make_dot(output))
            # print(make_dot(output))
            # make_arch = make_dot(output)
            # Source(make_arch).render('graph.png')
            # assert False
            ## output - N,par_len, num_labels --> N*par_len, num_labels
            output = output.reshape(-1,output.shape[2])
            ## target - 
            target = target[:,1:].reshape(-1)

            # print('output.shape',output.shape)
            # print('target.shape',target.shape)
            # print(f'{epoch} model params', list(model.parameters())[-1])
            # print('len params',len(list(model.parameters())))
            # print('trainable params: ',len(list(filter(lambda p: p.requires_grad, model.parameters()))))
            optimizer.zero_grad()

            loss = criterion(output,target)
            # loss.retain_grad()
            losses.append(loss.item())

            # print(f'{epoch} loss grads before', list(loss.grad)[-1])
            loss.backward()
            # print(f'{epoch} loss grads after', loss.grad)
            # print('model params')
            # count = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         print(p.grad,p.grad.norm())
            #         count +=1 
            # print(f'non none grads are {count}')
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)

            optimizer.step()
            # break #NOTE: break is there only for quick checking. Remove this for actual training.
            
        mean_loss = sum(losses)/len(losses)
        # scheduler.step(mean_loss)

        print(f"Mean loss for epoch {epoch} is {mean_loss}")
        # Validation
        model.eval()
        val_losses = []
        val_targets = []
        val_preds = []
        for batch_idx,batch in tqdm(enumerate(dev_generator)):
            inp_data,target = batch
            inp_data = inp_data.to(device)
            target = target.to(device)
            with torch.no_grad():
                output = model(inp_data,target[:,:-1])
                reshaped_output = output.reshape(-1,output.shape[2])
                reshaped_target = target[:,1:].reshape(-1)
                loss = criterion(reshaped_output,reshaped_target).item()
            val_losses.append(loss)
            flattened_target = target[:,1:].to('cpu').flatten()
            flattened_preds = torch.softmax(output,dim=-1).argmax(dim=-1).to('cpu').flatten()
            for target_i,pred_i in zip(flattened_target,flattened_preds):
                if target_i != 0:
                    val_targets.append(target_i)
                    val_preds.append(pred_i)
            # val_targets.append(target[:,1:].to('cpu').flatten())
            # output = torch.softmax(output,dim=-1).argmax(dim=-1)
            # val_preds.append(output.to('cpu').flatten())
            # break #NOTE: break is there only for quick checking. Remove this for actual training.

        loss = sum(val_losses) / len(val_losses)
        print(f"Validation loss at epoch {epoch} is {loss}")
        # val_targets = torch.cat(val_targets,dim=0)
        # val_preds = torch.cat(val_preds,dim=0)
        f1 = f1_score(val_targets,val_preds,average='macro')
        print(f'------Macro F1 score on dev set: {f1}------')
        if loss < best_val_loss:
            print(f"val loss less than previous best val loss of {best_val_loss}")
            best_val_loss = loss
            if args.save_model:
                dir_name = f"seed_{args.seed}_parlen_{args.max_par_len}_seqlen_{args.max_seq_len}.pt"
                output_path = os.path.join(args.save_path,dir_name)
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                print(f"Saving model to path {output_path}")
                torch.save(model,output_path)

        # Testing
        if epoch % args.test_interval == 0:
            model.eval()
            test_targets = []
            test_preds = []
            for batch_idx, batch in tqdm(enumerate(test_generator)):
                inp_data,target = batch
                inp_data = inp_data.to(device)
                target = target.to(device)
                with torch.no_grad():
                    output = model(inp_data,target[:,:-1])
                output = torch.softmax(output,dim=-1).argmax(dim=-1)
                flattened_target = target[:,1:].to('cpu').flatten()
                flattened_preds = output.to('cpu').flatten()
                for target_i,pred_i in zip(flattened_target,flattened_preds):
                    if target_i!=0:
                        test_targets.append(target_i)
                        test_preds.append(pred_i)
                # test_targets.append(target[:,1:].to('cpu').flatten())
                # test_preds.append(output.to('cpu').flatten())
                # break  #NOTE: break is there only for quick checking. Remove this for actual training. 
            
            # test_targets = torch.cat(test_targets,dim=0)
            # test_preds = torch.cat(test_preds,dim=0)
            # f1 = f1_score(target[:,1:].to('cpu').flatten(),output.to('cpu').flatten(),average='macro')
            f1 = f1_score(test_targets,test_preds,average='macro')
            print(f"------Macro F1 score on test set: {f1}------")



            


if __name__ == "__main__":

    args = get_args()
    train(args)