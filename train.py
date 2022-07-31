import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer

from dataloader import TextDataset, BatchTextCall
from model import MultiClass
from utils import load_config


def choose_bert_type(path, bert_type="tiny_albert"):
    """
    choose bert type for chinese, tiny_albert or macbert（bert）
    return: tokenizer, model
    """

    if bert_type == "albert":
        model_config = BertConfig.from_pretrained(path)
        model = AlbertModel.from_pretrained(path, config=model_config)
    elif bert_type == "bert" or bert_type == "roberta":
        model_config = BertConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, config=model_config)
    elif bert_type == "smartbert":
        print('start loading smartbert model......')
        model_config = BertConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, config=model_config)
    else:
        model_config, model = None, None
        print("ERROR, not choose model!")

    return model_config, model


def evaluation(model, test_dataloader, loss_func, label2ind_dict, save_path, valid_or_test="test"):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for ind, (token, segment, mask, label) in enumerate(test_dataloader):
        token = token.cuda()
        segment = segment.cuda()
        mask = mask.cuda()
        label = label.cuda()

        out = model(token, segment, mask)   
        loss = loss_func(out, label)
        
        total_loss += loss.detach().item()

        label = label.data.cpu().numpy()
        predic = torch.max(out.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    
    if valid_or_test == "test":
        report = metrics.classification_report(labels_all, predict_all, target_names=label2ind_dict.keys(), digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, total_loss / len(test_dataloader), report, confusion
    return acc, total_loss / len(test_dataloader)




def test_evaluation(model, test_dataloader, loss_func, label2ind_dict, save_path, valid_or_test="test"):
    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    stack = {}
    
    index = 0 
    for ind, (token, segment, mask, label) in enumerate(test_dataloader):
        token = token.cuda()
        segment = segment.cuda()
        mask = mask.cuda()
        label = label.cuda()

        out = model(token, segment, mask)
        
        label = label.data.cpu().numpy()

        predict = torch.max(out.data, 1)[1].cpu().numpy()
    
#         print('predict:',predict)
#         print('label:',label)
        
        if label[0] not in stack.keys():
            items = []
            items.append(predict[0])
            stack[label[0]] = items
        else:
            stack[label[0]].append(predict[0])
        
        print('index: ',index)
        
        index += 1
    
    
    with open('stack_ir.txt','a') as fw:
        print(stack)
        fw.write(str(stack))



def train(config):

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.backends.cudnn.benchmark = True
        
    print('start loading tokenzier.....')
        
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_path)

    print('start loading dataset.....')
    train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len)

    train_dataset = TextDataset(os.path.join(config.data_dir, "train.txt"))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                  collate_fn=train_dataset_call)

    model_config, bert_encode_model = choose_bert_type(config.pretrain_path, bert_type=config.bert_type)
    
    multi_classification_model = MultiClass(bert_encode_model, model_config,
                                            num_classes=7, pooling_type=config.pooling_type)
    multi_classification_model.cuda()
    
    
    num_train_optimization_steps = len(train_dataloader) * config.epoch
    param_optimizer = list(multi_classification_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=config.lr, correct_bias=not config.bertadam)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             int(num_train_optimization_steps * config.warmup_proportion),
                                                             num_train_optimization_steps)
    loss_func = F.cross_entropy

    loss_total, top_acc = [], 0
    for epoch in range(config.epoch):
        multi_classification_model.train()
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))
        for i, (token, segment, mask, label) in enumerate(tqdm_bar):
            token = token.cuda()
            segment = segment.cuda()
            mask = mask.cuda()
            label = label.cuda()
     
            multi_classification_model.zero_grad()
            out = multi_classification_model(token, segment, mask)
       
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())

            
            
        with open('logs/finetune_evmcode_ep200_logs.txt','a') as fw:
            log_str = "Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time)
            fw.write(log_str+'\n')
            print(log_str)            
            
        time.sleep(1)
    
    torch.save(multi_classification_model.state_dict(), config.save_path)

        
        
        
def evaluate_classifier(config):
    label2ind_dict = {'TOD': 0, 'Over&Under-flow': 1, 'Re-entrancy': 2, 'Timestamp-Dependency': 3, 'Tx.origin': 4, 'Uncheck-Send': 5, 'Unhandle-Exception':6}
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.backends.cudnn.benchmark = True
        
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_path)
    train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len)
        
    valid_dataset = TextDataset(os.path.join(config.data_dir, "dev.txt"))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                  collate_fn=train_dataset_call)

    
    
    model_config, bert_encode_model = choose_bert_type(config.pretrain_path, bert_type=config.bert_type)
    
    multi_classification_model = MultiClass(bert_encode_model, model_config,
                                            num_classes=7, pooling_type=config.pooling_type)
    multi_classification_model.cuda()
    
    multi_classification_model.load_state_dict(torch.load(config.save_path))
    
    loss_func = F.cross_entropy
    
    acc, loss, report, confusion = evaluation(multi_classification_model,
                                                  valid_dataloader, loss_func, label2ind_dict,
                                                  config.save_path)
    
    print("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
    print("Report:\n",report)
    print("Confusion:\n",confusion)
    
    
    
    
    pass


def test_classifier(config):
    label2ind_dict = {'TOD': 0, 'Over&Under-flow': 1, 'Re-entrancy': 2, 'Timestamp-Dependency': 3, 'Tx.origin': 4, 'Uncheck-Send': 5, 'Unhandle-Exception':6}
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.backends.cudnn.benchmark = True
        
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_path)
    train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len)
        
    test_dataset = TextDataset(os.path.join(config.data_dir, "test.txt"))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=10,
                                  collate_fn=train_dataset_call)

    
    
    model_config, bert_encode_model = choose_bert_type(config.pretrain_path, bert_type=config.bert_type)
    
    multi_classification_model = MultiClass(bert_encode_model, model_config,
                                            num_classes=7, pooling_type=config.pooling_type)
    multi_classification_model.cuda()
    
    multi_classification_model.load_state_dict(torch.load(config.save_path))
    
    loss_func = F.cross_entropy
    
    test_evaluation(multi_classification_model,test_dataloader, loss_func, label2ind_dict,
                                                  config.save_path)
    
#     print("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
#     print("Report:\n",report)
#     print("Confusion:\n",confusion)
    
    
    
    
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    print(type(config.lr), type(config.batch_size))
    config.lr = float(config.lr)
    
#     train(config)
    
#     evaluate_classifier(config)

    test_classifier(config)
