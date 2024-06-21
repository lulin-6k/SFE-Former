import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
torch.autograd.set_detect_anomaly(True)
def train_class(epoch_num, train_data_set, test_data_set, batch_size=4, net=None, Loss=None, optimizer=None,
                is_use_gpu=True, model_id=None, eval_num=1, log_path='', model_save_path='', canshu1=None, canshu2=None):
    if is_use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    log_path = ''
    acc_best = 0
    acc_list = []
    for epoch in range(epoch_num):
        net.train()
        dataloader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=0)
        loss_epoch = torch.FloatTensor([0]).to(device)
        for idx, (X, Y) in tqdm(enumerate(dataloader)):
            X = X.to(device)
            Y = Y.to(device)
            out1, out2 = net(X)
            optimizer.zero_grad()
            loss = canshu1*Loss(out1, Y.view(1, -1)[0]) + canshu2*Loss(out2, Y.view(1, -1)[0])
            loss.backward()
            optimizer.step()
            loss_epoch += loss
        print('epoch:[{}/{}],loss:{}'.format(epoch, epoch_num, (loss_epoch / len(dataloader)).data[0]))
        log_file = open(log_path, mode='a+')
        log_file.write('[epoch{}/{}],loss:{}'.format(epoch, epoch_num, (loss_epoch / len(dataloader)).data[0]))
        log_file.write('\n')
        log_file.close()
        if epoch % eval_num == 0:
            acc = test_class(data_set=test_data_set, batch_size=batch_size, net=net, is_use_gpu=is_use_gpu,
                             model_id=model_id, log_path=log_path)
            acc_list.append(acc)
            if acc > acc_best:
                acc_best = acc
                torch.save(net.state_dict(), model_save_path)

def test_class(data_set, batch_size=1, net=None, is_use_gpu=True, model_id=None, log_path=''):
    if is_use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    net.eval()
    best_acc = 0
    dataloader_eval = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
    # print('start test,the len of dataloader is ', len(dataloader_eval))
    label_totals = []
    predict_total_1 = []
    predict_total_2 = []

    for idx, (X, Y) in tqdm(enumerate(dataloader_eval)):
        X = X.to(device)
        Y = Y.to(device)
        out1, out2 = net(X)

        out1 = torch.softmax(out1, dim=1)
        out2 = torch.softmax(out2, dim=1)

        max_indices_1 = torch.max(out1, dim=1)[1]
        max_indices_2 = torch.max(out2, dim=1)[1]

        label_totals.append(Y.cpu().detach().numpy())
        predict_total_1.append(max_indices_1.cpu().detach().numpy())
        predict_total_2.append(max_indices_2.cpu().detach().numpy())

    # Convert lists to numpy arrays
    label_totals = np.concatenate(label_totals, axis=0)
    predict_total_1 = np.concatenate(predict_total_1, axis=0)
    predict_total_2 = np.concatenate(predict_total_2, axis=0)

    # Calculate metrics for each output separately
    accuracy_1 = accuracy_score(label_totals, predict_total_1)
    precision_1 = precision_score(label_totals, predict_total_1, average='weighted')
    recall_1 = recall_score(label_totals, predict_total_1, average='weighted')
    f1_1 = f1_score(label_totals, predict_total_1, average='weighted')

    accuracy_2 = accuracy_score(label_totals, predict_total_2)
    precision_2 = precision_score(label_totals, predict_total_2, average='weighted')
    recall_2 = recall_score(label_totals, predict_total_2, average='weighted')
    f1_2 = f1_score(label_totals, predict_total_2, average='weighted')

    accuracy = (accuracy_1 + accuracy_2) / 2
    precision = (precision_1+precision_2) / 2
    recall = (recall_1+recall_2) / 2
    f1 = (f1_1+f1_2) / 2

    log_file = open(log_path, mode='a+')
    log_file.write('-------------------------------------\n')
    log_file.write('[accuracy]:{}\n'.format(accuracy))
    log_file.write('[precision]:{}\n'.format(precision))
    log_file.write('[recall]:{}\n'.format(recall))
    log_file.write('[f1-score ]:{}\n'.format(f1))
    log_file.write('-------------------------------------\n')
    log_file.close()

    return accuracy
