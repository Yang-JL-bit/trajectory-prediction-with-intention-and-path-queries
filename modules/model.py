import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn import Sequential, ReLU
from torch.utils.data import DataLoader
from modules.feature_weighting import FeatureWeighting
from modules.a2a_interaction import A2A
from modules.early_stopping import EarlyStopping
from modules.LSTM import LSTM
from torch.utils.data.sampler import WeightedRandomSampler
import time


class RoadPredictionModel(nn.Module):
    def __init__(self, obs_len, input_size, hidden_size, num_layers, head_num, device = 'CPU') -> None:
        super(RoadPredictionModel, self).__init__()
        self.obs_len = obs_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_num = head_num
        self.device = device
        self.feature_weighting_target = FeatureWeighting(time_step=obs_len, feature_size=input_size)
        self.lstm_target = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.feature_weighting_surrounding = FeatureWeighting(time_step=obs_len, feature_size=input_size)
        self.lstm_surrounding = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.agent2agent = A2A(input_size=hidden_size, hidden_size=hidden_size, head_num=head_num)
        self.intention_prediction = Sequential(nn.Linear(hidden_size, 2 * hidden_size, dtype=torch.float64), ReLU(), nn.Linear(2 * hidden_size, 3, dtype=torch.float64), nn.Softmax(dim=-1))
    
    def forward(self, target_feature: torch.Tensor, surrounding_feature: torch.Tensor):
        bs = target_feature.shape[0]
        n_surr = surrounding_feature.shape[1]
        target_feature = self.feature_weighting_target(target_feature)
        surrounding_feature = self.feature_weighting_surrounding(surrounding_feature.flatten(0, 1)).reshape(bs, n_surr, self.obs_len, self.input_size)
        target_feature = self.lstm_target(target_feature)
        surrounding_feature = self.lstm_surrounding(surrounding_feature.flatten(0, 1)).reshape(bs, n_surr, self.hidden_size)
        target_feature = self.agent2agent(target_feature, surrounding_feature)
        intention_score = self.intention_prediction(target_feature)
        return intention_score
        

def cal_acc(pred, label):
    pred = torch.argmax(pred, dim=-1)
    acc = torch.sum(pred == label) / len(pred)
    return acc

def train_model(train_dataset, val_dataset, model: RoadPredictionModel, save_path, train_sample_weight, val_sample_weight, scalar, device = 'CPU', batch_size = 64, lr = 0.001, epoch = 100, patience = 0):
    train_sampler = WeightedRandomSampler(train_sample_weight, len(train_dataset), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss().to(device)
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=save_path)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    target_mean = scalar['target'][0].unsqueeze(0).unsqueeze(0).to(device)
    target_std = scalar['target'][1].unsqueeze(0).unsqueeze(0).to(device)
    surrounding_mean = scalar['surrounding'][0].unsqueeze(0).unsqueeze(2).to(device)
    surrounding_std = scalar['surrounding'][1].unsqueeze(0).unsqueeze(2).to(device)
    train_step = 0
    train_loss_list = []
    val_loss_list = []
    start_time = time.time()
    for i in range(epoch):
        print(f"--------------epoch {i + 1}--------------")
        #模型训练
        scheduler.step()
        model.train()
        for j, traj_data in enumerate(train_dataloader):
            target_feature = (traj_data['target_obs_traj'].to(device) - target_mean) / target_std
            surrounding_feature = (traj_data['surrounding_obs_traj'].to(device) - surrounding_mean) / surrounding_std
            lane_change_label = traj_data['lane_change_label'].to(torch.long).to(device)
            intention_score = model(target_feature, surrounding_feature)
            acc = cal_acc(intention_score, lane_change_label)
            loss = loss_fn(intention_score, lane_change_label)
            train_loss_list.append(loss.data.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_step % 10 == 0:
                print(f"train step: {train_step}, loss: {loss}, acc: {acc * 100 : .2f}%, time: {time.time() - start_time: .2f}s" )
            train_step += 1
        #模型验证
        val_loss, _ = val_model(val_dataset, model, val_sample_weight, scalar, device)
        val_loss_list.append(val_loss.data.cpu())
        early_stopping(val_loss, model)
        torch.save(train_loss_list, save_path + 'train_loss.pth')
        torch.save(val_loss_list, save_path + 'val_loss_list')
        if early_stopping.early_stop:
            print("early stopping!!")
            break
    print("训练完成！")

    
def test_model(test_dataset, model: RoadPredictionModel,  test_sample_weight, scalar, device = 'CPU', batch_size = 64):
    test_sampler = WeightedRandomSampler(test_sample_weight, len(test_dataset), replacement=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, sampler=test_sampler)
    target_mean = scalar['target'][0].unsqueeze(0).unsqueeze(0).to(device)
    target_std = scalar['target'][1].unsqueeze(0).unsqueeze(0).to(device)
    surrounding_mean = scalar['surrounding'][0].unsqueeze(0).unsqueeze(2).to(device)
    surrounding_std = scalar['surrounding'][1].unsqueeze(0).unsqueeze(2).to(device)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for traj_data in test_dataloader:
            target_feature = (traj_data['target_obs_traj'].to(device) - target_mean) / target_std
            surrounding_feature = (traj_data['surrounding_obs_traj'].to(device) - surrounding_mean) / surrounding_std
            lane_change_label = traj_data['lane_change_label'].to(torch.long).to(device)
            intention_score = model(target_feature, surrounding_feature)
            acc = cal_acc(intention_score, lane_change_label)
            correct += int(acc * len(intention_score))
            total += len(intention_score)
    test_accuracy = correct / total
    return test_accuracy



def val_model(val_dataset, model: RoadPredictionModel, val_sample_weight, scalar, device = 'CPU', batch_size = 64):
    val_sampler = WeightedRandomSampler(val_sample_weight, len(val_dataset), replacement=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    target_mean = scalar['target'][0].unsqueeze(0).unsqueeze(0).to(device)
    target_std = scalar['target'][1].unsqueeze(0).unsqueeze(0).to(device)
    surrounding_mean = scalar['surrounding'][0].unsqueeze(0).unsqueeze(2).to(device)
    surrounding_std = scalar['surrounding'][1].unsqueeze(0).unsqueeze(2).to(device)
    loss_fn = CrossEntropyLoss().to(device)
    correct = 0
    total = 0
    model.eval()
    val_loss_sum = torch.tensor(0.).to(device)
    with torch.no_grad():
        for i, traj_data in enumerate(val_dataloader):
            target_feature = (traj_data['target_obs_traj'].to(device) - target_mean) / target_std
            surrounding_feature = (traj_data['surrounding_obs_traj'].to(device) - surrounding_mean) / surrounding_std
            lane_change_label = traj_data['lane_change_label'].to(torch.long).to(device)
            intention_score = model(target_feature, surrounding_feature)
            acc = cal_acc(intention_score, lane_change_label)
            correct += int(acc * len(intention_score))
            total += len(intention_score)
            val_loss = loss_fn(intention_score, lane_change_label)
            val_loss_sum += val_loss * target_feature.shape[0]
    return val_loss_sum / len(val_dataset), correct / total    
    
    