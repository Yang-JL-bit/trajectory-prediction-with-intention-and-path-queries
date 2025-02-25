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
from modules.trajectory_generator import TrajectoryDecoder, Time2Centerline
from modules.loss_fn import loss_fn_traj
from modules.metrics import cal_traj_acc, cal_intention_acc, cal_minADE, cal_minFDE, cal_miss_rate
from modules.plot import visualization
import time
import random


class RoadPredictionModel(nn.Module):
    def __init__(self, obs_len, pred_len, input_size, hidden_size, num_layers, head_num, style_size, device, 
                 decoder_size, inputembedding_size, predict_trajectory = False, refinement_num = 5, 
                 top_k = 6, dt = 0.2, use_traj_prior = False, use_endpoint_prior = False) -> None:
        super(RoadPredictionModel, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_num = head_num
        self.style_size = style_size
        self.predict_trajectory = predict_trajectory
        self.inputembedding_size = inputembedding_size
        self.refinement_num = refinement_num
        self.top_k = top_k
        self.device = device
        self.dt = dt
        self.feature_weighting_target = FeatureWeighting(time_step=obs_len, feature_size=input_size, inputembedding_size=inputembedding_size)
        self.lstm_target = LSTM(input_size=inputembedding_size, hidden_size=hidden_size, num_layers=num_layers, device=device)
        self.feature_weighting_surrounding = FeatureWeighting(time_step=obs_len, feature_size=input_size, inputembedding_size=inputembedding_size)
        self.lstm_surrounding = LSTM(input_size=inputembedding_size, hidden_size=hidden_size, num_layers=num_layers, device=device)
        #compare
        # self.lstm_target = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, device=device)
        # self.lstm_surrounding = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, device=device)
        
        self.agent2agent = A2A(input_size=hidden_size, hidden_size=hidden_size, head_num=head_num)
        self.intention_prediction = Sequential(nn.Linear(hidden_size, 3, dtype=torch.float64))

        self.time2centerline = Time2Centerline(input_dim=hidden_size + 3 + self.style_size, driving_style_hidden_size= style_size, hidden_dim=decoder_size, n_predict=top_k)

        #compare
        self.agent2agent_fc = nn.Linear(9 * hidden_size, hidden_size, dtype=torch.float64)
        self.trajectory_generator = nn.LSTM(input_size=hidden_size + 3, hidden_size=decoder_size, num_layers=num_layers, batch_first=True, dtype=torch.float64)

        self.trajectory_decoder_left = TrajectoryDecoder(input_size=hidden_size + 3 + style_size, driving_style_hidden_size=style_size, hidden_size=decoder_size, num_layers=num_layers, n_predict=top_k, use_traj_prior=use_traj_prior, use_endpoint_prior=use_endpoint_prior)
        self.trajectory_decoder_keep = TrajectoryDecoder(input_size=hidden_size + 3 + style_size, driving_style_hidden_size=style_size, hidden_size=decoder_size, num_layers=num_layers, n_predict=top_k, use_traj_prior=use_traj_prior, use_endpoint_prior=use_endpoint_prior)
        self.trajectory_decoder_right = TrajectoryDecoder(input_size=hidden_size + 3 + style_size, driving_style_hidden_size=style_size, hidden_size=decoder_size, num_layers=num_layers, n_predict=top_k, use_traj_prior=use_traj_prior, use_endpoint_prior=use_endpoint_prior)
    
    
    #直接使用LSTM解码
    def forward(self, target_feature: torch.Tensor, surrounding_feature: torch.Tensor,
               origin_feature: torch.Tensor, centerline_info: torch.Tensor, driving_style_prior = None):
        bs = target_feature.shape[0]
        n_surr = surrounding_feature.shape[1]
        target_feature = self.feature_weighting_target(target_feature)
        surrounding_feature = self.feature_weighting_surrounding(surrounding_feature.flatten(0, 1)).view(bs, n_surr, self.obs_len, self.inputembedding_size)
        target_feature = self.lstm_target(target_feature)
        surrounding_feature = self.lstm_surrounding(surrounding_feature.flatten(0, 1)).view(bs, n_surr, self.hidden_size)
        target_feature = self.agent2agent(target_feature, surrounding_feature)
        
        #compare
        # target_feature = self.agent2agent_fc(torch.cat([target_feature, surrounding_feature.flatten(1,2)], dim=-1))

        intention_score = self.intention_prediction(target_feature)  #(bs, 3)
        if self.predict_trajectory:
            # 生成形状为 (bs, 3) 的意图向量，直接在目标设备上创建
            intentions = torch.eye(3, device=target_feature.device).unsqueeze(0).expand(bs, -1, -1)  # (bs, 3, 3)
            #driving_style_prior
            if driving_style_prior is not None:
                driving_style_left = torch.stack(driving_style_prior['left_turn'], dim=0).to(target_feature.device)
                driving_style_right = torch.stack(driving_style_prior['right_turn'], dim=0).to(target_feature.device)
                driving_style_keep = torch.stack(driving_style_prior['straight'], dim=0).to(target_feature.device)
            else:
                driving_style_left = driving_style_right = driving_style_keep = None
            trajectory_pred_left, confidence_left, endpoint_left = self.trajectory_decoder_left(target_feature, 
                                                                            intentions[:, 0, :], 
                                                                            self.pred_len,
                                                                            driving_style_left)
            trajectory_pred_keep, confidence_cur, endpoint_keep = self.trajectory_decoder_keep(target_feature, 
                                                                            intentions[:, 1, :], 
                                                                            self.pred_len,
                                                                            driving_style_keep)
            trajectory_pred_right, confidence_right, endpoint_right = self.trajectory_decoder_right(target_feature, 
                                                                            intentions[:, 2, :], 
                                                                            self.pred_len,
                                                                            driving_style_right)
            conbined_confidence = torch.stack([confidence_cur, confidence_left, confidence_right], dim=1)
            intention_weight = intention_score.unsqueeze(-1) #(bs, 3, 1)
            combined_weighted_confidence = intention_weight * conbined_confidence  #(bs, 3, 6)
            combined_weighted_confidence = combined_weighted_confidence.view(bs, -1) #(bs, 18)
            combined_trajectory = torch.cat([trajectory_pred_keep, trajectory_pred_left, trajectory_pred_right], dim=1)   #(bs, 18, n_pred, 2)
            combined_endpoint = torch.cat([endpoint_keep, endpoint_left, endpoint_right], dim=1)
            topk_scores, topk_indices = torch.topk(combined_weighted_confidence, self.top_k, dim=1)  # (bs, top_k)
            batch_indices = torch.arange(bs).unsqueeze(-1).expand(-1, self.top_k)  # (bs, top_k)
            selected_trajectories = combined_trajectory[batch_indices, topk_indices]  # (bs, k, n_pred, 2)
            selected_endpoint = combined_endpoint[batch_indices, topk_indices]
            return intention_score,  topk_scores, selected_trajectories, selected_endpoint
        return intention_score
            
            


def train_model(train_dataset, val_dataset, model: RoadPredictionModel, save_path, scalar, 
                device, predict_trajectory = True, driving_style_prior = None,batch_size = 64, 
                lr = 0.01, epoch = 100, patience = 0, top_k = 6, decay_rate = 0.5, decay_step = 5, checkpoint = None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=save_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_step, gamma=decay_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 50, 100], gamma=0.1)
    target_mean = scalar['target'][0].unsqueeze(0).unsqueeze(0).to(device)
    target_std = scalar['target'][1].unsqueeze(0).unsqueeze(0).to(device)
    surrounding_mean = scalar['surrounding'][0].unsqueeze(0).unsqueeze(2).to(device)
    surrounding_std = scalar['surrounding'][1].unsqueeze(0).unsqueeze(2).to(device)
    train_step = 0
    train_loss_list = []
    #加载checkpoint
    if checkpoint is not None:
        print("loading checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=checkpoint['epoch'])
        train_loss_list = torch.load(save_path + 'train_loss_list.pth')
    start_time = time.time()
    for i in range(epoch):
        print(f"--------------epoch {i + 1}--------------")
        #模型训练
        model.train()
        for j, traj_data in enumerate(train_dataloader):
            target_feature = (traj_data['target_obs_traj'].to(device) - target_mean) / target_std
            surrounding_feature = (traj_data['surrounding_obs_traj'].to(device) - surrounding_mean) / surrounding_std
            lane_change_label = traj_data['lane_change_label'].to(torch.long).to(device)
            origin_feature = traj_data['origin_feature'].to(device)
            centerline_info = traj_data['centerline_info'].to(device)
            candidate_trajectory_mask = torch.ones(target_feature.shape[0], top_k).to(device)
            future_trajectory_gt = traj_data['future_traj_gt'].to(device)
            if not predict_trajectory:
                intention_score = model(target_feature, surrounding_feature, origin_feature, centerline_info, driving_style_prior)
                acc = cal_intention_acc(intention_score, lane_change_label)
                loss = loss_fn(intention_score, lane_change_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if train_step % 10 == 0:
                    train_loss_list.append({'total_loss': loss.item(),
                        'intention_acc': acc,
                        })
                    print(f"train step: {train_step}, loss: {loss}, acc: {acc * 100 : .2f}%, time: {time.time() - start_time: .2f}s" )
                train_step += 1
            else:
                intention_score, weighted_trajectory_score, candidate_trajectory, endpoint = model(target_feature, surrounding_feature, 
                                                                                         origin_feature, centerline_info, driving_style_prior)
                traj_acc = cal_traj_acc(weighted_trajectory_score, candidate_trajectory, future_trajectory_gt, candidate_trajectory_mask)
                intention_acc = cal_intention_acc(intention_score, lane_change_label)
                loss, loss_intention_cls, loss_traj_cls, loss_traj_reg, endpoint_loss = loss_fn_traj(intention_score, lane_change_label, weighted_trajectory_score, 
                                                                                      future_trajectory_gt, candidate_trajectory, candidate_trajectory_mask, 
                                                                                      endpoint, device=device, alpha=1.0, beta=1.0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if train_step % 10 == 0:
                    train_loss_list.append({'total_loss': loss.item(),
                        'intention_cls_loss': loss_intention_cls.item(),
                        'traj_cls_loss': loss_traj_cls.item(),
                        'traj_reg_loss': loss_traj_reg.item(),
                        'intention_acc': intention_acc.item(),
                        'traj_acc': traj_acc.item(),
                        'endpoint_loss': endpoint_loss.item(),
                        })
                    print(f"train step: {train_step}, loss: {loss : .2f}, intention acc: {intention_acc * 100 : .2f}%, traj acc: {traj_acc * 100: .2f}%, traj reg loss: {loss_traj_reg : .2f}, endpoint loss: {endpoint_loss : .2f},  time: {time.time() - start_time: .2f}s" )
                train_step += 1
                
        #模型验证
        scheduler.step()
        val_loss = val_model(val_dataset, model, scalar, True, driving_style_prior=driving_style_prior, device=device, batch_size=256, top_k=top_k)
        early_stopping(val_loss, model)
        if save_path != None:
            torch.save(train_loss_list, save_path + 'train_loss_list.pth')
            save_checkpoint(model, optimizer, epoch, save_path + 'checkpoint.pth')
        if early_stopping.early_stop:
            print("early stopping!!")
            break
    print("训练完成！")

    
def test_model(test_dataset, model: RoadPredictionModel,  scalar, predict_trajectory = True, driving_style_prior = None,
               device = 'cpu', batch_size = 64, visulization = False, top_k=6):
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    target_mean = scalar['target'][0].unsqueeze(0).unsqueeze(0).to(device)
    target_std = scalar['target'][1].unsqueeze(0).unsqueeze(0).to(device)
    surrounding_mean = scalar['surrounding'][0].unsqueeze(0).unsqueeze(2).to(device)
    surrounding_std = scalar['surrounding'][1].unsqueeze(0).unsqueeze(2).to(device)
    correct = 0
    total = 0
    traj_loss_sum = torch.tensor(0.).to(device)
    minADE = torch.tensor(0.).to(device)
    minFDE = torch.tensor(0.).to(device)
    miss_sum = torch.tensor(0.).to(device)
    model.eval()
    with torch.no_grad():
        for traj_data in test_dataloader:
            target_feature = (traj_data['target_obs_traj'].to(device) - target_mean) / target_std
            surrounding_feature = (traj_data['surrounding_obs_traj'].to(device) - surrounding_mean) / surrounding_std
            lane_change_label = traj_data['lane_change_label'].to(torch.long).to(device)
            origin_feature = traj_data['origin_feature'].to(device)
            centerline_info = traj_data['centerline_info'].to(device)
            candidate_trajectory_mask = torch.ones(target_feature.shape[0], top_k).to(device)
            future_trajectory_gt = traj_data['future_traj_gt'].to(device)
            if not predict_trajectory:
                intention_score = model(target_feature, surrounding_feature, origin_feature, centerline_info, driving_style_prior)
                acc = cal_intention_acc(intention_score, lane_change_label)
                correct += int(acc * len(intention_score))
                total += len(intention_score)
            else:
                intention_score, weighted_trajectory_score, candidate_trajectory, endpoint = model(target_feature, surrounding_feature, 
                                                                                         origin_feature, centerline_info, driving_style_prior)
                minADE += cal_minADE(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k).sum(dim=-1)
                minFDE += cal_minFDE(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k).sum(dim=-1)
                miss_rate = cal_miss_rate(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k)
                miss_sum += int(miss_rate * len(intention_score))
                total += len(intention_score)
                if visulization:
                    visualization(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k)
                    break
        
    if not predict_trajectory:
        test_accuracy = correct / total
        return test_accuracy
    else:
        return minADE / len(test_dataset), minFDE / len(test_dataset), miss_sum / total



def val_model(val_dataset, model: RoadPredictionModel,  scalar, predict_trajectory = True, driving_style_prior = None,
              device = 'cpu', batch_size = 64, top_k = 6):
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    target_mean = scalar['target'][0].unsqueeze(0).unsqueeze(0).to(device)
    target_std = scalar['target'][1].unsqueeze(0).unsqueeze(0).to(device)
    surrounding_mean = scalar['surrounding'][0].unsqueeze(0).unsqueeze(2).to(device)
    surrounding_std = scalar['surrounding'][1].unsqueeze(0).unsqueeze(2).to(device)
    loss_fn = CrossEntropyLoss().to(device)
    correct = 0
    total = 0
    minADE = torch.tensor(0.).to(device)
    model.eval()
    val_loss_sum = torch.tensor(0.).to(device)
    with torch.no_grad():
        for i, traj_data in enumerate(val_dataloader):
            target_feature = (traj_data['target_obs_traj'].to(device) - target_mean) / target_std
            surrounding_feature = (traj_data['surrounding_obs_traj'].to(device) - surrounding_mean) / surrounding_std
            lane_change_label = traj_data['lane_change_label'].to(torch.long).to(device)
            origin_feature = traj_data['origin_feature'].to(device)
            centerline_info = traj_data['centerline_info'].to(device)
            candidate_trajectory_mask = torch.ones(target_feature.shape[0], top_k).to(device)
            future_trajectory_gt = traj_data['future_traj_gt'].to(device)
            if not predict_trajectory:
                intention_score = model(target_feature, surrounding_feature, origin_feature, driving_style_prior)
                intention_acc = cal_intention_acc(intention_score, lane_change_label)
                correct += int(intention_acc * len(intention_score))
                total += len(intention_score)
                val_loss = loss_fn(intention_score, lane_change_label)
                val_loss_sum += val_loss * target_feature.shape[0]
            else:
                intention_score, weighted_trajectory_score, candidate_trajectory, endpoint = model(target_feature, surrounding_feature, 
                                                                                         origin_feature, centerline_info, driving_style_prior)
                minADE += cal_minADE(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k).sum(dim=-1)
        if predict_trajectory:
            return minADE / len(val_dataset)
        else:
            return val_loss_sum / len(val_dataset), correct / total
    

#保存checkpoints
def save_checkpoint(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

    
