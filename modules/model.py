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
from modules.trajectory_generator import TrajectoryEvaluator, TrajectoryRefinement, Time2Centerline, trajectory_generator_by_torch
from torch.utils.data.sampler import WeightedRandomSampler
from modules.loss_fn import loss_fn_traj
from modules.metrics import cal_traj_acc, cal_intention_acc, cal_minADE, cal_minFDE, cal_miss_rate, cal_offroad_rate, cal_kinematic_feasibility_rate
from modules.plot import visualization
import time
import random


class RoadPredictionModel(nn.Module):
    def __init__(self, obs_len, pred_len, input_size, hidden_size, num_layers, head_num, style_size, device, 
                 decoder_size, inputembedding_size, predict_trajectory = False, refinement_num = 5, top_k = 6, dt = 0.2) -> None:
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
        # self.trajectory_evaluator = TrajectoryEvaluator(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
        # self.trajectory_refinement = TrajectoryRefinement(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
        self.time2centerline = Time2Centerline(input_dim=hidden_size + 3 + self.style_size, driving_style_hidden_size= style_size, hidden_dim=decoder_size, n_predict=top_k)

        #compare
        self.agent2agent_fc = nn.Linear(9 * hidden_size, hidden_size, dtype=torch.float64)
        self.trajectory_generator = nn.LSTM(input_size=hidden_size + 3, hidden_size=decoder_size, num_layers=num_layers, batch_first=True, dtype=torch.float64)

    

    def forward(self, target_feature: torch.Tensor, surrounding_feature: torch.Tensor,
               origin_feature: torch.Tensor, centerline_info: torch.Tensor):
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
            time_2_leftcenterline, confidence_left = self.time2centerline(target_feature, torch.tensor([[1.,0.,0.]] * bs).to(target_feature.device))
            time_2_currentcenterline, confidence_cur = self.time2centerline(target_feature, torch.tensor([[0.,1.,0.]] * bs).to(target_feature.device))
            time_2_rightcenterline, confidence_right = self.time2centerline(target_feature, torch.tensor([[0.,0.,1.]] * bs).to(target_feature.device))
            # print("time: ", time_2_leftcenterline)
            #利用多项式拟合轨迹
            trajectory_pred_left = trajectory_generator_by_torch(origin_feature, 
                                                                 centerline_info, 
                                                                 torch.tensor([[1.,0.,0.]] * bs).to(origin_feature.device), 
                                                                 time_2_leftcenterline, self.pred_len, self.dt)
            trajectory_pred_keep = trajectory_generator_by_torch(origin_feature, 
                                                                 centerline_info, 
                                                                 torch.tensor([[0.,1.,0.]] * bs).to(origin_feature.device), 
                                                                 time_2_currentcenterline, self.pred_len, self.dt)
            trajectory_pred_right = trajectory_generator_by_torch(origin_feature, 
                                                                 centerline_info, 
                                                                 torch.tensor([[0.,0.,1.]] * bs).to(origin_feature.device), 
                                                                 time_2_rightcenterline, self.pred_len, self.dt)   
            conbined_confidence = torch.stack([confidence_cur, confidence_left, confidence_right], dim=1) #(bs, 3, 6)
            # intention_weight = intention_score.softmax(dim=1).unsqueeze(-1) #(bs, 3, 1)
            intention_weight = intention_score.unsqueeze(-1) #(bs, 3, 1)
            combined_weighted_confidence = intention_weight * conbined_confidence  #(bs, 3, 6)
            combined_weighted_confidence = combined_weighted_confidence.view(bs, -1) #(bs, 18)
            combined_trajectory = torch.cat([trajectory_pred_keep, trajectory_pred_left, trajectory_pred_right], dim=1)   #(bs, 18, n_pred, 2)  
            topk_scores, topk_indices = torch.topk(combined_weighted_confidence, self.top_k, dim=1)  # (bs, top_k)

            # Gather the corresponding trajectories
            # Use advanced indexing to get the top-k trajectories
            batch_indices = torch.arange(bs).unsqueeze(-1).expand(-1, self.top_k)  # (bs, top_k)
            selected_trajectories = combined_trajectory[batch_indices, topk_indices]  # (bs, k, n_pred, 2)
            return intention_score,  topk_scores, selected_trajectories
        return intention_score
            
    # def forward(self, target_feature: torch.Tensor, surrounding_feature: torch.Tensor, 
    #             candidate_trajectory: torch.tensor, candidate_trajectory_mask: torch.tensor):
    #     bs = target_feature.shape[0]
    #     n_surr = surrounding_feature.shape[1]
    #     n_pred = candidate_trajectory.shape[2]
    #     target_feature = self.feature_weighting_target(target_feature)
    #     surrounding_feature = self.feature_weighting_surrounding(surrounding_feature.flatten(0, 1)).reshape(bs, n_surr, self.obs_len, self.input_size)
    #     target_feature = self.lstm_target(target_feature)
    #     surrounding_feature = self.lstm_surrounding(surrounding_feature.flatten(0, 1)).reshape(bs, n_surr, self.hidden_size)
    #     target_feature = self.agent2agent(target_feature, surrounding_feature)
    #     intention_score = self.intention_prediction(target_feature)  #(bs, 3)
    #     if self.predict_trajectory:
    #         trajectory_score_left = self.trajectory_evaluator(target_feature, torch.flatten(candidate_trajectory, 0, 1), torch.tensor([[1., 0., 0.]]).to(target_feature.device))\
    #             .reshape(candidate_trajectory.shape[0], candidate_trajectory.shape[1], 1)
    #         trajectory_score_keep = self.trajectory_evaluator(target_feature, torch.flatten(candidate_trajectory, 0, 1), torch.tensor([[0., 1., 0.]]).to(target_feature.device))\
    #             .reshape(candidate_trajectory.shape[0], candidate_trajectory.shape[1], 1)
    #         trajectory_score_right = self.trajectory_evaluator(target_feature, torch.flatten(candidate_trajectory, 0, 1), torch.tensor([[0., 0., 1.]]).to(target_feature.device))\
    #             .reshape(candidate_trajectory.shape[0], candidate_trajectory.shape[1], 1)  #(bs, n_candidate, 1)

    #         trajectory_score = torch.cat([trajectory_score_left, trajectory_score_keep, trajectory_score_right], dim=-1)
    #         intention_weight = torch.softmax(intention_score, dim=-1)
    #         weighted_trajectory_score = (trajectory_score * intention_weight.unsqueeze(1)).sum(-1)
    #         #refinement
    #         weighted_trajectory_score = weighted_trajectory_score + (candidate_trajectory_mask - 1) * 1e9
    #         topk_scores, topk_indices = torch.topk(weighted_trajectory_score, self.top_k, dim=1)
    #         topk_trajectory = torch.gather(
    #             candidate_trajectory,
    #             dim=1,
    #             index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_pred, 2)
    #         )
    #         refined_topk_trajectory = topk_trajectory.clone()
    #         for i in range(self.refinement_num):
    #             refined_topk_trajectory = self.trajectory_refinement(target_feature, 
    #                                                                  torch.flatten(refined_topk_trajectory, 0, 1)).reshape(topk_trajectory.shape)
    #         updated_candidate_trajectory = candidate_trajectory.clone()
    #         updated_candidate_trajectory.scatter_(
    #             dim=1,
    #             index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_pred, 2),
    #             src=refined_topk_trajectory
    #         )
    #         return intention_score, weighted_trajectory_score, updated_candidate_trajectory
            
    #     return intention_score
            
            
            


def train_model(train_dataset, val_dataset, model: RoadPredictionModel, save_path, scalar,
                device, predict_trajectory = True, batch_size = 64, lr = 0.01, epoch = 100, patience = 0, alpha = 1.0, beta = 1.0,
                top_k = 6, decay_rate = 0.5, decay_step = 5, checkpoint = None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=save_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_step, gamma=decay_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)
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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1, last_epoch=checkpoint['epoch'])
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
            # candidate_trajectory = traj_data['future_traj_pred'].to(device)
            # candidate_trajectory_mask = traj_data['future_traj_mask'].to(device)
            candidate_trajectory_mask = torch.ones(target_feature.shape[0], top_k).to(device)
            future_trajectory_gt = traj_data['future_traj_gt'].to(device)
            if not predict_trajectory:
                intention_score = model(target_feature, surrounding_feature, origin_feature, centerline_info)
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
                intention_score, weighted_trajectory_score, candidate_trajectory = model(target_feature, surrounding_feature, 
                                                                                         origin_feature, centerline_info)
                traj_acc = cal_traj_acc(weighted_trajectory_score, candidate_trajectory, future_trajectory_gt, candidate_trajectory_mask)
                intention_acc = cal_intention_acc(intention_score, lane_change_label)
                loss, loss_intention_cls, loss_traj_cls, loss_traj_reg = loss_fn_traj(intention_score, lane_change_label, weighted_trajectory_score, 
                                                                                      future_trajectory_gt, candidate_trajectory, candidate_trajectory_mask, 
                                                                                      device=device, alpha=alpha, beta=beta)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(model.gradients['time2centerline'])
                if train_step % 10 == 0:
                    train_loss_list.append({'total_loss': loss.item(),
                        'intention_cls_loss': loss_intention_cls.item(),
                        'traj_cls_loss': loss_traj_cls.item(),
                        'traj_reg_loss': loss_traj_reg.item(),
                        'intention_acc': intention_acc.item(),
                        'traj_acc': traj_acc.item(),
                        })
                    print(f"train step: {train_step}, loss: {loss}, intention acc: {intention_acc * 100 : .2f}%, traj acc: {traj_acc * 100: .2f}%, traj reg loss: {loss_traj_reg}, time: {time.time() - start_time: .2f}s" )
                train_step += 1
                
        #模型验证
        scheduler.step()
        val_loss = val_model(val_dataset, model, scalar, True, device=device, batch_size=256, alpha=alpha, beta=beta, top_k=top_k)
        early_stopping(val_loss, model)
        if save_path != None:
            torch.save(train_loss_list, save_path + 'train_loss_list.pth')
            save_checkpoint(model, optimizer, epoch, save_path + 'checkpoint.pth')
        if early_stopping.early_stop:
            print("early stopping!!")
            break
    print("训练完成！")

    
def test_model(test_dataset, model: RoadPredictionModel,  scalar, predict_trajectory = True, 
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
    offroad_num = torch.tensor(0.).to(device)
    offkinematic_num = torch.tensor(0.).to(device)
    model.eval()
    with torch.no_grad():
        for traj_data in test_dataloader:
            target_feature = (traj_data['target_obs_traj'].to(device) - target_mean) / target_std
            surrounding_feature = (traj_data['surrounding_obs_traj'].to(device) - surrounding_mean) / surrounding_std
            lane_change_label = traj_data['lane_change_label'].to(torch.long).to(device)
            origin_feature = traj_data['origin_feature'].to(device)
            centerline_info = traj_data['centerline_info'].to(device)
            candidate_trajectory_mask = torch.ones(target_feature.shape[0], top_k).to(device)
            # candidate_trajectory = traj_data['future_traj_pred'].to(device)
            # candidate_trajectory_mask = traj_data['future_traj_mask'].to(device)
            future_trajectory_gt = traj_data['future_traj_gt'].to(device)
            # driving_direction = traj_data['driving_direction'].to(device)
            # dataset_pointer = traj_data['dataset_pointer'].to(device)
            # driving_direction = driving_direction.unsqueeze(1).unsqueeze(1).repeat(1, top_k, 15)
            # lanes_info = [maps_info[i.item()] for i in dataset_pointer]
            if not predict_trajectory:
                intention_score = model(target_feature, surrounding_feature, origin_feature, centerline_info)
                acc = cal_intention_acc(intention_score, lane_change_label)
                correct += int(acc * len(intention_score))
                total += len(intention_score)
            else:
                intention_score, weighted_trajectory_score, candidate_trajectory = model(target_feature, surrounding_feature, 
                                                                                         origin_feature, centerline_info)
                minADE += cal_minADE(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k).sum(dim=-1)
                minFDE += cal_minFDE(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k).sum(dim=-1)
                miss_rate = cal_miss_rate(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k)
                miss_sum += int(miss_rate * len(intention_score))
                #offroad_rate
                # future_traj = candidate_trajectory[:, :, :, 0:2]
                # future_traj[:, :, :, 0] = torch.where(driving_direction == 1, -future_traj[:, :, :, 0], future_traj[:, :, : ,0])
                # future_traj[:, :, :, 1] = torch.where(driving_direction == 1, future_traj[:, :, :, 1], -future_traj[:, :, :, 1])
                # future_traj = future_traj + origin_feature[:, 0:2].unsqueeze(1).unsqueeze(1).repeat(1, top_k, 15, 1)
                # offroad_num += int(cal_offroad_rate(future_traj, lanes_info) * len(target_feature))
                # offkinematic_num += int(cal_kinematic_feasibility_rate(future_traj) * len(target_feature))
                total += len(intention_score)
                if visulization:
                    visualization(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=6)
                    break
        
    if not predict_trajectory:
        test_accuracy = correct / total
        return test_accuracy
    else:
        return minADE / len(test_dataset), minFDE / len(test_dataset), miss_sum / total, offroad_num / total, offkinematic_num / total



def val_model(val_dataset, model: RoadPredictionModel,  scalar, predict_trajectory = True, 
              device = 'cpu', batch_size = 64, alpha = 1.0, beta = 1.0, top_k = 6):
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
            # candidate_trajectory = traj_data['future_traj_pred'].to(device)
            # candidate_trajectory_mask = traj_data['future_traj_mask'].to(device)
            future_trajectory_gt = traj_data['future_traj_gt'].to(device)
            if not predict_trajectory:
                intention_score = model(target_feature, surrounding_feature, origin_feature, centerline_info)
                intention_acc = cal_intention_acc(intention_score, lane_change_label)
                correct += int(intention_acc * len(intention_score))
                total += len(intention_score)
                val_loss = loss_fn(intention_score, lane_change_label)
                val_loss_sum += val_loss * target_feature.shape[0]
            else:
                intention_score, weighted_trajectory_score, candidate_trajectory = model(target_feature, surrounding_feature, 
                                                                                         origin_feature, centerline_info)
                minADE += cal_minADE(weighted_trajectory_score, candidate_trajectory, candidate_trajectory_mask, future_trajectory_gt, top_k=top_k).sum(dim=-1)
                # traj_acc = cal_traj_acc(weighted_trajectory_score, candidate_trajectory, future_trajectory_gt, candidate_trajectory_mask)
                # correct += int(traj_acc * len(weighted_trajectory_score[candidate_trajectory_mask > 0]))
                # total += len(weighted_trajectory_score[candidate_trajectory_mask > 0])
                # val_loss, _, _, _ = loss_fn_traj(intention_score, lane_change_label, weighted_trajectory_score, 
                #                           future_trajectory_gt, candidate_trajectory, candidate_trajectory_mask, 
                #                           device, alpha=alpha, beta=beta) 
                # val_loss_sum += val_loss * target_feature.shape[0]
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

    
