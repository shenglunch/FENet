import torch.nn.functional as F

def model_loss(disp_ests, disp_gt, mask):
    weights = [1.0, 1.0]#, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses), all_losses

def smooth_l1_loss(disp_ests, disp_gt, mask):
    weights = [1.0, 1.0, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses), all_losses

def crossentropy_loss(costs, mask):
    weights = [0.5, 0.5, 0.5, 1.0] # kitti12-1 kitti12-2 kitti12-3 kitti12-10 kitti15-3
    # weights = [1.0, 1.0, 0.0, 0.0] #kitti12-stage-1
    # weights = [1.0, 0.5, 0.9, 1.0] #kitti12-stage-2 kitti12-4
    # weights = [1.0, 1.0, 1.0, 1.0] # kitti12-5 kitti15-2
    # weights = [0.5, 1.0, 1.0, 1.0] # kitti15-1 kitti12-6 kitti12-7(trian1) kitti12-8(trian2) kitti12-9(trian3) kitti12-11(trian3)
    cost_losses = []
    for i, (cost, weight) in enumerate(zip(costs, weights)):
        cost_losses.append(-cost[mask].mean() * weight) 
    return sum(cost_losses)