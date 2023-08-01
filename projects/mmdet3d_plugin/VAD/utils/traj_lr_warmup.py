import torch

def get_traj_warmup_loss_weight(
    cur_epoch,
    tot_epoch,
    start_pos=0.3,
    end_pos=0.35,
    scale_weight=1.1
):
    epoch_percentage = cur_epoch / tot_epoch
    sigmoid_input = 5 / (end_pos-start_pos) * epoch_percentage - 2.5 * (end_pos+start_pos) / (end_pos - start_pos)

    return scale_weight * torch.sigmoid(torch.tensor(sigmoid_input))
