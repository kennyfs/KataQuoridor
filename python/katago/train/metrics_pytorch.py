from typing import Any, Dict, List
import math

from ..train.model_pytorch import EXTRA_SCORE_DISTR_RADIUS, Model, compute_gain, ExtraOutputs, MetadataEncoder, SoftPlusWithGradientFloorFunction
from ..train.trainloop_helpers import env_flag
from ..train import modelconfigs

import torch
import torch.nn
import torch.nn.functional

def cross_entropy(pred_logits, target_probs, dim):
    return -torch.sum(target_probs * torch.nn.functional.log_softmax(pred_logits, dim=dim), dim=dim)

def huber_loss(x, y, delta):
    abs_diff = torch.abs(x - y)
    return torch.where(
        abs_diff > delta,
        (0.5 * delta * delta) + delta * (abs_diff - delta),
        0.5 * abs_diff * abs_diff,
    )

def constant_like(data, other_tensor):
    return torch.tensor(data, dtype=other_tensor.dtype, device=other_tensor.device, requires_grad=False)

class Metrics:
    def __init__(self, world_size: int, raw_model: Model):
        self.world_size = world_size
        self.pos_len = raw_model.pos_len
        self.pos_area = raw_model.pos_len * raw_model.pos_len
        self.is_quoridor = modelconfigs.is_quoridor(raw_model.config)
        self.is_quoridor4 = modelconfigs.is_quoridor4(raw_model.config)

        if self.is_quoridor4:
            # Quoridor Four-at-a-Table (docs/q4/Q4IO.md §5, §8): 3 policy planes on 11 x 11, 5 value classes.
            self.policy_len = 3 * self.pos_area
            self.value_len = modelconfigs.Q4_NUM_VALUE_LOGITS
            self.num_td_values = 4
            self.num_futurepos_values = 0
            self.num_seki_logits = 0
            self.scorebelief_len = 0
            self.scoremean_multiplier = 1.0
            self.score_belief_offset_vector = None
            self.seki_ema_on_device = False
            self.moving_unowned_proportion_sum = 0.0
            self.moving_unowned_proportion_weight = 0.0
            self.shortterm_value_error_multiplier = raw_model.shortterm_value_error_multiplier
            # Pawn plane: all 121 cells; wall planes: the 10 x 10 anchor grid at [ay][ax] (the 11th row and column
            # are padding, Q4IO §1). Slots outside are never legal.
            valid_mask = torch.zeros((3, self.pos_len, self.pos_len), dtype=torch.float32)
            valid_mask[0, :, :] = 1.0
            valid_mask[1, :10, :10] = 1.0
            valid_mask[2, :10, :10] = 1.0
            self.valid_action_mask = valid_mask.view(1, self.policy_len)
        elif self.is_quoridor:
            self.policy_len = 3 * self.pos_len * self.pos_len
            self.value_len = 2
            self.num_td_values = 2
            self.num_futurepos_values = 0
            self.num_seki_logits = 0
            self.scorebelief_len = 0
            self.scoremean_multiplier = 1.0
            self.score_belief_offset_vector = None
            self.seki_ema_on_device = False
            self.moving_unowned_proportion_sum = 0.0
            self.moving_unowned_proportion_weight = 0.0

            valid_mask = torch.zeros((3, self.pos_len, self.pos_len), dtype=torch.float32)
            valid_mask[0, :, :] = 1.0
            valid_mask[1, :8, :8] = 1.0
            valid_mask[2, :8, :8] = 1.0
            self.valid_action_mask = valid_mask.view(1, 3 * self.pos_len * self.pos_len)
        else:
            self.policy_len = raw_model.pos_len * raw_model.pos_len + 1
            self.value_len = 3
            self.num_td_values = 3
            self.num_futurepos_values = 2
            self.num_seki_logits = 4
            self.scorebelief_len = 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)

            self.scoremean_multiplier = raw_model.scoremean_multiplier

            self.score_belief_offset_vector = raw_model.value_head.score_belief_offset_vector
            # Keeping the seki moving average on the model device avoids a per-batch
            # GPU->CPU sync in the training loss and lets the loss be torch.compiled.
            self.seki_ema_on_device = env_flag("KATAGO_SEKI_EMA_ON_DEVICE", default=True)
            if self.seki_ema_on_device:
                metric_device = self.score_belief_offset_vector.device
                self.moving_unowned_proportion_sum = torch.zeros([], device=metric_device, dtype=torch.float32)
                self.moving_unowned_proportion_weight = torch.zeros([], device=metric_device, dtype=torch.float32)
            else:
                self.moving_unowned_proportion_sum = 0.0
                self.moving_unowned_proportion_weight = 0.0

    def state_dict(self):
        # Checkpoints always store plain floats regardless of where the moving
        # average lives at runtime.
        moving_sum = self.moving_unowned_proportion_sum
        moving_weight = self.moving_unowned_proportion_weight
        if isinstance(moving_sum, torch.Tensor):
            moving_sum = moving_sum.item()
        if isinstance(moving_weight, torch.Tensor):
            moving_weight = moving_weight.item()
        return dict(
            moving_unowned_proportion_sum = moving_sum,
            moving_unowned_proportion_weight = moving_weight,
        )
    def load_state_dict(self, state_dict: Dict[str,Any]):
        moving_sum = state_dict["moving_unowned_proportion_sum"]
        moving_weight = state_dict["moving_unowned_proportion_weight"]
        if isinstance(moving_sum, torch.Tensor):
            moving_sum = moving_sum.item()
        if isinstance(moving_weight, torch.Tensor):
            moving_weight = moving_weight.item()
        if self.seki_ema_on_device:
            self.moving_unowned_proportion_sum.fill_(moving_sum)
            self.moving_unowned_proportion_weight.fill_(moving_weight)
        else:
            self.moving_unowned_proportion_sum = moving_sum
            self.moving_unowned_proportion_weight = moving_weight

    def loss_policy_player_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.policy_len,)
        assert target_probs.shape == pred_logits.shape
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return global_weight * weight * loss

    def loss_policy_opponent_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.policy_len,)
        assert target_probs.shape == pred_logits.shape
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 0.15 * global_weight * weight * loss

    def loss_qvalues_samplewise(self, pred_wl_pretanh, pred_score_prescaled, target_wl, target_score, target_visits, global_weight):
        n = pred_wl_pretanh.shape[0]
        assert pred_wl_pretanh.shape == (n, self.policy_len)
        assert pred_score_prescaled.shape == (n, self.policy_len)
        assert target_wl.shape == (n, self.policy_len)
        assert target_score.shape == (n, self.policy_len)
        assert target_visits.shape == (n, self.policy_len)

        mask = (target_visits != 0).float()
        sqrtvisits = torch.sqrt(target_visits)
        sum_sqrtvisits = torch.sum(sqrtvisits, dim=1)

        pred_wl_logits = pred_wl_pretanh * mask * 2.0
        target_wl_probs = (1.0 + target_wl) / 2.0

        loss_qvalues_winloss_by_move = torch.nn.functional.binary_cross_entropy_with_logits(pred_wl_logits, target_wl_probs, reduction="none")
        loss_qvalues_winloss = torch.sum(
            loss_qvalues_winloss_by_move * sqrtvisits,
            dim=1,
        ) / (sum_sqrtvisits + 1.0)  # Add 1.0 to sum of sqrt visits so that we don't divide by 0 if we have no such data.

        pred_score = pred_score_prescaled * mask * self.scoremean_multiplier
        loss_qvalues_score_by_move = huber_loss(pred_score, target_score, delta = 12.0) * sqrtvisits
        loss_qvalues_score = torch.sum(
            loss_qvalues_score_by_move * sqrtvisits,
            dim=1,
        ) / (sum_sqrtvisits + 1.0)  # Add 1.0 to sum of sqrt visits so that we don't divide by 0 if we have no such data.

        return 1.5 * global_weight * loss_qvalues_winloss, 0.0008 * global_weight * loss_qvalues_score


    def loss_value_samplewise(self, pred_logits, target_probs, weight, global_weight):
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, self.value_len)
        assert target_probs.shape == (n, self.value_len)
        assert weight.shape == (n,)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 1.20 * global_weight * weight * loss

    def loss_td_value_samplewise(self, pred_logits, target_probs, weight, global_weight):
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, self.num_td_values, self.value_len)
        assert target_probs.shape == (n, self.num_td_values, self.value_len)
        assert weight.shape == (n,)
        assert global_weight.shape == (n,)
        loss = cross_entropy(pred_logits, target_probs, dim=2) - cross_entropy(torch.log(target_probs + 1.0e-30), target_probs, dim=2)
        return 1.20 * global_weight.unsqueeze(1) * weight.unsqueeze(1) * loss

    def loss_td_score_samplewise(self, pred, target, weight, global_weight):
        n = pred.shape[0]
        assert pred.shape == (n, self.num_td_values)
        assert target.shape == (n, self.num_td_values)
        loss = torch.sum(huber_loss(pred, target, delta = 12.0), dim=1)
        return 0.0004 * global_weight * weight * loss


    def loss_ownership_samplewise(self, pred_pretanh, target, weight, mask, mask_sum_hw, global_weight):
        # This uses a formulation where each batch element cares about its average loss.
        # In particular this means that ownership loss predictions on small boards "count more" per spot.
        # Not unlike the way that policy and value loss are also equal-weighted by batch element.
        n = pred_pretanh.shape[0]
        assert pred_pretanh.shape == (n, 1, self.pos_len, self.pos_len)
        assert target.shape == (n, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)
        pred_logits = pred_pretanh.view(-1,self.pos_area) * 2.0
        target_probs = (1.0 + target.view(-1,self.pos_area)) / 2.0
        loss = torch.sum(
            torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target_probs, reduction="none") * mask.view(-1,self.pos_area),
            dim=1,
        ) / mask_sum_hw
        return 1.5 * global_weight * weight * loss


    def loss_scoring_samplewise(self, pred_scoring, target, weight, mask, mask_sum_hw, global_weight):
        n = pred_scoring.shape[0]
        assert pred_scoring.shape == (n, 1, self.pos_len, self.pos_len)
        assert target.shape == (n, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)

        loss = torch.sum(torch.square(pred_scoring.squeeze(1) - target) * mask, dim=(1,2)) / mask_sum_hw
        # Simple huberlike transform to reduce crazy values
        loss = 4.0 * (torch.sqrt(loss * 0.5 + 1.0) - 1.0)
        return global_weight * weight * loss


    def loss_futurepos_samplewise(self, pred_pretanh, target, weight, mask, mask_sum_hw, global_weight):
        # The futurepos targets extrapolate a fixed number of steps into the future independent
        # of board size. So unlike the ownership above, generally a fixed number of spots are going to be
        # "wrong" independent of board size, so we should just equal-weight the prediction per spot.
        # However, on larger boards often the entropy of where the future moves will be should be greater
        # and also in the event of capture, there may be large captures that don't occur on small boards,
        # causing some scaling with board size. So, I dunno, let's compromise and scale by sqrt(boardarea).
        # Also, the further out targets should be weighted a little less due to them being higher entropy
        # due to simply being farther in the future, so multiply by [1,0.25].
        n = pred_pretanh.shape[0]
        assert pred_pretanh.shape == (n, self.num_futurepos_values, self.pos_len, self.pos_len)
        assert target.shape == (n, self.num_futurepos_values, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)
        loss = torch.square(torch.tanh(pred_pretanh) - target) * mask.unsqueeze(1)
        loss = loss * constant_like([1.0,0.25], loss).view(1,2,1,1)
        loss = torch.sum(loss, dim=(1, 2, 3)) / torch.sqrt(mask_sum_hw)
        return 0.25 * global_weight * weight * loss


    def loss_seki_samplewise(self, pred_logits, target, target_ownership, weight, mask, mask_sum_hw, global_weight, is_training, skip_moving_update):
        assert self.num_seki_logits == 4
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, self.num_seki_logits, self.pos_len, self.pos_len)
        assert target.shape == (n, self.pos_len, self.pos_len)
        assert target_ownership.shape == (n, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)

        owned_target = torch.square(target_ownership)
        unowned_target = 1.0 - owned_target
        unowned_proportion = torch.sum(unowned_target * mask, dim=(1, 2)) / (1.0 + mask_sum_hw)
        unowned_proportion = torch.mean(unowned_proportion * weight)
        if is_training:
            if not skip_moving_update:
                if self.seki_ema_on_device:
                    with torch.no_grad():
                        self.moving_unowned_proportion_sum.mul_(0.998).add_(unowned_proportion.detach())
                        self.moving_unowned_proportion_weight.mul_(0.998).add_(1.0)
                else:
                    self.moving_unowned_proportion_sum *= 0.998
                    self.moving_unowned_proportion_weight *= 0.998
                    self.moving_unowned_proportion_sum += unowned_proportion.item()
                    self.moving_unowned_proportion_weight += 1.0
            moving_unowned_proportion = self.moving_unowned_proportion_sum / self.moving_unowned_proportion_weight
            seki_weight_scale = 8.0 * 0.005 / (0.005 + moving_unowned_proportion)
        else:
            seki_weight_scale = 7.0

        # Loss for predicting the exact sign of seki points
        sign_pred = pred_logits[:, 0:3, :, :]
        sign_target = torch.stack(
            (
                1.0 - torch.square(target),
                torch.nn.functional.relu(target),
                torch.nn.functional.relu(-target),
            ),
            dim=1,
        )
        loss_sign = torch.sum(cross_entropy(sign_pred, sign_target, dim=1) * mask, dim=(1, 2))

        # Loss for generally predicting points that nobody will own
        neutral_pred = torch.stack(
            (pred_logits[:, 3, :, :], torch.zeros_like(target_ownership)), dim=1
        )
        neutral_target = torch.stack((unowned_target, owned_target), dim=1)
        loss_neutral = torch.sum(cross_entropy(neutral_pred, neutral_target, dim=1) * mask, dim=(1, 2))

        loss = loss_sign + 0.5 * loss_neutral
        loss = loss / mask_sum_hw
        return (global_weight * seki_weight_scale * weight * loss, seki_weight_scale)


    def loss_scoremean_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == target.shape
        assert pred.ndim == 1
        loss = huber_loss(pred, target, delta = 12.0)
        return 0.0015 * global_weight * weight * loss


    def loss_scorebelief_cdf_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.scorebelief_len,)
        assert target_probs.shape == pred_logits.shape
        pred_cdf = torch.cumsum(torch.nn.functional.softmax(pred_logits, dim=1), dim=1)
        target_cdf = torch.cumsum(target_probs, dim=1)
        loss = torch.sum(torch.square(pred_cdf-target_cdf),axis=1)
        return 0.020 * global_weight * weight * loss

    def loss_scorebelief_pdf_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.scorebelief_len,)
        assert target_probs.shape == pred_logits.shape
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 0.020 * global_weight * weight * loss

    def loss_scorestdev_samplewise(self, pred, scorebelief_logits, global_weight):
        assert pred.ndim == 1
        assert scorebelief_logits.shape[1:] == (self.scorebelief_len,)
        assert self.score_belief_offset_vector.shape == (self.scorebelief_len,)
        scorebelief_probs = torch.nn.functional.softmax(scorebelief_logits, dim=1)
        expected_score_from_belief = torch.sum(scorebelief_probs * self.score_belief_offset_vector.view(1,-1),dim=1,keepdim=True)
        stdev_of_belief = torch.sqrt(0.001 + torch.sum(
            scorebelief_probs * torch.square(
                self.score_belief_offset_vector.view(1,-1) - expected_score_from_belief
            ),
            dim=1
        ))
        loss = huber_loss(pred, stdev_of_belief, delta = 10.0)
        return 0.001 * global_weight * loss

    def loss_lead_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == target.shape
        assert pred.ndim == 1
        loss = huber_loss(pred, target, delta = 8.0)
        return 0.0060 * global_weight * weight * loss

    def loss_variance_time_samplewise(self, pred, target, weight, global_weight):
        assert pred.shape == target.shape
        assert pred.ndim == 1
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction.
        loss = huber_loss(pred, target + 1.0e-5, delta = 50.0)
        return 0.0003 * global_weight * weight * loss


    def loss_shortterm_value_error_samplewise(self, pred, td_value_pred_logits, td_value_target_probs, weight, global_weight):
        td_value_pred_probs = torch.softmax(td_value_pred_logits[:,2,:],axis=1)
        predvalue = (td_value_pred_probs[:,0] - td_value_pred_probs[:,1]).detach()
        realvalue = td_value_target_probs[:,2,0] - td_value_target_probs[:,2,1]
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction, 0.01%.
        sqerror = torch.square(predvalue-realvalue) + 1.0e-8
        loss = huber_loss(pred, sqerror, delta = 0.4)
        return 2.0 * global_weight * weight * loss

    def loss_shortterm_score_error_samplewise(self, pred, td_score_pred, td_score_target, weight, global_weight):
        predscore = td_score_pred[:,2].detach()
        realscore = td_score_target[:,2]
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction, one hundredth of a point.
        sqerror = torch.square(predscore-realscore) + 1.0e-4
        loss = huber_loss(pred, sqerror, delta = 100.0)
        return 0.00002 * global_weight * weight * loss

    def accuracy1(self, pred_logits, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * (torch.argmax(pred_logits,dim=1) == torch.argmax(target_probs,dim=1)))

    def target_entropy(self, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * -torch.sum(target_probs * torch.log(target_probs + 1e-30), dim=-1))

    def square_value(self, value_logits, global_weight):
        if self.is_quoridor:
            return torch.sum(global_weight * torch.square(torch.sum(torch.softmax(value_logits, dim=1) * constant_like([1, -1], global_weight), dim=1)))
        return torch.sum(global_weight * torch.square(torch.sum(torch.softmax(value_logits,dim=1) * constant_like([1,-1,0],global_weight), dim=1)))

    @staticmethod
    def get_model_norms(raw_model):
        reg_dict: Dict[str,List] = {}
        raw_model.add_reg_dict(reg_dict)

        device = reg_dict["normal"][0].device
        dtype = torch.float32

        with torch.no_grad():
            norms: Dict[str,float] = {}
            for group_name in reg_dict:
                if len(reg_dict[group_name]) > 0:
                    norm = torch.zeros([],device=device,dtype=dtype)
                    for tensor in reg_dict[group_name]:
                        norm += torch.sum(tensor * tensor)
                    norms[group_name] = torch.sqrt(norm).detach().cpu().item()

        return norms

    @staticmethod
    def get_model_norm_metrics(raw_model):
        """Model norm metrics in the naming used by the training metrics dict."""
        norms = Metrics.get_model_norms(raw_model)
        return {f"norm_{group_name}_batch": value for group_name, value in norms.items()}

    # A channel counts as dead when its norm is at most this fraction of the reference norm of
    # its tensor, see reference_channel_norm. Shared with resurrect_dead_channels.py.
    DEAD_CHANNEL_NORM_FRAC = 0.01

    @staticmethod
    def is_output_channel_tensor(param):
        """True for nn.Linear [out, in] and conv [out, in, kh, kw] weights, whose dim 0 indexes
        output channels. Biases and norm gammas and betas are excluded, including ones stored as
        [1, C, 1, 1]."""
        return param.dim() in (2, 4) and param.shape[0] >= 2 and param[0].numel() >= 2

    @staticmethod
    def reference_channel_norm(norms):
        """Typical norm of the live channels of a tensor, given the norms of all its channels:
        the median over channels whose norm exceeds 0.1% of the largest, so that a tensor where
        most channels are exactly zero still gets a reference from the live ones. Returns a 0-dim
        tensor, which is 0 when every channel is near zero."""
        largest = norms.max() if norms.numel() > 0 else torch.zeros([], device=norms.device)
        live = norms[norms > 1e-3 * largest]
        return live.median() if live.numel() > 0 else torch.zeros([], device=norms.device)

    @staticmethod
    def get_output_channel_metrics(raw_model, floor_norms_by_param=None):
        """Fractions of output channels, over all output-channel tensors of the model, that are
        dead or sitting at the weight decay floor.

        deadrows_batch: channels whose weight norm is at most DEAD_CHANNEL_NORM_FRAC times the
        tensor's reference channel norm. Exactly-zero channels always count. Such a channel
        contributes nothing and, if the tensors reading it are also zero, can never recover.
        floorrows_batch: channels whose norm is at or below the tensor's weight decay floor. Only
        reported when floors are given (see -wd-floor-frac in train.py)."""
        num_channels = 0
        num_dead = 0
        num_at_floor = 0
        with torch.no_grad():
            for param in raw_model.parameters():
                if not Metrics.is_output_channel_tensor(param):
                    continue
                r = torch.linalg.vector_norm(param, dim=tuple(range(1, param.dim())), dtype=torch.float32)
                ref = Metrics.reference_channel_norm(r)
                num_channels += r.numel()
                num_dead += int((r <= Metrics.DEAD_CHANNEL_NORM_FRAC * ref).sum().item())
                if floor_norms_by_param is not None and param in floor_norms_by_param:
                    num_at_floor += int((r <= floor_norms_by_param[param]).sum().item())
        metrics = {"deadrows_batch": num_dead / max(1, num_channels)}
        if floor_norms_by_param is not None:
            metrics["floorrows_batch"] = num_at_floor / max(1, num_channels)
        return metrics

    def get_specific_norms_and_gradient_stats(self,raw_model):
        with torch.no_grad():
            params = {}
            for name, param in raw_model.named_parameters():
                params[name] = param

            stats = {}
            def add_norm_and_grad_stats(name):
                param = params[name]
                if name.endswith(".weight"):
                    fanin = param.shape[1]
                elif name.endswith(".gamma"):
                    fanin = 1
                elif name.endwith(".beta"):
                    fanin = 1
                else:
                    assert False, "unimplemented case to compute stats on parameter"

                # 1.0 means that the average squared magnitude of a parameter in this tensor is around where
                # it would be at initialization, assuming it uses the activation that the model generally
                # uses (e.g. relu or mish)
                param_scale = torch.sqrt(torch.mean(torch.square(param))) / compute_gain(raw_model.activation) * math.sqrt(fanin)
                stats[f"{name}.SCALE_batch"] = param_scale

                # How large is the gradient, on the same scale?
                stats[f"{name}.GRADSC_batch"] = torch.sqrt(torch.mean(torch.square(param.grad))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

                # And how large is the component of the gradient that is orthogonal to the overall magnitude of the parameters?
                orthograd = param.grad - param * (torch.sum(param.grad * param) / (1e-20 + torch.sum(torch.square(param))))
                stats[f"{name}.OGRADSC_batch"] = torch.sqrt(torch.mean(torch.square(orthograd))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

            add_norm_and_grad_stats("blocks.1.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.1.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.1.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.6.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.6.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.6.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.10.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.10.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.10.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.16.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.16.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.16.normactconvq.norm.gamma")

            add_norm_and_grad_stats("policy_head.conv1p.weight")
            add_norm_and_grad_stats("value_head.conv1.weight")
            add_norm_and_grad_stats("intermediate_policy_head.conv1p.weight")
            add_norm_and_grad_stats("intermediate_value_head.conv1.weight")

        return stats

    def metrics_dict_batchwise(
        self,
        raw_model,
        model_output_postprocessed_byheads,
        extra_outputs,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        seki_loss_scale,
        variance_time_loss_scale,
        main_loss_scale,
        intermediate_loss_scale,
        include_model_norms=True,
    ):
        results = self.metrics_dict_batchwise_single_heads_output(
            raw_model,
            model_output_postprocessed_byheads[0],
            batch,
            is_training=is_training,
            soft_policy_weight_scale=soft_policy_weight_scale,
            disable_optimistic_policy=disable_optimistic_policy,
            meta_kata_only_soft_policy=meta_kata_only_soft_policy,
            value_loss_scale=value_loss_scale,
            td_value_loss_scales=td_value_loss_scales,
            seki_loss_scale=seki_loss_scale,
            variance_time_loss_scale=variance_time_loss_scale,
            is_intermediate=False,
            include_model_norms=include_model_norms,
        )
        if main_loss_scale is not None:
            results["loss_sum"] = main_loss_scale * results["loss_sum"]

        if raw_model.get_has_intermediate_head():
            assert len(model_output_postprocessed_byheads) > 1
            if raw_model.training:
                assert intermediate_loss_scale is not None
            else:
                if intermediate_loss_scale is None:
                    intermediate_loss_scale = 1.0

            if intermediate_loss_scale is not None:
                iresults = self.metrics_dict_batchwise_single_heads_output(
                    raw_model,
                    model_output_postprocessed_byheads[1],
                    batch,
                    is_training=is_training,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    disable_optimistic_policy=disable_optimistic_policy,
                    meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=seki_loss_scale,
                    variance_time_loss_scale=variance_time_loss_scale,
                    is_intermediate=True,
                )
                for key,value in iresults.items():
                    if key != "loss_sum":
                        results["I"+key] = value
                results["loss_sum"] = results["loss_sum"] + intermediate_loss_scale * iresults["loss_sum"]

        # Only the aggregate loss participates in backward. Keeping every
        # logging component attached makes autograd treat dozens of metrics
        # as differentiable outputs and retain their graphs until logging.
        for key, value in results.items():
            if key != "loss_sum" and isinstance(value, torch.Tensor):
                results[key] = value.detach()
        return results

    def metrics_dict_batchwise_single_heads_output(
        self,
        raw_model,
        model_output_postprocessed,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        seki_loss_scale,
        variance_time_loss_scale,
        is_intermediate,
        include_model_norms=True,
    ):
        if self.is_quoridor4:
            return self.metrics_dict_batchwise_single_heads_output_q4(
                raw_model=raw_model,
                model_output_postprocessed=model_output_postprocessed,
                batch=batch,
                soft_policy_weight_scale=soft_policy_weight_scale,
                value_loss_scale=value_loss_scale,
                is_intermediate=is_intermediate,
                include_model_norms=include_model_norms,
            )
        if self.is_quoridor:
            return self.metrics_dict_batchwise_single_heads_output_quoridor(
                raw_model=raw_model,
                model_output_postprocessed=model_output_postprocessed,
                batch=batch,
                is_training=is_training,
                soft_policy_weight_scale=soft_policy_weight_scale,
                disable_optimistic_policy=disable_optimistic_policy,
                meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                value_loss_scale=value_loss_scale,
                td_value_loss_scales=td_value_loss_scales,
                seki_loss_scale=seki_loss_scale,
                variance_time_loss_scale=variance_time_loss_scale,
                is_intermediate=is_intermediate,
                include_model_norms=include_model_norms,
            )
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            ownership_pretanh,
            pred_scoring,
            futurepos_pretanh,
            seki_logits,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
            scorebelief_logits,
        ) = model_output_postprocessed

        input_binary_nchw = batch["binaryInputNCHW"]
        input_global_nc = batch["globalInputNC"]
        target_policy_ncmove = batch["policyTargetsNCMove"]
        target_global_nc = batch["globalTargetsNC"]
        score_distribution_ns = batch["scoreDistrN"]
        target_value_nchw = batch["valueTargetsNCHW"]

        mask = input_binary_nchw[:, 0, :, :].contiguous()
        mask_sum_hw = torch.sum(mask,dim=(1,2))

        n = input_binary_nchw.shape[0]
        h = input_binary_nchw.shape[2]
        w = input_binary_nchw.shape[3]

        policymask = torch.cat((mask.view(n,h*w),mask.new_ones((n,1))),dim=1)

        target_policy_player = target_policy_ncmove[:, 0, :]
        target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
        target_policy_opponent = target_policy_ncmove[:, 1, :]
        target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)
        target_policy_player_soft = (target_policy_player + 1e-7) * policymask
        target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
        target_policy_player_soft /= torch.sum(target_policy_player_soft, dim=1, keepdim=True)
        target_policy_opponent_soft = (target_policy_opponent + 1e-7) * policymask
        target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
        target_policy_opponent_soft /= torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

        target_weight_policy_player = target_global_nc[:, 26]
        target_weight_policy_opponent = target_global_nc[:, 28]

        target_value = target_global_nc[:, 0:3]
        target_scoremean = target_global_nc[:, 3]
        target_td_value = torch.stack(
            (target_global_nc[:, 4:7], target_global_nc[:, 8:11], target_global_nc[:, 12:15]), dim=1
        )
        target_td_score = torch.cat(
            (target_global_nc[:, 7:8], target_global_nc[:, 11:12], target_global_nc[:, 15:16]), dim=1
        )
        target_lead = target_global_nc[:, 21]
        target_variance_time = target_global_nc[:, 22]
        global_weight = target_global_nc[:, 25]
        target_weight_ownership = target_global_nc[:, 27]
        target_weight_lead = target_global_nc[:, 29]
        target_weight_futurepos = target_global_nc[:, 33]
        target_weight_scoring = target_global_nc[:, 34]
        target_weight_value = 1.0 - target_global_nc[:, 35]
        target_weight_td_value = 1.0 - target_global_nc[:, 24]

        target_score_distribution = score_distribution_ns / 100.0

        target_ownership = target_value_nchw[:, 0, :, :]
        target_seki = target_value_nchw[:, 1, :, :]
        target_futurepos = target_value_nchw[:, 2:4, :, :]
        target_scoring = target_value_nchw[:, 4, :, :] / 120.0

        predict_q_values = False
        if raw_model.config["version"] <= 11:
            assert raw_model.policy_head.num_policy_outputs == 4
            policy_opt_loss_scale = 1.000
            long_policy_opt_loss_scale = 0.0
            short_policy_opt_loss_scale = 0.0
        elif raw_model.config["version"] <= 15:
            assert raw_model.policy_head.num_policy_outputs == 6
            policy_opt_loss_scale = 0.930
            long_policy_opt_loss_scale = 0.100
            short_policy_opt_loss_scale = 0.200
        elif raw_model.config["version"] <= 16:  # version 16 has predict_q_values implied
            assert raw_model.policy_head.num_policy_outputs == 8
            policy_opt_loss_scale = 0.930
            long_policy_opt_loss_scale = 0.100
            short_policy_opt_loss_scale = 0.200
            predict_q_values = True
        elif raw_model.config["version"] <= 17:
            assert raw_model.policy_head.num_policy_outputs == 6 or raw_model.policy_head.num_policy_outputs == 8
            policy_opt_loss_scale = 0.930
            long_policy_opt_loss_scale = 0.100
            short_policy_opt_loss_scale = 0.200
            predict_q_values = bool(raw_model.config.get("predict_q_values"))
        else:
            raise RuntimeError("unsupported version: " + str(raw_model.config["version"]))

        loss_policy_player = self.loss_policy_player_samplewise(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        ).sum()
        loss_policy_opponent = self.loss_policy_opponent_samplewise(
            policy_logits[:, 1, :],
            target_policy_opponent,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        target_weight_policy_player_soft = target_weight_policy_player
        target_weight_policy_opponent_soft = target_weight_policy_opponent
        if meta_kata_only_soft_policy:
            metadata_input_nc = batch["metadataInputNC"]
            assert metadata_input_nc.shape[0] == target_weight_policy_player_soft.shape[0]
            # 151 indicates source 0 = katago
            target_weight_policy_player_soft = target_weight_policy_player_soft * metadata_input_nc[:,151]
            target_weight_policy_opponent_soft = target_weight_policy_opponent_soft * metadata_input_nc[:,151]

        loss_policy_player_soft = self.loss_policy_player_samplewise(
            policy_logits[:, 2, :],
            target_policy_player_soft,
            target_weight_policy_player_soft,
            global_weight,
        ).sum()
        loss_policy_opponent_soft = self.loss_policy_opponent_samplewise(
            policy_logits[:, 3, :],
            target_policy_opponent_soft,
            target_weight_policy_opponent_soft,
            global_weight,
        ).sum()

        if raw_model.config["version"] <= 11:
            target_weight_longoptimistic_policy = torch.zeros_like(global_weight)
            loss_longoptimistic_policy = torch.zeros_like(loss_policy_player)
        elif disable_optimistic_policy:
            target_weight_longoptimistic_policy = target_weight_policy_player * 0.5
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()
        else:
            # Long-term optimistic policy. Weight policy by:
            # Final game win squared (squaring discourages draws)
            win_squared = torch.square(
                target_global_nc[:, 0] # win (or draw, weighted by draw utility)
                + 0.5 * target_global_nc[:, 2] # noresult
            )
            # Or the score outcome of the game being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            longterm_score_stdevs_excess = (target_global_nc[:, 3] - pred_scoremean.detach()) / torch.sqrt(torch.square(pred_scorestdev.detach()) + 0.25)
            target_weight_longoptimistic_policy = torch.clamp(
                win_squared + torch.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0,
                max=1.0,
            )
            target_weight_longoptimistic_policy = (
                target_weight_longoptimistic_policy
                * target_weight_policy_player # game has normal target
                * target_weight_ownership # and also actually ended in full ownership and score, not a sidepos
            )
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()

        assert len(loss_longoptimistic_policy.shape) == 0
        assert len(target_weight_longoptimistic_policy.shape) == 1
        assert target_weight_longoptimistic_policy.shape[0] == n
        target_weight_longoptimistic_policy_sum = (global_weight * target_weight_longoptimistic_policy).sum()

        if raw_model.config["version"] <= 11:
            target_weight_shortoptimistic_policy = torch.zeros_like(global_weight)
            loss_shortoptimistic_policy = torch.zeros_like(loss_policy_player)
        elif disable_optimistic_policy:
            target_weight_shortoptimistic_policy = target_weight_policy_player * 0.5
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()
        else:
            # Short-term optimistic policy. Weight policy by:
            # The shortterm value outcome being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            shortterm_value_actual = target_global_nc[:, 12] - target_global_nc[:, 13]
            shortterm_value_pred = torch.nn.functional.softmax(td_value_logits[:, 2, :].detach(), dim=1)
            shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
            shortterm_value_stdevs_excess = (shortterm_value_actual - shortterm_value_pred) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
            # Or the shortterm score outcome being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            shortterm_score_stdevs_excess = (target_global_nc[:, 15] - pred_td_score[:,2].detach()) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
            target_weight_shortoptimistic_policy = torch.clamp(
                torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0) + torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0,
                max=1.0,
            )
            target_weight_shortoptimistic_policy = (
                target_weight_shortoptimistic_policy
                * target_weight_policy_player # game has normal target
                * target_weight_ownership # and also actually ended in full ownership and score, not a sidepos
            )
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()

        assert len(loss_shortoptimistic_policy.shape) == 0
        assert len(target_weight_shortoptimistic_policy.shape) == 1
        assert target_weight_shortoptimistic_policy.shape[0] == n
        target_weight_shortoptimistic_policy_sum = (global_weight * target_weight_shortoptimistic_policy).sum()


        loss_value = self.loss_value_samplewise(
            value_logits, target_value, target_weight_value, global_weight
        ).sum()

        loss_td_value_unsummed = self.loss_td_value_samplewise(
            td_value_logits, target_td_value, target_weight_td_value, global_weight
        )
        assert self.num_td_values == 3
        loss_td_value1 = loss_td_value_unsummed[:,0].sum()
        loss_td_value2 = loss_td_value_unsummed[:,1].sum()
        loss_td_value3 = loss_td_value_unsummed[:,2].sum()

        loss_td_score = self.loss_td_score_samplewise(
            pred_td_score, target_td_score, target_weight_ownership, global_weight
        ).sum()

        loss_ownership = self.loss_ownership_samplewise(
            ownership_pretanh,
            target_ownership,
            target_weight_ownership,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_scoring = self.loss_scoring_samplewise(
            pred_scoring,
            target_scoring,
            target_weight_scoring,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_futurepos = self.loss_futurepos_samplewise(
            futurepos_pretanh,
            target_futurepos,
            target_weight_futurepos,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        (loss_seki,seki_weight_scale) = self.loss_seki_samplewise(
            seki_logits,
            target_seki,
            target_ownership,
            target_weight_ownership,
            mask,
            mask_sum_hw,
            global_weight,
            is_training,
            skip_moving_update=is_intermediate,
        )
        loss_seki = loss_seki.sum()
        seki_weight_scale = seki_weight_scale.sum() if not isinstance(seki_weight_scale,float) else seki_weight_scale
        loss_scoremean = self.loss_scoremean_samplewise(
            pred_scoremean,
            target_scoremean,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_cdf = self.loss_scorebelief_cdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_pdf = self.loss_scorebelief_pdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorestdev = self.loss_scorestdev_samplewise(
            pred_scorestdev,
            scorebelief_logits,
            global_weight,
        ).sum()
        loss_lead = self.loss_lead_samplewise(
            pred_lead,
            target_lead,
            target_weight_lead,
            global_weight,
        ).sum()
        loss_variance_time = self.loss_variance_time_samplewise(
            pred_variance_time,
            target_variance_time,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_value_error = self.loss_shortterm_value_error_samplewise(
            pred_shortterm_value_error,
            td_value_logits,
            target_td_value,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_score_error = self.loss_shortterm_score_error_samplewise(
            pred_shortterm_score_error,
            pred_td_score,
            target_td_score,
            target_weight_ownership,
            global_weight,
        ).sum()

        if not predict_q_values:
            target_weight_qvalues = torch.zeros_like(global_weight)
            loss_qvalues_winloss = torch.zeros_like(loss_policy_player)
            loss_qvalues_score = torch.zeros_like(loss_policy_player)
        else:
            target_qvalue_ncmove = batch["qValueTargetsNCMove"]
            target_weight_qvalues = torch.zeros_like(global_weight)
            loss_qvalues_winloss, loss_qvalues_score = self.loss_qvalues_samplewise(
                policy_logits[:, 6, :],
                policy_logits[:, 7, :],
                target_qvalue_ncmove[:, 0, :] / 32000.0,
                target_qvalue_ncmove[:, 1, :] / 60.0,
                target_qvalue_ncmove[:, 2, :],
                global_weight
            )
            loss_qvalues_winloss = loss_qvalues_winloss.sum()
            loss_qvalues_score = loss_qvalues_score.sum()

        loss_sum = (
            loss_policy_player * policy_opt_loss_scale
            + loss_policy_opponent
            + loss_policy_player_soft * soft_policy_weight_scale
            + loss_policy_opponent_soft * soft_policy_weight_scale
            + loss_longoptimistic_policy * long_policy_opt_loss_scale
            + loss_shortoptimistic_policy * short_policy_opt_loss_scale
            + loss_value * value_loss_scale
            + loss_td_value1 * td_value_loss_scales[0]
            + loss_td_value2 * td_value_loss_scales[1]
            + loss_td_value3 * td_value_loss_scales[2]
            + loss_td_score
            + loss_ownership
            + loss_scoring * 0.25
            + loss_futurepos
            + loss_seki * seki_loss_scale
            + loss_scoremean
            + loss_scorebelief_cdf
            + loss_scorebelief_pdf
            + loss_scorestdev
            + loss_lead
            + loss_variance_time * variance_time_loss_scale
            + loss_shortterm_value_error
            + loss_shortterm_score_error
            + loss_qvalues_winloss
            + loss_qvalues_score
        )

        policy_acc1 = self.accuracy1(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        )
        square_value = self.square_value(value_logits, global_weight)

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_opponent_soft,
            "p0lopt_sum": loss_longoptimistic_policy,
            "p0loptw_sum": target_weight_longoptimistic_policy_sum,
            "p0sopt_sum": loss_shortoptimistic_policy,
            "p0soptw_sum": target_weight_shortoptimistic_policy_sum,
            "vloss_sum": loss_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdsloss_sum": loss_td_score,
            "oloss_sum": loss_ownership,
            "sloss_sum": loss_scoring,
            "fploss_sum": loss_futurepos,
            "skloss_sum": loss_seki,
            "smloss_sum": loss_scoremean,
            "sbcdfloss_sum": loss_scorebelief_cdf,
            "sbpdfloss_sum": loss_scorebelief_pdf,
            "sdregloss_sum": loss_scorestdev,
            "leadloss_sum": loss_lead,
            "vtimeloss_sum": loss_variance_time,
            "evstloss_sum": loss_shortterm_value_error,
            "esstloss_sum": loss_shortterm_score_error,
            "qwlloss_sum": loss_qvalues_winloss,
            "qscloss_sum": loss_qvalues_score,
            "loss_sum": loss_sum,
            "pacc1_sum": policy_acc1,
            "vsquare_sum": square_value,
        }

        if is_intermediate:
            return results
        else:
            weight = global_weight.sum()
            nsamples = int(global_weight.shape[0])
            policy_target_entropy = self.target_entropy(
                target_policy_player,
                target_weight_policy_player,
                global_weight,
            )
            soft_policy_target_entropy = self.target_entropy(
                target_policy_player_soft,
                target_weight_policy_player,
                global_weight,
            )

            extra_results = {
                "wsum": weight * self.world_size,
                "nsamp": nsamples * self.world_size,
                "ptentr_sum": policy_target_entropy,
                "ptsoftentr_sum": soft_policy_target_entropy,
                "sekiweightscale_sum": seki_weight_scale * weight,
            }

            if include_model_norms:
                extra_results.update(self.get_model_norm_metrics(raw_model))

            for key,value in extra_results.items():
                results[key] = value
            return results

    def metrics_dict_batchwise_single_heads_output_quoridor(
        self,
        raw_model,
        model_output_postprocessed,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        seki_loss_scale,
        variance_time_loss_scale,
        is_intermediate,
        include_model_norms=True,
    ):
        # Training data and training exist only for the current Quoridor I/O version (docs/QuoridorIOv2.md).
        io_version = modelconfigs.get_quoridor_io_version(raw_model.config)
        assert io_version == modelconfigs.QUORIDOR_TRAINING_IO_VERSION, (
            f"only Quoridor I/O v{modelconfigs.QUORIDOR_TRAINING_IO_VERSION} models can be trained, got v{io_version}"
            " (use a *_quoridor_v3 model config; quoridor_upgrade_v2_to_v3.py upgrades a v2 checkpoint)")
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_variance_time,
            pred_utility_score,
            pred_utility_score_stdev,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
            trajectory_pretanh,
            wall_graph_pretanh,
            pred_lead,
            pred_remaining_turns,
        ) = model_output_postprocessed

        input_binary_nchw = batch["binaryInputNCHW"]
        target_policy_ncmove = batch["policyTargetsNCMove"]
        target_global_nc = batch["globalTargetsNC"]
        target_value_nchw = batch["valueTargetsNCHW"]

        n = input_binary_nchw.shape[0]

        valid_mask = self.valid_action_mask.to(device=policy_logits.device)
        policy_logits = policy_logits.reshape(n, 6, self.policy_len)
        policy_logits = policy_logits.masked_fill(valid_mask == 0, -10000.0)

        target_policy_player = target_policy_ncmove[:, 0, :] * valid_mask
        sum_p0 = torch.sum(target_policy_player, dim=1, keepdim=True)
        target_policy_player = target_policy_player / torch.clamp(sum_p0, min=1e-8)

        target_policy_opponent = target_policy_ncmove[:, 1, :] * valid_mask
        sum_p1 = torch.sum(target_policy_opponent, dim=1, keepdim=True)
        target_policy_opponent = target_policy_opponent / torch.clamp(sum_p1, min=1e-8)

        target_policy_player_soft = (target_policy_player + 1e-7) * valid_mask
        target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
        target_policy_player_soft /= torch.clamp(torch.sum(target_policy_player_soft, dim=1, keepdim=True), min=1e-8)

        target_policy_opponent_soft = (target_policy_opponent + 1e-7) * valid_mask
        target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
        target_policy_opponent_soft /= torch.clamp(torch.sum(target_policy_opponent_soft, dim=1, keepdim=True), min=1e-8)

        global_weight = target_global_nc[:, 25]
        target_weight_policy_player = target_global_nc[:, 26]
        target_weight_policy_opponent = target_global_nc[:, 28]

        loss_policy_player = self.loss_policy_player_samplewise(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        ).sum()

        loss_policy_opponent = self.loss_policy_opponent_samplewise(
            policy_logits[:, 1, :],
            target_policy_opponent,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        loss_policy_player_soft = self.loss_policy_player_samplewise(
            policy_logits[:, 2, :],
            target_policy_player_soft,
            target_weight_policy_player,
            global_weight,
        ).sum()

        loss_policy_opponent_soft = self.loss_policy_opponent_samplewise(
            policy_logits[:, 3, :],
            target_policy_opponent_soft,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        if disable_optimistic_policy:
            target_weight_longoptimistic_policy = target_weight_policy_player * 0.5
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()
            target_weight_shortoptimistic_policy = target_weight_policy_player * 0.5
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()
        else:
            win_squared = torch.square(target_global_nc[:, 0])
            target_weight_longoptimistic_policy = torch.clamp(
                win_squared, min=0.0, max=1.0
            ) * target_weight_policy_player
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()

            # Short-term optimistic policy, as upstream: weight by the short-term (horizon index 2,
            # globalTargetsNC[12:16]) value or utility-score outcome being around 1.5 sigma better than expected.
            # There is no TD-score head, so the score excess is measured against the utility-score
            # prediction, which is also what the shortterm score error head is trained against.
            shortterm_value_actual = target_global_nc[:, 12] - target_global_nc[:, 13]
            shortterm_value_pred = torch.nn.functional.softmax(td_value_logits[:, 2, :].detach(), dim=1)
            shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
            shortterm_value_stdevs_excess = (shortterm_value_actual - shortterm_value_pred) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
            shortterm_score_stdevs_excess = (target_global_nc[:, 15] - pred_utility_score.detach()) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
            # Rows whose C15 is not usable (C70 = 1, converted lambda > 0 data) contribute only the value term.
            target_weight_shortterm_score = 1.0 - target_global_nc[:, 70]
            target_weight_shortoptimistic_policy = torch.clamp(
                torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0)
                + target_weight_shortterm_score * torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0,
                max=1.0,
            )
            target_weight_shortoptimistic_policy = (
                target_weight_shortoptimistic_policy
                * target_weight_policy_player # game has normal target
                * target_global_nc[:, 27] # and the game actually ended, not a side position
            )
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()

        target_weight_longoptimistic_policy_sum = (global_weight * target_weight_longoptimistic_policy).sum()
        target_weight_shortoptimistic_policy_sum = (global_weight * target_weight_shortoptimistic_policy).sum()

        # Value loss (2 logits: win, loss)
        target_value = target_global_nc[:, 0:2]
        target_value = target_value / torch.clamp(torch.sum(target_value, dim=1, keepdim=True), min=1e-8)
        target_weight_value = 1.0 - target_global_nc[:, 35]
        loss_value = (1.50 * global_weight * target_weight_value * cross_entropy(value_logits, target_value, dim=1)).sum()

        # TD-Value loss (4 horizons, each 2 logits)
        target_td_value = torch.stack(
            (target_global_nc[:, 4:6], target_global_nc[:, 8:10], target_global_nc[:, 12:14], target_global_nc[:, 16:18]),
            dim=1,
        )
        target_td_value = target_td_value / torch.clamp(torch.sum(target_td_value, dim=2, keepdim=True), min=1e-8)
        target_weight_td_value = 1.0 - target_global_nc[:, 24]

        td_ce = cross_entropy(td_value_logits, target_td_value, dim=2)  # (N, 4)
        loss_td_value1 = (global_weight * target_weight_td_value * td_ce[:, 0]).sum()
        loss_td_value2 = (global_weight * target_weight_td_value * td_ce[:, 1]).sum()
        loss_td_value3 = (global_weight * target_weight_td_value * td_ce[:, 2]).sum()
        loss_td_value4 = (global_weight * target_weight_td_value * td_ce[:, 3]).sum()
        loss_td_value = 0.20 * 0.25 * (loss_td_value1 + loss_td_value2 + loss_td_value3 + loss_td_value4)

        # Outcome targets (cpp/dataio/trainingwrite.h), all from the side to move's view and in moves. Draws are
        # written with u = 0 (weighted) and no lead (weight 0).
        target_weight_outcome = target_global_nc[:, 27]

        # Loss scales follow upstream (loss_*_samplewise above), converted to Quoridor units: a tempo / move of margin
        # is roughly 3 Go points of typical error, so Huber deltas are /3 and weights x9 (same gradient per unit of
        # relative error in the L2 region); time-like targets use games ~4x shorter than 19x19 Go (weights x16).
        # Utility score u (KataGo's scoreMean): the game's final u, globalTargetsNC[20].
        # Upstream scoremean: 0.0015, delta 12 -> 0.0135, delta 4.
        target_utility_score = target_global_nc[:, 20]
        utility_score_huber = huber_loss(pred_utility_score, target_utility_score, delta=4.0)
        loss_utility_score = (0.0135 * global_weight * target_weight_outcome * utility_score_huber).sum()

        # Utility-score stdev: modeled on upstream's scorestdev, but with no score belief head to take a stdev of,
        # it is trained so that stdev^2 regresses the squared error of the (detached) utility-score prediction.
        utility_score_sqerror = torch.square(pred_utility_score.detach() - target_utility_score) + 1.0e-4
        utility_score_stdev_huber = huber_loss(torch.square(pred_utility_score_stdev), utility_score_sqerror, delta=10.0)
        loss_utility_score_stdev = (0.004 * global_weight * target_weight_outcome * utility_score_stdev_huber).sum()

        # Tempo lead s (KataGo's lead): globalTargetsNC[21], weight [29] (Huber delta=1.0)
        target_lead = target_global_nc[:, 21]
        target_weight_lead = target_global_nc[:, 29]
        # Upstream lead: 0.006, delta 8 -> 0.054, delta 3.
        lead_huber = huber_loss(pred_lead, target_lead, delta=3.0)
        loss_lead = (0.054 * global_weight * target_weight_lead * lead_huber).sum()

        # Remaining plies (training-only auxiliary target): plies until the game ends, globalTargetsNC[23], / 300.
        target_remaining_turns = target_global_nc[:, 23] / 300.0
        remaining_turns_huber = huber_loss(pred_remaining_turns, target_remaining_turns, delta=0.25)
        # No upstream counterpart; weighted so a typical error (~0.15, i.e. ~45 plies) costs ~0.05, like vtime.
        loss_remaining_turns = (5.0 * global_weight * target_weight_outcome * remaining_turns_huber).sum()

        # Shortterm winloss / score error: as upstream's loss_shortterm_{value,score}_error_samplewise,
        # against the short-term TD targets (horizon index 2: value = globalTargetsNC[12:14], score = [15], which
        # is the searches' utility score). There is no TD-score head, so the score error is that of the
        # utility-score prediction.
        shortterm_value_probs = torch.softmax(td_value_logits[:, 2, :], dim=1)
        shortterm_value_pred = (shortterm_value_probs[:, 0] - shortterm_value_probs[:, 1]).detach()
        shortterm_value_real = target_td_value[:, 2, 0] - target_td_value[:, 2, 1]
        shortterm_value_sqerror = torch.square(shortterm_value_pred - shortterm_value_real) + 1.0e-8
        loss_shortterm_value_error = (
            2.0 * global_weight * target_weight_td_value
            * huber_loss(pred_shortterm_value_error, shortterm_value_sqerror, delta=0.4)
        ).sum()
        # Upstream: 0.00002, delta 100 on squared points -> squared units /9: 0.0016, delta 11.
        # C70 is 1 minus the weight of the short-term score target C15 (0 when written; 1 on rows converted from data
        # with a time bonus lambda > 0, where C15 holds the searches' old utility scores, quoridor_convert_tdata_lambda0.py).
        shortterm_score_sqerror = torch.square(pred_utility_score.detach() - target_global_nc[:, 15]) + 1.0e-4
        loss_shortterm_score_error = (
            0.0016 * global_weight * target_weight_td_value * (1.0 - target_global_nc[:, 70])
            * huber_loss(pred_shortterm_score_error, shortterm_score_sqerror, delta=11.0)
        ).sum()

        # Trajectory loss (BCEWithLogits, 2 channels)
        target_trajectory = target_value_nchw[:, 0:2, :, :]
        target_weight_aux = target_weight_outcome
        bce_traj = torch.nn.functional.binary_cross_entropy_with_logits(trajectory_pretanh, target_trajectory, reduction="none")
        bce_traj_sample = torch.mean(bce_traj, dim=(1, 2, 3))
        loss_trajectory = (0.02 * global_weight * target_weight_aux * bce_traj_sample).sum()

        # Wall Graph loss (BCEWithLogits on active 8x8 anchor region, 2 channels)
        target_wall_graph = target_value_nchw[:, 2:4, :8, :8]
        pred_wall_graph = wall_graph_pretanh[:, :, :8, :8]
        bce_wall = torch.nn.functional.binary_cross_entropy_with_logits(pred_wall_graph, target_wall_graph, reduction="none")
        bce_wall_sample = torch.mean(bce_wall, dim=(1, 2, 3))
        loss_wall_graph = (0.02 * global_weight * target_weight_aux * bce_wall_sample).sum()

        # Variance Time loss. Upstream: 0.0003, delta 50 turns -> games ~4x shorter: 0.005, delta 12.
        target_variance_time = target_global_nc[:, 22]
        vtime_huber = huber_loss(pred_variance_time, target_variance_time, delta=12.0)
        loss_variance_time = (0.005 * global_weight * target_weight_aux * vtime_huber).sum()

        # Total Loss sum
        loss_sum = (
            loss_policy_player
            + loss_policy_opponent
            + loss_policy_player_soft * 0.05 * soft_policy_weight_scale
            + loss_policy_opponent_soft * 0.02 * soft_policy_weight_scale
            + loss_longoptimistic_policy * 0.10
            + loss_shortoptimistic_policy * 0.05
            + loss_value * value_loss_scale
            + loss_td_value
            + loss_utility_score
            + loss_utility_score_stdev
            + loss_lead
            + loss_remaining_turns
            + loss_shortterm_value_error
            + loss_shortterm_score_error
            + loss_trajectory
            + loss_wall_graph
            + loss_variance_time * variance_time_loss_scale
        )

        policy_acc1 = self.accuracy1(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        )
        square_value = self.square_value(value_logits, global_weight)

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_opponent_soft,
            "p0lopt_sum": loss_longoptimistic_policy,
            "p0loptw_sum": target_weight_longoptimistic_policy_sum,
            "p0sopt_sum": loss_shortoptimistic_policy,
            "p0soptw_sum": target_weight_shortoptimistic_policy_sum,
            "vloss_sum": loss_value,
            "tdvloss_sum": loss_td_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdvloss4_sum": loss_td_value4,
            # Quoridor: smloss = utility score u, leadloss = tempo lead s, rtloss = remaining plies.
            "smloss_sum": loss_utility_score,
            "leadloss_sum": loss_lead,
            "rtloss_sum": loss_remaining_turns,
            "trajloss_sum": loss_trajectory,
            "wallloss_sum": loss_wall_graph,
            "vtimeloss_sum": loss_variance_time,
            "tdsloss_sum": torch.zeros_like(loss_value),
            "oloss_sum": torch.zeros_like(loss_value),
            "sloss_sum": torch.zeros_like(loss_value),
            "fploss_sum": torch.zeros_like(loss_value),
            "skloss_sum": torch.zeros_like(loss_value),
            "sbcdfloss_sum": torch.zeros_like(loss_value),
            "sbpdfloss_sum": torch.zeros_like(loss_value),
            "sdregloss_sum": loss_utility_score_stdev,
            "evstloss_sum": loss_shortterm_value_error,
            "esstloss_sum": loss_shortterm_score_error,
            "qwlloss_sum": torch.zeros_like(loss_value),
            "qscloss_sum": torch.zeros_like(loss_value),
            "loss_sum": loss_sum,
            "pacc1_sum": policy_acc1,
            "vsquare_sum": square_value,
        }

        if is_intermediate:
            return results
        else:
            weight = global_weight.sum()
            nsamples = int(global_weight.shape[0])
            policy_target_entropy = self.target_entropy(
                target_policy_player,
                target_weight_policy_player,
                global_weight,
            )
            soft_policy_target_entropy = self.target_entropy(
                target_policy_player_soft,
                target_weight_policy_player,
                global_weight,
            )

            extra_results = {
                "wsum": weight * self.world_size,
                "nsamp": nsamples * self.world_size,
                "ptentr_sum": policy_target_entropy,
                "ptsoftentr_sum": soft_policy_target_entropy,
                "sekiweightscale_sum": torch.zeros_like(weight),
            }

            if include_model_norms:
                extra_results.update(self.get_model_norm_metrics(raw_model))

            for key, value in extra_results.items():
                results[key] = value
            return results


    # -----------------------------------------------------------------------------------------------------------------
    # Quoridor Four-at-a-Table (Q4). The loss mirrors the Duel loss above (metrics_dict_batchwise_single_heads_output_
    # quoridor), which itself follows upstream KataGo; docs/q4/rounds/R5.md lists every term with its weight and source.

    Q4_MASKED_LOGIT = -10000.0  # as the Duel loss masks invalid policy slots

    @staticmethod
    def q4_alive_value_mask(input_global_nc):
        """[N, 5] float mask of the value classes that can win: the alive relative seats (global inputs 8..11,
        Q4IO §4; the globals are already relative to the seat to move) and the draw, which is always possible."""
        alive = input_global_nc[:, 8:12]
        return torch.cat((alive, torch.ones_like(alive[:, :1])), dim=1)

    @staticmethod
    def q4_mask_value_logits(value_logits, alive_mask):
        """Masks eliminated seats out of the value softmax (Plan §7 item 3). value_logits [N, 5] or [N, H, 5].
        masked_fill replaces the logit, so an eliminated seat's logit gets exactly zero gradient."""
        if value_logits.dim() == 3:
            alive_mask = alive_mask.unsqueeze(1)
        return value_logits.masked_fill(alive_mask == 0, Metrics.Q4_MASKED_LOGIT)

    @staticmethod
    def q4_mover_utility(value_probs, num_alive):
        """Plan §8.3 utility of the seat to move (relative seat 0) for a value vector [me, next, across, previous,
        draw], with winLossUtilityFactor = 1: 2 v[0] + (2 / n) v[draw] - 1. It is KataGo's win - loss for n = 2."""
        return 2.0 * value_probs[..., 0] + (2.0 / num_alive) * value_probs[..., 4] - 1.0

    def q4_policy_logits_and_target(self, logits, target, n):
        """Flattened [N, 363] logits with the never-legal slots masked, and the normalized target over valid slots."""
        valid_mask = self.valid_action_mask.to(device=logits.device)
        logits = logits.reshape(n, self.policy_len).masked_fill(valid_mask == 0, Metrics.Q4_MASKED_LOGIT)
        target = target * valid_mask
        target = target / torch.clamp(torch.sum(target, dim=1, keepdim=True), min=1e-8)
        return logits, target

    def q4_soft_policy_target(self, target):
        """KataGo's soft policy target: target ^ 0.25 renormalized (over the valid slots)."""
        valid_mask = self.valid_action_mask.to(device=target.device)
        soft = torch.pow((target + 1e-7) * valid_mask, 0.25)
        return soft / torch.clamp(torch.sum(soft, dim=1, keepdim=True), min=1e-8)

    def metrics_dict_batchwise_single_heads_output_q4(
        self,
        raw_model,
        model_output_postprocessed,
        batch,
        soft_policy_weight_scale,
        value_loss_scale,
        is_intermediate,
        include_model_norms=True,
    ):
        io_version = modelconfigs.get_q4_io_version(raw_model.config)
        assert io_version in modelconfigs.Q4_TRAINING_IO_VERSIONS, (
            f"only Q4 I/O v{modelconfigs.Q4_TRAINING_IO_VERSIONS} models can be trained, got v{io_version}")
        (
            policy_logits,      # [N, 2, 3, 11, 11]: search policy, style policy
            value_logits,       # [N, 5]
            pred_misc,          # [N, 6] raw
            trajectory_logits,  # [N, 1, 11, 11]
            paths_logits,       # [N, 4, 11, 11]
            walls_logits,       # [N, 8, 11, 11]
            td_value_logits,    # [N, 4, 5]
            policy_aux_logits,  # [N, 3, 3, 11, 11]: next-seat policy, soft search policy, soft next-seat policy
        ) = model_output_postprocessed

        input_global_nc = batch["globalInputNC"]
        target_policy_ncmove = batch["policyTargetsNCMove"]
        target_global_nc = batch["globalTargetsNC"]
        target_value_nchw = batch["valueTargetsNCHW"]
        n = target_global_nc.shape[0]

        global_weight = target_global_nc[:, 25]
        target_weight_policy_player = target_global_nc[:, 26]
        target_weight_outcome = target_global_nc[:, 27]
        target_weight_policy_next = target_global_nc[:, 28]
        target_weight_policy_style = target_global_nc[:, 29]
        target_weight_td_value = 1.0 - target_global_nc[:, 33]
        target_weight_value = 1.0 - target_global_nc[:, 34]

        # ---- Policies. Weights as the Duel loss: search policy 1.0; next-seat policy (KataGo's opponent policy)
        # 0.15 inside loss_policy_opponent_samplewise; soft policies 0.05 / 0.02 x soft_policy_weight_scale.
        p0_logits, target_policy_player = self.q4_policy_logits_and_target(policy_logits[:, 0], target_policy_ncmove[:, 0], n)
        pstyle_logits, target_policy_style = self.q4_policy_logits_and_target(policy_logits[:, 1], target_policy_ncmove[:, 1], n)
        pnext_logits, target_policy_next = self.q4_policy_logits_and_target(policy_aux_logits[:, 0], target_policy_ncmove[:, 2], n)
        p0soft_logits, _ = self.q4_policy_logits_and_target(policy_aux_logits[:, 1], target_policy_ncmove[:, 0], n)
        pnextsoft_logits, _ = self.q4_policy_logits_and_target(policy_aux_logits[:, 2], target_policy_ncmove[:, 2], n)
        target_policy_player_soft = self.q4_soft_policy_target(target_policy_player)
        target_policy_next_soft = self.q4_soft_policy_target(target_policy_next)

        loss_policy_player = self.loss_policy_player_samplewise(
            p0_logits, target_policy_player, target_weight_policy_player, global_weight).sum()
        loss_policy_next = self.loss_policy_opponent_samplewise(
            pnext_logits, target_policy_next, target_weight_policy_next, global_weight).sum()
        loss_policy_player_soft = self.loss_policy_player_samplewise(
            p0soft_logits, target_policy_player_soft, target_weight_policy_player, global_weight).sum()
        loss_policy_next_soft = self.loss_policy_opponent_samplewise(
            pnextsoft_logits, target_policy_next_soft, target_weight_policy_next, global_weight).sum()
        # Style policy (variant 1, the played action, C1 / C29): it takes the channel of Duel's long-term optimistic
        # policy (Q4IO §5.1), whose weight 0.10 it keeps.
        loss_policy_style = self.loss_policy_player_samplewise(
            pstyle_logits, target_policy_style, target_weight_policy_style, global_weight).sum()

        # ---- Value (5 classes, eliminated seats masked out of the softmax). Duel: 1.50 x value_loss_scale.
        alive_mask = self.q4_alive_value_mask(input_global_nc)
        value_logits_masked = self.q4_mask_value_logits(value_logits, alive_mask)
        target_value = target_global_nc[:, 0:5]
        loss_value = (1.50 * global_weight * target_weight_value * cross_entropy(value_logits_masked, target_value, dim=1)).sum()

        # ---- TD value, 4 horizons as Duel: C5-9, C10-14, C15-19 and the search value C20-24, weight 1 - C33,
        # 0.20 * 0.25 per horizon (Duel), same masking.
        target_td_value = torch.stack(
            (target_global_nc[:, 5:10], target_global_nc[:, 10:15], target_global_nc[:, 15:20], target_global_nc[:, 20:25]),
            dim=1,
        )
        target_td_value = target_td_value / torch.clamp(torch.sum(target_td_value, dim=2, keepdim=True), min=1e-8)
        td_value_logits_masked = self.q4_mask_value_logits(td_value_logits, alive_mask)
        td_ce = cross_entropy(td_value_logits_masked, target_td_value, dim=2)  # [N, 4]
        loss_td_value1 = (global_weight * target_weight_td_value * td_ce[:, 0]).sum()
        loss_td_value2 = (global_weight * target_weight_td_value * td_ce[:, 1]).sum()
        loss_td_value3 = (global_weight * target_weight_td_value * td_ce[:, 2]).sum()
        loss_td_value4 = (global_weight * target_weight_td_value * td_ce[:, 3]).sum()
        loss_td_value = 0.20 * 0.25 * (loss_td_value1 + loss_td_value2 + loss_td_value3 + loss_td_value4)

        # ---- Misc (Q4IO §5.2), raw outputs.
        # Slot 0: plies to the end / 100 (target C35, weighted by C27). Duel's remaining plies: 5.0 x Huber(delta 0.25)
        # on plies / 300; the same loss per ply of error on plies / 100 is 5.0 / 9 x Huber(delta 0.75).
        target_remaining_plies = target_global_nc[:, 35] / 100.0
        loss_remaining_plies = (
            (5.0 / 9.0) * global_weight * target_weight_outcome
            * huber_loss(pred_misc[:, 0], target_remaining_plies, delta=0.75)
        ).sum()
        # Slots 1-4: each relative seat's final walls-only distance / 32 (C36-39, weights C40-43, 0 for a seat
        # eliminated before the end). Duel's lead loss 0.054 x Huber(delta 3) on the distance in cells (32 x slot).
        # The coefficient is divided by the number of seats with nonzero weight, so a row's total matches Duel's
        # single lead term (the weights equal the row's outcome weight C27, which can be fractional, so their sum
        # is not a seat count).
        target_final_dist = target_global_nc[:, 36:40]
        target_weight_final_dist = target_global_nc[:, 40:44]
        num_weighted_seats = (target_weight_final_dist > 0).sum(dim=1, keepdim=True).clamp(min=1)
        final_dist_huber = huber_loss(32.0 * pred_misc[:, 1:5], target_final_dist, delta=3.0)
        loss_final_dist = (
            0.054 / num_weighted_seats * global_weight.unsqueeze(1) * target_weight_final_dist * final_dist_huber
        ).sum()
        # Slot 5: short-term value error, as Duel / upstream loss_shortterm_value_error_samplewise (2.0, Huber delta
        # 0.4, squared softplus x shortterm_value_error_multiplier) against the short-term TD target (horizon index
        # 2, C15-19). KataGo's value (win - loss) is replaced by the Plan §8.3 utility of the seat to move.
        num_alive = torch.clamp(torch.sum(input_global_nc[:, 8:12], dim=1), min=1.0)
        shortterm_value_pred = self.q4_mover_utility(torch.softmax(td_value_logits_masked[:, 2, :], dim=1), num_alive).detach()
        shortterm_value_real = self.q4_mover_utility(target_td_value[:, 2, :], num_alive)
        shortterm_value_sqerror = torch.square(shortterm_value_pred - shortterm_value_real) + 1.0e-8
        pred_shortterm_value_error = SoftPlusWithGradientFloorFunction.apply(pred_misc[:, 5], 0.05, True) * self.shortterm_value_error_multiplier
        loss_shortterm_value_error = (
            2.0 * global_weight * target_weight_td_value
            * huber_loss(pred_shortterm_value_error, shortterm_value_sqerror, delta=0.4)
        ).sum()

        # ---- Spatial heads: BCE, Duel's trajectory / wall-graph weight 0.02 each, weighted by the outcome weight
        # C27, mean over the valid cells of the valid channels (Duel: mean over the cells).
        # Exported trajectory of me (valueTargetsNCHW C0).
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        loss_trajectory = (
            0.02 * global_weight * target_weight_outcome
            * torch.mean(bce(trajectory_logits[:, 0], target_value_nchw[:, 0], reduction="none"), dim=(1, 2))
        ).sum()
        # All seats' paths (C0-3): a seat eliminated at this row has weight 0 (alive flags = global inputs 8..11).
        seat_alive = input_global_nc[:, 8:12]
        paths_bce = torch.mean(bce(paths_logits, target_value_nchw[:, 0:4], reduction="none"), dim=(2, 3))  # [N, 4]
        paths_bce = torch.sum(paths_bce * seat_alive, dim=1) / torch.clamp(torch.sum(seat_alive, dim=1), min=1.0)
        loss_paths = (0.02 * global_weight * target_weight_outcome * paths_bce).sum()
        # Future walls (C4-11) on the 10 x 10 anchor domain, channel 2k / 2k + 1 = relative seat k.
        walls_bce = bce(walls_logits[:, :, :10, :10], target_value_nchw[:, 4:12, :10, :10], reduction="none")
        walls_bce = torch.mean(walls_bce, dim=(2, 3)).view(n, 4, 2).mean(dim=2)  # [N, 4]
        walls_bce = torch.sum(walls_bce * seat_alive, dim=1) / torch.clamp(torch.sum(seat_alive, dim=1), min=1.0)
        loss_walls = (0.02 * global_weight * target_weight_outcome * walls_bce).sum()

        loss_sum = (
            loss_policy_player
            + loss_policy_next
            + loss_policy_player_soft * 0.05 * soft_policy_weight_scale
            + loss_policy_next_soft * 0.02 * soft_policy_weight_scale
            + loss_policy_style * 0.10
            + loss_value * value_loss_scale
            + loss_td_value
            + loss_remaining_plies
            + loss_final_dist
            + loss_shortterm_value_error
            + loss_trajectory
            + loss_paths
            + loss_walls
        )

        policy_acc1 = self.accuracy1(p0_logits, target_policy_player, target_weight_policy_player, global_weight)
        value_probs = torch.softmax(value_logits_masked, dim=1)
        square_value = torch.sum(global_weight * torch.square(self.q4_mover_utility(value_probs, num_alive)))

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_next,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_next_soft,
            "pstyleloss_sum": loss_policy_style,
            "vloss_sum": loss_value,
            "tdvloss_sum": loss_td_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdvloss4_sum": loss_td_value4,
            "rtloss_sum": loss_remaining_plies,
            "fdistloss_sum": loss_final_dist,
            "evstloss_sum": loss_shortterm_value_error,
            "trajloss_sum": loss_trajectory,
            "pathsloss_sum": loss_paths,
            "wallloss_sum": loss_walls,
            "loss_sum": loss_sum,
            "pacc1_sum": policy_acc1,
            "vsquare_sum": square_value,
        }

        if is_intermediate:
            return results
        weight = global_weight.sum()
        extra_results = {
            "wsum": weight * self.world_size,
            "nsamp": int(global_weight.shape[0]) * self.world_size,
            "ptentr_sum": self.target_entropy(target_policy_player, target_weight_policy_player, global_weight),
            "ptsoftentr_sum": self.target_entropy(target_policy_player_soft, target_weight_policy_player, global_weight),
        }
        if include_model_norms:
            extra_results.update(self.get_model_norm_metrics(raw_model))
        results.update(extra_results)
        return results
