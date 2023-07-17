import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss, L1Loss

SECONDS_IN_HOUR = 60 * 60
PADDING_VALUE = -100


def calculate_corridor_func(device, video_length, mean_video_length, d=5):
    """
    Calculates the corridor function, presented in the paper:
    https://arxiv.org/pdf/2002.11367.pdf. 
    This implementation follows the same notions as the paper.
    """
    c_x = torch.arange(video_length, device=device)

    g_t = video_length - c_x
    n_t = torch.maximum(mean_video_length - c_x, torch.zeros_like(c_x))

    a_t = 1 - (2 / (1 + torch.exp((c_x / video_length) * d)))

    c_t = (a_t * g_t) + ((1 - a_t) * n_t)

    return c_t


def calculate_corridor_mask(preds, labels, mean_video_length, d=5, tolerance=0):
    """
    Calculates mask of which pred lays between the corridor function and label.
    Following https://arxiv.org/pdf/2002.11367.pdf.
    This implementation follows the same notions as the paper
    """
    c_t = calculate_corridor_func(
        device=preds.device,
        video_length=len(preds),
        mean_video_length=mean_video_length,
        d=d,
    )

    mask = torch.logical_or(
        torch.logical_and(c_t <= preds, preds <= labels + tolerance),
        torch.logical_and(labels - tolerance <= preds, preds <= c_t),
    )

    return mask, c_t


def calculate_corridor_weights(
    preds,
    labels,
    video_length,
    mean_video_length,
    d=5,
    tolerance=0,
    off_corridor_penalty=1,
):
    """
    Calculates the loss weight for each index.
    Following https://arxiv.org/pdf/2002.11367.pdf.
    This implementation follows the same notions as the paper.
    """
    w = torch.ones(len(preds) - video_length, device=preds.device) * PADDING_VALUE

    p = preds[:video_length]
    l = labels[:video_length]
    mask, c_t = calculate_corridor_mask(
        preds=p, labels=l, mean_video_length=mean_video_length, d=d, tolerance=tolerance
    )

    weights = torch.pow(torch.abs(p - l) / torch.abs(c_t - l), 2)

    weights[torch.logical_not(mask)] = off_corridor_penalty
    weights = torch.cat([weights, w])

    return weights


class ETCLoss(_Loss):
    """
    The input of this loss is B X S X 2, where the vector in zero dim is the ETC normalized
    by max hours and the vector in 1 dim is the Progress (0-1) of this second.
    S is the sequence (video) length.
    The labels are given as dict with two keys 'etc' and 'progress'. Each holding matrix of shape (B X S).
    This is because the batch can hold videos with different sizes,
    for example:
    Batch Size is 1 and sequence (video) length is 5 
    pred: [[[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]]
    labels:
        {
            "etc": [[0., 0., 0., 0., 0.]]
            "progress": [[0., 0., 0., 0., 0.]]
        }
    S will be the size of the longest video in the batch and the extra sequence for each video
    will padded using PADDING_VALUE and be ignored
    """

    def __init__(
        self,
        device,
        mean_length,
        max_hours=3,
        off_corridor_penalty=1,
        alpha=1,
        beta=1,
        gamma=1,
        delta=1,
        d=5,
        **rest
    ):
        super().__init__(reduction="none")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.device = device
        self.mean_length = mean_length
        self.d = d
        self.off_corridor_penalty = off_corridor_penalty
        self.max_hours = max_hours

    def seq_mae(self, preds: Tensor, labels: Tensor, mask, weights=None):
        if weights is None:
            weights = torch.ones_like(preds)

        abs_error = torch.abs(preds - labels) * weights

        lengths = torch.sum(mask, dim=1)

        abs_error[mask == False] = 0

        mses = torch.sum(abs_error, dim=1) / lengths
        return torch.mean(mses)

    def seq_smape(self, preds, labels, mask, weights):
        lengths = torch.sum(mask, dim=1)

        smape_pp = (
            torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels))
        ) * weights

        smape_pp[mask == False] = 0

        smape = torch.sum(smape_pp, dim=1) / lengths

        return torch.mean(smape)

    def seq_var_loss(self, preds, labels):
        rolled_preds = torch.roll(preds, 1, dims=1)
        rolled_preds[:, 0] = preds[:, 0]
        return self.seq_mae(preds, rolled_preds, labels != PADDING_VALUE)

    def corridor_weights(self, preds, labels):
        weights = torch.zeros_like(preds)
        for i in range(len(preds)):
            lengths = torch.sum(labels != PADDING_VALUE, dim=1)
            weights[i] = calculate_corridor_weights(
                preds=preds[i] * SECONDS_IN_HOUR * self.max_hours,
                labels=labels[i] * SECONDS_IN_HOUR * self.max_hours,
                video_length=lengths[i],
                mean_video_length=self.mean_length,
                d=self.d,
                off_corridor_penalty=self.off_corridor_penalty,
            )
        return weights

    def forward(self, preds: Tensor, labels: Tensor):
        etc_preds = preds[:, :, 0]
        prog_preds = preds[:, :, 1]

        weights = self.corridor_weights(etc_preds, labels["etc"])

        if not self.weighted_var_loss:
            interval_loss_weights = torch.ones_like(weights)

        loss = (
            (
                self.alpha
                * self.seq_mae(
                    etc_preds, labels["etc"], labels["etc"] != PADDING_VALUE, weights
                )
            )
            + (
                self.beta
                * self.seq_smape(
                    etc_preds, labels["etc"], labels["etc"] != PADDING_VALUE, weights
                )
            )
            + (
                self.gamma
                * self.seq_mae(
                    prog_preds,
                    labels["progress"],
                    labels["etc"] != PADDING_VALUE,
                    weights,
                )
            )
            + (
                self.delta
                * self.seq_var_loss(etc_preds, labels["etc"], interval_loss_weights)
            )
        )
        return loss


class TotalVarLoss(_Loss):
    """
    Loss for the ETCouple model.
    When using a model like ETCouple, we would like to smooth the result by total var.
    Given the wanted loss (L1/L2), averaging each couple losses, then adding the real diff (which is negative)
    in order to allow the loss to be 0.

    penalty_type: allowing to penalize the result by progress / late stage

    """

    def __init__(self, device, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.device = device

    def forward(
        self,
        preds0: Tensor,
        preds1: Tensor,
        labels0: Tensor,
        labels1: Tensor,
    ) -> float:
        total_var = 0
        for i in range(len(labels0)):
            var_loss = (
                torch.pow((labels0[i] - preds0[i]), 2)
                + torch.pow((labels1[i] - preds1[i]), 2)
            ) / 2
            diff = torch.abs(
                torch.abs(preds0[i] - preds1[i]) - torch.abs(labels0[i] - labels1[i])
            )
            total_var += var_loss + diff

        total_var /= len(labels0)

        return total_var


class SMAPELoss(_Loss):
    def __init__(self, device, **rest):
        super().__init__(reduction="none")
        self.l1_loss = L1Loss(reduction="none")
        self.device = device

    def forward(
        self,
        preds0: Tensor,
        preds1: Tensor,
        labels0: Tensor,
        labels1: Tensor,
    ):
        l1_loss_0 = self.l1_loss(preds0, labels0)
        x = preds0 + labels0
        l1_loss_1 = self.l1_loss(preds1, labels1)
        y = preds1 + labels1

        return torch.mean(l1_loss_0 / x) + torch.mean(l1_loss_1 / y)


class ETCoupleLoss(_Loss):
    """
    Loss for ETCouple model. Calculate the loss given two points (current second and interval second before).
    """

    def __init__(
        self,
        device,
        reduction: str = "mean",
        mae_weight=1,
        smape_weight=1,
        total_var_weight=1,
    ) -> None:
        super().__init__(reduction=reduction)

        self.total_var = TotalVarLoss(device, reduction=reduction)
        self.smape_loss = SMAPELoss(device)
        self.l1_loss = L1Loss()
        self.mae_weight = mae_weight
        self.smape_weight = smape_weight
        self.total_var_weight = total_var_weight

    def forward(self, preds: Tensor, labels: Tensor):
        prog_preds = preds[0]
        prog_labels = labels["progress"]

        etc_preds = preds[1]
        etc_labels = labels["etc"]

        prog_mae = self.l1_loss.forward(prog_preds, prog_labels)
        etc_mae = self.l1_loss.forward(etc_preds, etc_labels)
        mae = self.mae_weight * (prog_mae + etc_mae)

        etc_first_preds = etc_preds[0::2]
        etc_second_preds = etc_preds[1::2]
        etc_first_labels = etc_labels[0::2]
        etc_second_labels = etc_labels[1::2]

        smape = self.smape_weight * self.smape_loss.forward(
            etc_first_preds, etc_second_preds, etc_first_labels, etc_second_labels
        )

        total_var = self.total_var_weight * self.total_var.forward(
            etc_first_preds, etc_second_preds, etc_first_labels, etc_second_labels
        )

        return [mae, smape, total_var]
