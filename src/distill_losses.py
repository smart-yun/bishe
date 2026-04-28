from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_like(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    if teacher.shape[-2:] == student.shape[-2:]:
        return teacher
    return F.interpolate(
        teacher,
        size=student.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )


def kd_kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float = 4.0,
    conf_thresh: float = 0.0,
) -> torch.Tensor:
    """
    Pixel-wise logit distillation for dense semantic segmentation.
    If conf_thresh > 0, only high-confidence teacher pixels are used.
    """
    teacher_logits = _resize_like(teacher_logits, student_logits)

    log_p_s = F.log_softmax(student_logits / tau, dim=1)
    p_t = F.softmax(teacher_logits / tau, dim=1)

    # KL per pixel: shape (N, H, W)
    per_pixel_kl = F.kl_div(
        log_p_s,
        p_t,
        reduction='none'
    ).sum(dim=1) * (tau ** 2)

    if conf_thresh > 0:
        teacher_conf = p_t.max(dim=1).values  # (N, H, W)
        mask = (teacher_conf >= conf_thresh).float()
        return (per_pixel_kl * mask).sum() / mask.sum().clamp_min(1.0)

    return per_pixel_kl.mean()


def channel_wise_divergence(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Channel-wise distillation (CWD).

    Each channel is normalized into a probability map over spatial positions,
    then KL divergence is minimized channel by channel.

    Args:
        student_feat: (N, C, H, W)
        teacher_feat: (N, C, H, W)
        tau: temperature used in spatial softmax
    """
    teacher_feat = _resize_like(teacher_feat, student_feat)

    if student_feat.shape[1] != teacher_feat.shape[1]:
        raise ValueError(
            'CWD requires matching channel dimensions. '
            f'Got student C={student_feat.shape[1]} and teacher C={teacher_feat.shape[1]}. '
            'For this project, prefer mlp_bottleneck pruning when using CWD.'
        )

    n, c, _, _ = student_feat.shape
    s = student_feat.reshape(n * c, -1)
    t = teacher_feat.reshape(n * c, -1)

    log_p_s = F.log_softmax(s / tau, dim=1)
    p_t = F.softmax(t / tau, dim=1)

    return F.kl_div(log_p_s, p_t, reduction='batchmean') * (tau ** 2)
