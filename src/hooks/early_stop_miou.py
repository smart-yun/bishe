from __future__ import annotations

from typing import Optional

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EarlyStopMIOUHook(Hook):
    """Early stop by validation mIoU plateau.

    Args:
        monitor (str): Metric key to monitor. Default: 'mIoU'.
        min_delta (float): Minimum improvement to be considered progress.
        patience (int): Number of consecutive validations without sufficient
            improvement before stopping.
        rule (str): 'greater' (maximize) or 'less' (minimize).
    """

    priority = 'LOW'

    def __init__(
        self,
        monitor: str = 'mIoU',
        min_delta: float = 0.05,
        patience: int = 5,
        rule: str = 'greater',
    ) -> None:
        if patience <= 0:
            raise ValueError(f'patience must be > 0, got {patience}')
        if rule not in ('greater', 'less'):
            raise ValueError(f"rule must be 'greater' or 'less', got {rule}")

        self.monitor = monitor
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.rule = rule

        self.best: Optional[float] = None
        self.bad_count = 0

    def _is_improved(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.rule == 'greater':
            return (current - self.best) >= self.min_delta
        return (self.best - current) >= self.min_delta

    def _request_stop(self, runner) -> None:
        # Different mmengine versions may use different stop flags.
        if hasattr(runner, 'train_loop') and hasattr(runner.train_loop, 'stop_training'):
            runner.train_loop.stop_training = True
        if hasattr(runner, '_stop_training'):
            runner._stop_training = True
        if hasattr(runner, 'should_stop'):
            runner.should_stop = True

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        metrics = metrics or {}
        if self.monitor not in metrics:
            runner.logger.warning(
                f'[EarlyStopMIOUHook] monitor key {self.monitor!r} not in metrics: '
                f'{list(metrics.keys())}. Skip this val epoch.'
            )
            return

        current = float(metrics[self.monitor])

        if self._is_improved(current):
            prev_best = self.best
            self.best = current
            self.bad_count = 0
            if prev_best is None:
                runner.logger.info(
                    f'[EarlyStopMIOUHook] init best {self.monitor}={self.best:.4f}'
                )
            else:
                runner.logger.info(
                    f'[EarlyStopMIOUHook] improved: {self.monitor} '
                    f'{prev_best:.4f} -> {self.best:.4f}'
                )
            return

        self.bad_count += 1
        runner.logger.info(
            f'[EarlyStopMIOUHook] no significant improvement '
            f'({self.bad_count}/{self.patience}), current={current:.4f}, '
            f'best={self.best:.4f}, min_delta={self.min_delta:.4f}'
        )

        if self.bad_count >= self.patience:
            runner.logger.warning(
                f'[EarlyStopMIOUHook] trigger early stop: '
                f'{self.monitor} has not improved by >= {self.min_delta} '
                f'for {self.patience} consecutive validations.'
            )
            self._request_stop(runner)
