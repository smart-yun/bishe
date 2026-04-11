# -*- coding: utf-8 -*-
"""Generate baseline-vs-pruned comparison table and conclusion sentences.

Supported input modes:
1) Pair mode (recommended)
   --baseline-json <baseline_metrics.json>
   --pruned-json   <pruned_metrics.json>

2) Single tp summary mode
   --tp-summary <torch_prune_segformer summary json>
   baseline=history[0], pruned=history[-1].
   In this mode, Params and MACs are guaranteed if present.

    python src/compare_baseline_pruned.py \
    --tp-summary exports/tp_local_iter_summary.json \
    --title "TP Local Iterative prune compare" \
    --out-md exports/tp_local_iter_compare.md \
    --out-json exports/tp_local_iter_compare.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class MetricPack:
    name: str
    miou: Optional[float] = None
    params: Optional[float] = None
    flops: Optional[float] = None
    latency_ms: Optional[float] = None
    fps: Optional[float] = None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _from_baseline_metrics(name: str, data: Dict[str, Any]) -> MetricPack:
    miou = _safe_get(data, 'miou_metrics', 'mIoU')

    params = _safe_get(data, 'complexity', 'params')
    if params is None:
        params_m = _safe_get(data, 'complexity', 'params_m')
        if params_m is not None:
            params = float(params_m) * 1e6

    flops = _safe_get(data, 'complexity', 'flops')
    if flops is None:
        flops_g = _safe_get(data, 'complexity', 'flops_g')
        if flops_g is not None:
            flops = float(flops_g) * 1e9

    latency_ms = _safe_get(data, 'latency', 'latency_ms_mean')
    fps = _safe_get(data, 'latency', 'fps_mean')

    return MetricPack(
        name=name,
        miou=float(miou) if miou is not None else None,
        params=float(params) if params is not None else None,
        flops=float(flops) if flops is not None else None,
        latency_ms=float(latency_ms) if latency_ms is not None else None,
        fps=float(fps) if fps is not None else None,
    )


def _from_tp_summary(data: Dict[str, Any]) -> Tuple[MetricPack, MetricPack]:
    history = data.get('history', [])
    if len(history) < 2:
        raise ValueError('tp summary history is too short to compare baseline and pruned.')

    h0 = history[0]
    hl = history[-1]

    base = MetricPack(
        name=f"baseline(step={h0.get('step', 0)})",
        params=float(h0.get('params')) if h0.get('params') is not None else None,
        flops=float(h0.get('macs')) if h0.get('macs') is not None else None,
    )
    pruned = MetricPack(
        name=f"pruned(step={hl.get('step', len(history)-1)})",
        params=float(hl.get('params')) if hl.get('params') is not None else None,
        flops=float(hl.get('macs')) if hl.get('macs') is not None else None,
    )
    return base, pruned


def _fmt_num(v: Optional[float]) -> str:
    if v is None:
        return '-'
    if abs(v) >= 1e9:
        return f'{v/1e9:.3f} G'
    if abs(v) >= 1e6:
        return f'{v/1e6:.3f} M'
    return f'{v:.4f}'


def _fmt_plain(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return '-'
    return f'{v:.{digits}f}'


def _higher_better(base: Optional[float], pruned: Optional[float]) -> Optional[float]:
    if base in (None, 0) or pruned is None:
        return None
    return (pruned - base) / base * 100.0


def _lower_better(base: Optional[float], pruned: Optional[float]) -> Optional[float]:
    if base in (None, 0) or pruned is None:
        return None
    return (base - pruned) / base * 100.0


def _fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return '-'
    tag = 'UP' if v >= 0 else 'DOWN'
    return f'{tag} {abs(v):.2f}%'


def _build_table(base: MetricPack, pruned: MetricPack) -> str:
    d_miou = _higher_better(base.miou, pruned.miou)
    d_params = _lower_better(base.params, pruned.params)
    d_flops = _lower_better(base.flops, pruned.flops)
    d_lat = _lower_better(base.latency_ms, pruned.latency_ms)
    d_fps = _higher_better(base.fps, pruned.fps)

    rows = [
        '| Metric | Baseline | Pruned | Improvement |',
        '|---|---:|---:|---:|',
        f'| mIoU (%) | {_fmt_plain(base.miou, 2)} | {_fmt_plain(pruned.miou, 2)} | {_fmt_delta(d_miou)} |',
        f'| Params | {_fmt_num(base.params)} | {_fmt_num(pruned.params)} | {_fmt_delta(d_params)} |',
        f'| FLOPs/MACs | {_fmt_num(base.flops)} | {_fmt_num(pruned.flops)} | {_fmt_delta(d_flops)} |',
        f'| Latency (ms) | {_fmt_plain(base.latency_ms)} | {_fmt_plain(pruned.latency_ms)} | {_fmt_delta(d_lat)} |',
        f'| FPS | {_fmt_plain(base.fps)} | {_fmt_plain(pruned.fps)} | {_fmt_delta(d_fps)} |',
    ]
    return '\n'.join(rows)


def _build_conclusion(base: MetricPack, pruned: MetricPack) -> str:
    d_params = _lower_better(base.params, pruned.params)
    d_flops = _lower_better(base.flops, pruned.flops)
    d_lat = _lower_better(base.latency_ms, pruned.latency_ms)
    d_fps = _higher_better(base.fps, pruned.fps)
    d_miou = _higher_better(base.miou, pruned.miou)

    lines = []
    if d_params is not None:
        ratio = (base.params / pruned.params) if (base.params and pruned.params) else None
        if ratio is not None:
            lines.append(f'- Params reduced by {d_params:.2f}% (compression ratio {ratio:.2f}x).')
        else:
            lines.append(f'- Params reduced by {d_params:.2f}%.')
    if d_flops is not None:
        lines.append(f'- FLOPs/MACs reduced by {d_flops:.2f}%.')
    if d_lat is not None:
        if d_lat >= 0:
            lines.append(f'- Latency reduced by {d_lat:.2f}%.')
        else:
            lines.append(f'- Latency increased by {abs(d_lat):.2f}%.')
    elif d_fps is not None:
        if d_fps >= 0:
            lines.append(f'- FPS improved by {d_fps:.2f}%.')
        else:
            lines.append(f'- FPS decreased by {abs(d_fps):.2f}%.')

    if d_miou is not None:
        if d_miou >= 0:
            lines.append(f'- mIoU improved by {d_miou:.2f}%.')
        else:
            drop = abs(d_miou)
            if drop <= 1.0:
                lines.append(f'- mIoU drop is small ({drop:.2f}%), usually acceptable.')
            else:
                lines.append(f'- mIoU dropped by {drop:.2f}%, fine-tuning is recommended.')

    if d_params is not None and d_params >= 30:
        final = 'Overall, pruning provides strong compression gains.'
    elif d_params is not None and d_params >= 10:
        final = 'Overall, pruning provides moderate compression gains.'
    else:
        final = 'Overall, compression gain is limited; consider stronger pruning or strategy tuning.'

    if d_miou is not None and d_miou < -1.0:
        final += ' Accuracy recovery via fine-tuning should be prioritized.'

    if not lines:
        lines.append('- Not enough comparable metrics in input JSONs.')

    lines.append(f'- Conclusion: {final}')
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Auto compare baseline vs pruned metrics')
    p.add_argument('--baseline-json', type=str, default=None)
    p.add_argument('--pruned-json', type=str, default=None)
    p.add_argument('--tp-summary', type=str, default=None)
    p.add_argument('--title', type=str, default='Baseline vs Pruned Comparison')
    p.add_argument('--out-md', type=str, default='exports/baseline_vs_pruned_report.md')
    p.add_argument('--out-json', type=str, default='exports/baseline_vs_pruned_compare.json')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    use_summary = args.tp_summary is not None
    use_pair = args.baseline_json is not None and args.pruned_json is not None
    if use_summary == use_pair:
        raise ValueError('Use either --tp-summary OR (--baseline-json and --pruned-json).')

    if use_summary:
        baseline, pruned = _from_tp_summary(_load_json(Path(args.tp_summary)))
        mode = 'tp-summary'
    else:
        baseline = _from_baseline_metrics('baseline', _load_json(Path(args.baseline_json)))
        pruned = _from_baseline_metrics('pruned', _load_json(Path(args.pruned_json)))
        mode = 'baseline+pruned'

    table = _build_table(baseline, pruned)
    conclusion = _build_conclusion(baseline, pruned)

    md = '\n'.join([
        f'# {args.title}',
        '',
        f'- Mode: `{mode}`',
        f'- Baseline: `{baseline.name}`',
        f'- Pruned: `{pruned.name}`',
        '',
        '## Final Comparison Table',
        '',
        table,
        '',
        '## Conclusion Sentences',
        '',
        conclusion,
        '',
    ])

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding='utf-8')

    payload = {
        'mode': mode,
        'baseline': baseline.__dict__,
        'pruned': pruned.__dict__,
        'table_markdown': table,
        'conclusion': conclusion,
        'report_markdown': str(out_md),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[OK] report markdown:', out_md)
    print('[OK] compare json   :', out_json)
    print('\n' + table)
    print('\n' + conclusion)


if __name__ == '__main__':
    main()
