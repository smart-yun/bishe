from __future__ import annotations
"""RailSem19 dataset adapter for MMSegmentation.

This project keeps split files (data/splits_mmseg/*.txt) where each line is an
image-relative path ending with .jpg, e.g.:

    rs19_val/rs02074.jpg

Masks live under data_prefix.seg_map_path with the same relative path but .png:

    rs19_val/rs02074.png

MMSeg 1.x no longer registers the legacy "CustomDataset" by default, so we
provide a tiny BaseSegDataset subclass that implements this jpg->png mapping.
"""



import os.path as osp
from typing import List

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class RS19JpgListDataset(BaseSegDataset):
    """RailSem19 dataset from a txt listing jpg paths.

    Expected ann_file format: one relative image path per line.
    """

    def load_data_list(self) -> List[dict]:
        # Resolve ann_file path.
        ann_file = self.ann_file
        if ann_file is None:
            raise ValueError('ann_file must be provided')

        # Note: BaseSegDataset will usually join ann_file/data_prefix with data_root.
        # We therefore try a couple of candidates instead of joining blindly.
        candidates = [ann_file]
        if not osp.isabs(ann_file):
            candidates.append(osp.join(self.data_root, ann_file))
        ann_path = None
        for p in candidates:
            if osp.exists(p):
                ann_path = p
                break
        if ann_path is None:
            raise FileNotFoundError(f'ann_file not found. tried: {candidates!r}')

        data_list: List[dict] = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            for raw in f:
                img_rel = raw.strip()
                if not img_rel:
                    continue

                # Map image -> mask by swapping suffix to self.seg_map_suffix.
                stem, _ = osp.splitext(img_rel)
                seg_rel = stem + self.seg_map_suffix

                # Important: in MMSeg 1.x, data_prefix is typically already expanded
                # to include data_root, so DO NOT prepend data_root again.
                img_prefix = self.data_prefix.get('img_path', '')
                seg_prefix = self.data_prefix.get('seg_map_path', '')

                img_path = osp.join(img_prefix, img_rel)
                seg_map_path = osp.join(seg_prefix, seg_rel)

                data_list.append(
                    dict(
                        img_path=img_path,
                        seg_map_path=seg_map_path,
                        reduce_zero_label=self.reduce_zero_label,
                        # Required by mmseg transforms (e.g. LoadAnnotations)
                        seg_fields=[],
                        img_fields=[],
                    )
                )

        return data_list
