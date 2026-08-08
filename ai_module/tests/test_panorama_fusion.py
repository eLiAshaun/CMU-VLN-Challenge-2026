import numpy as np

from integrations.perception.panorama_fusion import _bbox, _overlap


def test_overlap_reports_iou_and_containment_for_duplicate_masks():
    first = np.zeros((10, 20), dtype=bool)
    second = np.zeros_like(first)
    first[2:8, 3:9] = True
    second[3:7, 4:8] = True

    iou, containment = _overlap(first, second)

    assert iou == 16 / 36
    assert containment == 1.0
    assert _bbox(first) == [3, 2, 9, 8]
