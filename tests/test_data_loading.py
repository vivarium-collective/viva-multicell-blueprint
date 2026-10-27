import numpy as np
import pytest
from skimage.io import imsave

from viva_multicell_blueprint.blueprint.data_loading import from_cell_table, from_label_image


def _png(tmp_path, arr):
    p = tmp_path / "labels.png"
    imsave(p, arr.astype(np.uint8), check_contrast=False)
    return p


def test_label_image_centroids_sizes(tmp_path):
    a = np.zeros((10, 12), dtype=np.uint8)
    a[0:2, 0:2] = 1      # 4 px, centroid x=0.5 y=0.5
    a[4:7, 5:9] = 2      # 12 px, centroid x=6.5 y=5.0
    a[8:10, 10:12] = 3   # 4 px
    cells = from_label_image(_png(tmp_path, a), type_map={1: "A", 2: "B", 3: "A"})
    assert [c["id"] for c in cells] == [1, 2, 3]
    assert [c["type"] for c in cells] == ["A", "B", "A"]
    assert [c["size"] for c in cells] == [4, 12, 4]
    assert cells[0]["centroid"] == pytest.approx((0.5, 0.5))
    assert cells[1]["centroid"] == pytest.approx((6.5, 5.0))  # (x, y)


def test_label_image_hole_and_background(tmp_path):
    a = np.zeros((7, 7), dtype=np.uint8)
    a[1:6, 1:6] = 1
    a[3, 3] = 0  # hole: medium, not a cell, not counted in size
    cells = from_label_image(_png(tmp_path, a))
    assert len(cells) == 1
    assert cells[0]["size"] == 24
    assert cells[0]["type"] == "cell"  # default when no type_map
    assert cells[0]["centroid"] == pytest.approx((3.0, 3.0))


def test_label_image_unmapped_label_raises(tmp_path):
    a = np.zeros((3, 3), dtype=np.uint8)
    a[0, 0] = 1
    with pytest.raises(KeyError):
        from_label_image(_png(tmp_path, a), type_map={2: "B"})


def test_cell_table_shape(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("id,x,y,type,size\n1,1.5,2.5,A,10\n2,4,5,B,20.5\n")
    cells = from_cell_table(p)
    assert cells == [
        {"id": 1, "type": "A", "centroid": (1.5, 2.5), "size": 10.0},
        {"id": 2, "type": "B", "centroid": (4.0, 5.0), "size": 20.5},
    ]


def test_cell_table_missing_column(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("id,x,y,type\n1,1,2,A\n")
    with pytest.raises(ValueError, match="size"):
        from_cell_table(p)


def test_single_cell_degenerate(tmp_path):
    a = np.zeros((4, 4), dtype=np.uint8)
    a[1:3, 1:3] = 5
    cells = from_label_image(_png(tmp_path, a))
    assert len(cells) == 1 and cells[0]["id"] == 5 and cells[0]["size"] == 4
    p = tmp_path / "one.csv"
    p.write_text("id,x,y,type,size\n7,0,0,A,1\n")
    assert len(from_cell_table(p)) == 1


def test_cell_table_type_named_NA(tmp_path):
    p = tmp_path / "na.csv"
    p.write_text("id,x,y,type,size\n1,0,0,NA,1\n2,1,1,None,1\n")
    assert [c["type"] for c in from_cell_table(p)] == ["NA", "None"]


def test_unmapped_label_message(tmp_path):
    a = np.zeros((3, 3), dtype=np.uint8)
    a[0, 0] = 1
    with pytest.raises(KeyError, match="label 1 not in type_map"):
        from_label_image(_png(tmp_path, a), type_map={2: "B"})
