"""
prepare_facebook_data.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains functionality for converting the raw 128-dimensional Facebook
Page-Page .npz data obtained from the course-linked source:

https://github.com/gayanku/facebook_page-page_dataset

The source archive uses different array names and edge storage from those
expected by PyG's FacebookPagePage loader. This script repackages the data
into the schema expected by the loader without otherwise modifying the
feature, label, or graph-topology values. The source page_name array is
retained as node-name metadata for later qualitative analysis, but is not
used as an input feature by the model.

References:

    Facebook Page-Page dataset:
    https://github.com/gayanku/facebook_page-page_dataset

    PyG FacebookPagePage:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/
    torch_geometric.datasets.FacebookPagePage.html
"""
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

DEFAULT_SOURCE: Final[Path] = Path("data/source/facebook.npz")
DEFAULT_OUTPUT: Final[Path] = Path("data/FacebookPagePage/raw/facebook.npz")


def convert_facebook_data(
        source_path: Path = DEFAULT_SOURCE,
        output_path: Path = DEFAULT_OUTPUT,
) -> None:
    """
        Convert the source Facebook Page-Page archive to the schema expected by
        PyG's FacebookPagePage loader.

        Args:
            source_path: Path to the original course-linked dataset archive.
            output_path: Path at which to write the PyG-compatible archive.
        """
    with np.load(source_path, allow_pickle=True) as source:
        features: NDArray[np.float32] = source["X"]
        target: NDArray[np.int8] = source["y"]
        edges: NDArray[np.int64] = np.column_stack(
            (source["edges_x"], source["edges_y"])
        )
        page_name: NDArray[np.object_] = source["page_name"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=features,
        target=target,
        edges=edges,
        page_name=page_name,
    )


def main() -> None:
    """Prepare the Facebook Page-Page dataset using the default paths."""
    convert_facebook_data()


if __name__ == "__main__":
    main()
