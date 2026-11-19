"""Filters listed as "deflate shuffle" are applied in the listed order

Older build_dmrpp versions write compressionType="deflate shuffle"; newer ones
write "shuffle deflate". HDF5 can only apply these two filters in one order
(shuffle, then deflate), so both spellings describe the same bytes. The parser
builds the codec list in the listed order, so the older spelling cannot be decoded.
Seen in PO.DAAC ECCO sidecars.

Run: python 01_deflate_shuffle_order.py   (needs pydap[virtualizarr] and h5py)
"""

import pathlib
import sys
import tempfile
import warnings

import h5py
import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

import pydap
from pydap.virtualizarr.parser import DMRParser, DMRPPParser

HEADER = (
    '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
    '<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" '
    'xmlns:dmrpp="http://xml.opendap.org/dap/dmrpp/1.0.0#" '
    'dapVersion="4.0" dmrVersion="1.0" name="t.h5">\n'
)
TMP = pathlib.Path(tempfile.mkdtemp())
DATA = TMP / "t.h5"


def chunk_tags(dataset, extra=""):
    """One <dmrpp:chunk> per stored chunk of an h5py dataset."""
    tags = []
    for index in range(dataset.id.get_num_chunks()):
        info = dataset.id.get_chunk_info(index)
        position = ",".join(map(str, info.chunk_offset))
        tags.append(
            f'<dmrpp:chunk offset="{info.byte_offset}" nBytes="{info.size}" '
            f'chunkPositionInArray="[{position}]"{extra}/>'
        )
    return "".join(tags)


def open_sidecar(body):
    """Write the sidecar next to the data file and open it with pydap's parser."""
    (TMP / "t.h5.dmrpp").write_text(HEADER + body + "</Dataset>\n")
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    store = DMRParser(f"file://{DATA}.dmrpp", object_store=registry).parse_dataset()
    return zarr.open_group(store, mode="r", zarr_format=3)


def report(reproduces, expected, got):
    print(f"pydap {pydap.__version__}")
    print(f"expected: {expected}")
    print(f"got:      {got}")
    print("REPRODUCES" if reproduces else "DOES NOT REPRODUCE")
    sys.exit(1 if reproduces else 0)


def attempt(function):
    """Return (value, None), or (None, 'ErrorType: message') if it raises."""
    try:
        return function(), None
    except Exception as error:  # the reproducer reports whatever is raised
        return None, f"{type(error).__name__}: {error}"


def same(left, right):
    return left is not None and np.array_equal(np.asarray(left), np.asarray(right))


if __name__ == "__main__":
    expected = np.arange(24, dtype="float32").reshape(4, 6) * 1.5
    with h5py.File(DATA, "w") as f:
        tags = chunk_tags(
            f.create_dataset(
                "data", data=expected, chunks=(2, 6), shuffle=True, compression="gzip"
            )
        )
    body = (
        '<Dimension name="y" size="4"/><Dimension name="x" size="6"/>'
        '<Float32 name="data"><Dim name="/y"/><Dim name="/x"/>'
        '<dmrpp:chunks compressionType="deflate shuffle" byteOrder="LE">'
        f"<dmrpp:chunkDimensionSizes>2 6</dmrpp:chunkDimensionSizes>{tags}"
        "</dmrpp:chunks></Float32>"
    )
    got, error = attempt(lambda: open_sidecar(body)["data"][...])
    report(not same(got, expected), expected.tolist(), error or got.tolist())
