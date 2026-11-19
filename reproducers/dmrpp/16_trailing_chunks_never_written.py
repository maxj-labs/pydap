"""A variable whose last chunks were never written

HDF5 allocates no storage for a chunk nothing was written to, so the sidecar
lists only the chunks that exist. The parser sizes the chunk grid from the
listed chunk keys, so when the missing chunks are at the end the manifest is
smaller than the array's chunk grid.

Run: python 16_trailing_chunks_never_written.py   (needs pydap[virtualizarr] and h5py)
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
    with h5py.File(DATA, "w") as f:
        d = f.create_dataset("profile", shape=(4,), dtype="int32", chunks=(2,), fillvalue=-9999)
        d[:2] = [1, 2]
        tags = chunk_tags(d)
    body = (
        '<Dimension name="profile" size="4"/>'
        '<Int32 name="profile"><Dim name="/profile"/>'
        '<dmrpp:chunks fillValue="-9999" byteOrder="LE">'
        f"<dmrpp:chunkDimensionSizes>2</dmrpp:chunkDimensionSizes>{tags}"
        "</dmrpp:chunks></Int32>"
    )
    expected = np.array([1, 2, -9999, -9999], dtype="int32")
    got, error = attempt(lambda: open_sidecar(body)["profile"][...])
    report(not same(got, expected), expected.tolist(), error or got.tolist())
