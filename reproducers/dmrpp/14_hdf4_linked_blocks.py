"""One HDF4 variable stored as <dmrpp:block> elements makes the whole file unreadable

HDF4 SDS data stored in linked blocks (an appendable SDS, AIRS Level 3) is
described by <dmrpp:block> elements instead of <dmrpp:chunk>. The parser only
looks for <dmrpp:chunk>, builds an empty manifest, and the error that raises
is for the whole store: `NormalData` beside it cannot be opened either.
The sidecar is modelled on Hyrax's test file `SDS_unlimited_2.hdf.dmrpp`.

Run: python 14_hdf4_linked_blocks.py   (needs pydap[virtualizarr] and h5py)
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
    stored = np.arange(130, dtype="<i4") + 1
    DATA.write_bytes(stored.tobytes())  # raw bytes are enough: only offsets matter
    body = (
        '<Dimension name="fakeDim0" size="12"/><Dimension name="fakeDim1" size="10"/>'
        '<Int32 name="AppendableData"><Dim name="/fakeDim0"/><Dim name="/fakeDim1"/>'
        '<dmrpp:chunks byteOrder="LE">'
        '<dmrpp:block offset="0" nBytes="400"/>'
        '<dmrpp:block offset="400" nBytes="40"/>'
        '<dmrpp:block offset="440" nBytes="40"/>'
        "</dmrpp:chunks></Int32>"
        '<Int32 name="NormalData"><Dim name="/fakeDim1"/>'
        '<dmrpp:chunks byteOrder="LE">'
        '<dmrpp:chunk offset="480" nBytes="40" chunkPositionInArray="[0]"/>'
        "</dmrpp:chunks></Int32>"
    )
    expected = stored[120:]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got, error = attempt(lambda: open_sidecar(body)["NormalData"][...])
    print("warnings:", [str(w.message) for w in caught])
    report(not same(got, expected), expected.tolist(), error or got.tolist())
