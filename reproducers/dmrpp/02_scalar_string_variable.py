"""A scalar String variable is given a chunk manifest of the wrong shape

Sidecars describe a scalar String (a CF "crs" container variable, say) with
fillValue="unsupported-string" and one chunk. The variable cannot be decoded as
a Zarr array, but it should either be skipped with a warning or parsed without
breaking; the numeric variables beside it must stay readable.
Seen in ORNL CMS_CO2_Fluxes_TBMO_1315.

Run: python 02_scalar_string_variable.py   (needs pydap[virtualizarr] and h5py)
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
    expected = np.arange(6, dtype="float64")
    with h5py.File(DATA, "w") as f:
        tags = chunk_tags(f.create_dataset("data", data=expected, chunks=(6,)))
    body = (
        '<Dimension name="x" size="6"/>'
        '<Float64 name="data"><Dim name="/x"/><dmrpp:chunks byteOrder="LE">'
        f"<dmrpp:chunkDimensionSizes>6</dmrpp:chunkDimensionSizes>{tags}"
        "</dmrpp:chunks></Float64>"
        '<String name="crs"><Attribute name="grid_mapping_name" type="String">'
        "<Value>latitude_longitude</Value></Attribute>"
        '<dmrpp:chunks fillValue="unsupported-string">'
        '<dmrpp:chunk offset="14258254" nBytes="1"/></dmrpp:chunks></String>'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got, error = attempt(lambda: open_sidecar(body)["data"][...])
    print("warnings:", [str(w.message) for w in caught])
    report(not same(got, expected), expected.tolist(), error or got.tolist())
