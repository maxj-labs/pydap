"""Attributes in different containers with the same name overwrite each other

Container attributes are flattened into their parent. Two containers that
each hold an attribute called `units` leave one `units` on the variable: the
last one wins and nothing is reported.

Run: python 13_container_attributes_overwrite.py   (needs pydap[virtualizarr] and h5py)
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
    xml = (
        HEADER + '<Dimension name="x" size="2"/><Int32 name="v"><Dim name="/x"/>'
        '<Attribute name="first" type="Container"><Attribute name="units" type="String">'
        "<Value>metres</Value></Attribute></Attribute>"
        '<Attribute name="second" type="Container"><Attribute name="units" type="String">'
        "<Value>seconds</Value></Attribute></Attribute>"
        "<dmrpp:chunks><dmrpp:chunkDimensionSizes>2</dmrpp:chunkDimensionSizes>"
        '<dmrpp:chunk offset="10" nBytes="8" chunkPositionInArray="[0]"/>'
        "</dmrpp:chunks></Int32></Dataset>"
    )
    from xml.etree import ElementTree as ET

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        variables, _ = DMRPPParser(root=ET.fromstring(xml)).parse_dataset()
    attributes = variables["v"]["attributes"]
    print("warnings:", [str(w.message) for w in caught])
    report(
        len(attributes) == 1 and not caught,
        "both values kept (e.g. first.units and second.units), or a warning",
        f"{attributes}, {len(caught)} warning(s)",
    )
