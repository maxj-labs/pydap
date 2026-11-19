"""Inlined data with no sibling <dmrpp:chunks> element

The same six cases as reproducer 10, without a <dmrpp:chunks> element next to
the inlined data. For <dmrpp:compact> the parser never assigns `codecs` or
`chunkmanifest`, the UnboundLocalError is caught, and the variable is dropped
with a generic warning.

Run: python 11_inline_data_without_chunks_element.py   (needs pydap[virtualizarr] and h5py)
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
    import base64
    import zlib

    def encode(array, element):
        raw = array.tobytes()
        if element == "missingdata":
            raw = zlib.compress(raw)
        return base64.b64encode(raw).decode()

    with h5py.File(DATA, "w") as f:
        f.create_dataset("unused", data=np.zeros(1))
    failures = []
    for element in ("compact", "missingdata"):
        for shape in ((), (10,), (2, 5)):
            expected = np.arange(int(np.prod(shape)), dtype="<i8").reshape(shape) + 7
            dims = "".join(f'<Dim size="{n}"/>' for n in shape)
            body = (
                f'<Int64 name="v">{dims}'
                f"<dmrpp:{element}>{encode(expected, element)}</dmrpp:{element}></Int64>"
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                got, error = attempt(lambda: open_sidecar(body)["v"][...])
            ok = same(got, expected) and got.shape == expected.shape
            shown = error or f"shape {got.shape}, values {got.tolist()}"
            if caught and not ok:
                shown += f" (warned: {caught[0].message})"
            print(f"{element:12s} {len(shape)}-D: {'ok' if ok else 'WRONG: ' + shown}")
            if not ok:
                failures.append(f"{element} {len(shape)}-D")
    report(bool(failures), "all six cases read back the encoded values", failures or "all ok")
