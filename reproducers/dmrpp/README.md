# Reproducers for pydap's DMR++ parser (`pydap.virtualizarr.parser`)

One standalone script per problem. Each writes a small HDF5 file (or raw bytes) and a
hand-written DMR++ sidecar to a temporary directory, opens it with `DMRParser`, reads
values through the `ManifestStore`, and prints what it expected, what it got, and
`REPRODUCES` (exit 1) or `DOES NOT REPRODUCE` (exit 0).

Run one: `python 01_deflate_shuffle_order.py` in an environment with
`pydap[virtualizarr]` and `h5py`.

## Results, 2026-11-19

Environment: Linux, Python 3.13, virtualizarr 2.8.0, zarr 3.4.2, obstore 0.12.1.
"main" is pydap `b130d60` (3.5.12.dev5). "fixed" is branch `fix/dmrpp-all` of
`maxj-labs/pydap`, which merges all the fix branches below.

Not run on the 3.5.11 release. The parser there differs from `main` only by #742
(inlined data was base64 text under the key "0.0"), so 10 and 11 fail differently on
3.5.11 and the others should behave the same; that is from the diff, not from a run.

| Script | What goes wrong on main | main | fixed | Fix branch |
|---|---|---|---|---|
| 01 | `compressionType="deflate shuffle"` cannot be decoded | reproduces | fixed | `fix/dmrpp-filter-order` |
| 02 | a scalar String variable stops every variable from opening | reproduces | fixed (skipped, warning) | `fix/dmrpp-string-variables` |
| 03 | a numeric scalar with stored data raises on read | reproduces | fixed | `fix/dmrpp-scalar-and-unwritten` |
| 04 | a never-written variable stops every variable from opening | reproduces | fixed (fill value) | `fix/dmrpp-scalar-and-unwritten` |
| 05 | `<dmrpp:FixedLengthStringArray>` stops every variable from opening | reproduces | fixed | `fix/dmrpp-string-variables` |
| 06 | `byteOrder="BE"` ignored: wrong values, no error | reproduces | fixed | `fix/dmrpp-byte-order` |
| 07 | placeholder `href` stops every variable from opening | reproduces | fixed (skipped, warning) | `fix/dmrpp-chunk-href` |
| 08 | a chunk without `href` inherits the previous chunk's | reproduces | fixed | `fix/dmrpp-chunk-href` |
| 09 | the caller's registry is replaced, external chunks unreachable | reproduces | fixed | `fix/dmrpp-chunk-href` |
| 10 | 2-D inlined data cannot be read | reproduces | fixed | `fix/dmrpp-inline-data` |
| 11 | `<dmrpp:compact>` without `<dmrpp:chunks>` drops the variable | reproduces | fixed | `fix/dmrpp-inline-data` |
| 12 | an AttributeError from the store becomes an XML syntax error | reproduces | fixed | `fix/dmrpp-constructor-errors` |
| 13 | same-named attributes in two containers overwrite silently | reproduces | warns (last one still kept) | `fix/dmrpp-container-attributes` |
| 14 | an HDF4 `<dmrpp:block>` variable stops every variable from opening | reproduces | fixed (skipped, warning) | `fix/dmrpp-scalar-and-unwritten` |
| 15 | validation issues lost because the inner parser is built twice | does not reproduce | n/a | none |
| 16 | trailing chunks never written: IndexError on read | reproduces | fixed (fill value) | `fix/dmrpp-scalar-and-unwritten` |
| 17 | a Container attribute inside a Container raises KeyError | reproduces | fixed | `fix/dmrpp-container-attributes` |

Script 16 prints asyncio "Task exception was never retrieved" noise on stderr before
its result on `main`.

## Version floor (no script)

`pydap[virtualizarr]` allows `virtualizarr>=2.6`. Running script 10 against pydap
`main` with virtualizarr 2.6.0 and zarr 3.1.6 fails all six cases with "Each chunk
entry must be of the form dict(path=<str>, offset=<int>, length=<int>)". With
virtualizarr 2.6.1 the result is the same as with 2.8.0. Branch
`fix/virtualizarr-floor` raises the floor to 2.6.1.

## Where the cases come from

- 01, 02, 04, 05, 07: seen in NASA sidecars (PO.DAAC ECCO, ORNL CMS_CO2 and ATom,
  PO.DAAC OMG). The scripts are synthetic stand-ins; the real granules were only
  checked against VirtualiZarr's own parser, not pydap's.
- 14: `<dmrpp:block>` is in Hyrax's test file `SDS_unlimited_2.hdf.dmrpp` and in AIRS
  Level 3 sidecars.
- 17: legal DAP4, not found in the 20 real sidecars on this machine.
- The rest come from reading the parser source.

## Not covered

No reproducer and no fix for: the unbounded read of the sidecar, the Expat version
check, pydap importing two helpers outside VirtualiZarr's documented API, and the
`"__inline__"` path sentinel (harmless). These are design questions, not failures a
script can show.
