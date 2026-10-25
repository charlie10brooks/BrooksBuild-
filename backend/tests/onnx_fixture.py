"""Builds a tiny valid ONNX model without the ``onnx`` package (hand-encoded protobuf) so the onnxruntime path is really exercised.

Graph: x[1,3,h,w] --Concat(axis=1, x4)--> y[1,12,h,w].  Channels 0..2 hold R,G,B; argmax therefore picks the brightest channel
of each pixel (first copy wins ties), i.e. a 'dominant colour' segmentation network.
"""
import json
from pathlib import Path


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint(field << 3 | wire)


def _ld(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _s(field: int, s: str) -> bytes:
    return _ld(field, s.encode())


def _i(field: int, v: int) -> bytes:
    return _tag(field, 0) + _varint(v)


def _dim_param(name: str) -> bytes:
    return _ld(1, _s(2, name))


def _dim_val(v: int) -> bytes:
    return _ld(1, _i(1, v))


def _value_info(name: str, dims: list) -> bytes:
    shape = b"".join(dims)
    tensor = _i(1, 1) + _ld(2, shape)                 # elem_type FLOAT, shape
    typ = _ld(1, tensor)                              # TypeProto.tensor_type
    return _s(1, name) + _ld(2, typ)


def build_model() -> bytes:
    node = _s(1, "x") + _s(1, "x") + _s(1, "x") + _s(1, "x") + _s(2, "y") + _s(3, "cat") + _s(4, "Concat") + \
        _ld(5, _s(1, "axis") + _i(3, 1) + _i(20, 2))
    graph = _ld(1, node) + _s(2, "dominant_colour") + \
        _ld(11, _value_info("x", [_dim_val(1), _dim_val(3), _dim_param("h"), _dim_param("w")])) + \
        _ld(12, _value_info("y", [_dim_val(1), _dim_val(12), _dim_param("h"), _dim_param("w")]))
    opset = _s(1, "") + _i(2, 13)
    return _i(1, 8) + _ld(8, opset) + _s(2, "terrascope-test") + _ld(7, graph)


def write_model(dir_: Path, class_map=None) -> Path:
    p = Path(dir_) / "dominant.onnx"
    p.write_bytes(build_model())
    p.with_suffix(".json").write_text(json.dumps({"name": "dominant-colour-test", "input_size": None, "scale": 255, "mean": [0, 0, 0],
                                                  "std": [1, 1, 1], "class_map": class_map or {"0": 4, "1": 3, "2": 8}, "context_px": 0}))
    return p
