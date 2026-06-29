# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Multi-layer self-attention ONNX generator (TRT-28040 perf demo).
# Stacks N attention blocks so the model is compute-bound (the single-layer
# sample is overhead-bound at batch=1, hiding the multi-GPU speedup). Also emits
# a matching polygraphy context-parallel sharding hint (one attention_layers
# entry per block, keyed by each block's uniquely-named q_scaled_i tensor).
#
#   python3 create_onnx_multilayer.py --layers 8 --output attn_ml.onnx --hint attn_ml_hint.json
import argparse
import json
import math

import numpy as np
import onnx
import onnx_graphsurgeon as gs

NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_DIM = NUM_HEADS * HEAD_DIM  # 4096
OPSET = 17


@gs.Graph.register()
def op1(self, op, a, attrs=None):
    return self.layer(op=op, inputs=[a], attrs=attrs or {}, outputs=[op + "_o"])[0]


@gs.Graph.register()
def matmul(self, a, b):
    return self.layer(op="MatMul", inputs=[a, b], outputs=["mm_o"])[0]


@gs.Graph.register()
def transpose(self, a, perm):
    return self.layer(
        op="Transpose", inputs=[a], attrs={"perm": perm}, outputs=["tr_o"]
    )[0]


@gs.Graph.register()
def reshape(self, data, shape):
    return self.layer(
        op="Reshape", inputs=[data, shape], attrs={"allowzero": 0}, outputs=["rs_o"]
    )[0]


@gs.Graph.register()
def softmax(self, a, axis=-1):
    return self.layer(op="Softmax", inputs=[a], attrs={"axis": axis}, outputs=["sm_o"])[
        0
    ]


@gs.Graph.register()
def cast(self, a, to):
    return self.layer(op="Cast", inputs=[a], attrs={"to": to}, outputs=["cast_o"])[0]


@gs.Graph.register()
def binop(self, op, a, b):
    return self.layer(op=op, inputs=[a, b], outputs=[op + "_o"])[0]


@gs.Graph.register()
def reduce_mean(self, a, axes):
    return self.layer(
        op="ReduceMean",
        inputs=[a],
        attrs={"axes": axes, "keepdims": 1},
        outputs=["rm_o"],
    )[0]


@gs.Graph.register()
def shape_op(self, a):
    return self.layer(op="Shape", inputs=[a], outputs=["sh_o"])[0]


@gs.Graph.register()
def gather(self, data, indices):
    return self.layer(
        op="Gather", inputs=[data, indices], attrs={"axis": 0}, outputs=["ga_o"]
    )[0]


@gs.Graph.register()
def unsqueeze(self, a, axes):
    return self.layer(op="Unsqueeze", inputs=[a, axes], outputs=["un_o"])[0]


@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(
        op="Concat", inputs=inputs, attrs={"axis": axis}, outputs=["cc_o"]
    )[0]


def attention_block(graph, x, idx, rng):
    axes_0 = np.array([0], dtype=np.int64)

    def w():
        # Scale ~1/sqrt(hidden) so stacked matmuls stay numerically bounded
        # (no residual/norm between blocks in this synthetic model).
        return (
            rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)) / math.sqrt(HIDDEN_DIM)
        ).astype(np.float16)

    def s32(v):
        return np.array([v], dtype=np.float32)

    q_proj = graph.matmul(x, w())
    k_proj = graph.matmul(x, w())
    v_proj = graph.matmul(x, w())

    def to_heads(proj):
        sh = graph.shape_op(proj)
        sd = graph.unsqueeze(graph.gather(sh, np.array(0, dtype=np.int64)), axes_0)
        bd = graph.unsqueeze(graph.gather(sh, np.array(1, dtype=np.int64)), axes_0)
        tgt = graph.concat(
            [
                sd,
                bd,
                np.array([NUM_HEADS], dtype=np.int64),
                np.array([HEAD_DIM], dtype=np.int64),
            ]
        )
        return graph.reshape(proj, tgt)

    q4, k4, v4 = to_heads(q_proj), to_heads(k_proj), to_heads(v_proj)

    def rmsnorm(t):
        f = graph.cast(t, onnx.TensorProto.FLOAT)
        sq = graph.binop("Pow", f, s32(2.0))
        mean = graph.reduce_mean(sq, axes=[-1])
        rms = graph.op1("Sqrt", graph.binop("Add", mean, s32(1e-6)))
        inv = graph.binop("Div", s32(1.0), rms)
        nf = graph.binop("Mul", f, inv)
        n16 = graph.cast(nf, onnx.TensorProto.FLOAT16)
        wt = rng.standard_normal((1, 1, 1, HEAD_DIM)).astype(np.float16)
        return graph.binop("Mul", wt, n16)

    qn, kn = rmsnorm(q4), rmsnorm(k4)
    qa = graph.transpose(qn, perm=[1, 2, 0, 3])
    ka = graph.transpose(kn, perm=[1, 2, 0, 3])
    va = graph.transpose(v4, perm=[1, 2, 0, 3])

    def to_attn(t):
        sh = graph.shape_op(t)
        b = graph.unsqueeze(graph.gather(sh, np.array(0, dtype=np.int64)), axes_0)
        h = graph.unsqueeze(graph.gather(sh, np.array(1, dtype=np.int64)), axes_0)
        d = graph.unsqueeze(graph.gather(sh, np.array(3, dtype=np.int64)), axes_0)
        tgt = graph.concat([b, h, np.array([-1], dtype=np.int64), d])
        return graph.reshape(t, tgt)

    qr, kr, vr = to_attn(qa), to_attn(ka), to_attn(va)
    sc = np.array([math.sqrt(math.sqrt(1.0 / HEAD_DIM))], dtype=np.float16)
    q_scaled = graph.binop("Mul", qr, sc)
    q_scaled.name = "q_scaled_%d" % idx  # hint targets this per layer
    kt = graph.transpose(kr, perm=[0, 1, 3, 2])
    ks = graph.binop("Mul", kt, sc)
    qk = graph.matmul(q_scaled, ks)
    aw = graph.softmax(qk, axis=-1)
    ao = graph.matmul(aw, vr)
    at = graph.transpose(ao, perm=[2, 0, 1, 3])
    sh = graph.shape_op(at)
    sd = graph.unsqueeze(graph.gather(sh, np.array(0, dtype=np.int64)), axes_0)
    bd = graph.unsqueeze(graph.gather(sh, np.array(1, dtype=np.int64)), axes_0)
    hh = graph.unsqueeze(
        graph.binop(
            "Mul",
            graph.gather(sh, np.array(2, dtype=np.int64)),
            graph.gather(sh, np.array(3, dtype=np.int64)),
        ),
        axes_0,
    )
    flat = graph.reshape(at, graph.concat([sd, bd, hh]))
    return graph.matmul(flat, w())  # output projection -> block output


def build(layers):
    rng = np.random.default_rng(42)
    graph = gs.Graph(opset=OPSET)
    x = gs.Variable(
        "input", dtype=np.float16, shape=["sequence_length", "batch_size", HIDDEN_DIM]
    )
    graph.inputs = [x]
    for i in range(layers):
        x = attention_block(graph, x, i, rng)
    x.name = "output"
    x.dtype = np.float16
    x.shape = ["sequence_length", "batch_size", HIDDEN_DIM]
    graph.outputs = [x]
    graph.cleanup().toposort()
    m = gs.export_onnx(graph)
    m.ir_version = 8
    return m


def make_hint(layers, path):
    hint = {
        "parallelism": "CP",
        "attention_layers": [
            {
                "q": "q_scaled_%d" % i,
                "gather_kv": True,
                "gather_q": False,
                "replace": None,
                "polygraphy_class": "AttentionLayerHint",
            }
            for i in range(layers)
        ],
        "dist_collectives": {
            "group_size": 0,
            "root": -1,
            "nb_rank": 2,
            "reduce_op": "max",
            "groups": [],
            "polygraphy_class": "DistCollective",
        },
        "inputs": [
            {
                "name": "input",
                "seq_len_idx": 0,
                "rank": 3,
                "polygraphy_class": "ShardTensor",
            }
        ],
        "outputs": [
            {
                "name": "output",
                "seq_len_idx": 0,
                "rank": 3,
                "polygraphy_class": "ShardTensor",
            }
        ],
        "k_seq_len_idx": 3,
        "v_seq_len_idx": 2,
        "kv_rank": 4,
        "polygraphy_class": "ShardHints",
    }
    with open(path, "w") as f:
        json.dump(hint, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--output", default="attn_ml.onnx")
    ap.add_argument("--hint", default="attn_ml_hint.json")
    a = ap.parse_args()
    m = build(a.layers)
    onnx.save(m, a.output)  # inline weights (keep <2GB protobuf limit)
    make_hint(a.layers, a.hint)
    print(
        "Saved %s (%d layers, %d nodes) + hint %s"
        % (a.output, a.layers, len(m.graph.node), a.hint)
    )


if __name__ == "__main__":
    main()
