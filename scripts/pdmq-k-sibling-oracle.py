#!/usr/bin/env python3
import argparse, math, random, struct
from dataclasses import dataclass

D=256; QK=32; QBLOCKS=D//QK

def f16_to_f32_bits(h): return struct.unpack('<e', struct.pack('<H', h & 0xffff))[0]
def f32_to_f16_bits(x): return struct.unpack('<H', struct.pack('<e', float(x)))[0]
def pack_i8x4(vs):
    out=0
    for i,v in enumerate(vs): out |= (int(v)&0xff) << (8*i)
    return out
def unpack_i8(w,l):
    v=(w>>(8*l))&0xff
    return v-256 if v>=128 else v
def pack_q4x8(cs):
    out=0
    for i,c in enumerate(cs): out |= (int(c)&15) << (4*i)
    return out
def unpack_q4(w,l): return (w>>(4*l))&15
@dataclass
class QPackRow: payload_i32:list; scales_f32:list
@dataclass
class Packed8KRow: payload_i32:list; scales_f16_bits:list

def expand_packed8_to_i8_words(k):
    out=[]
    for g in range(D//4):
        vals=[]
        for i in range(4):
            d=4*g+i
            vals.append(unpack_q4(k.payload_i32[d//8], d&7)-8)
        out.append(pack_i8x4(vals))
    return out

def scalar_qk_packed8(q,k):
    total=0.0
    for qb in range(QBLOCKS):
        acc=0
        for w8 in range(4):
            kw=k.payload_i32[qb*4+w8]
            for j in range(8):
                d=qb*QK+w8*8+j
                acc += unpack_i8(q.payload_i32[d//4], d&3) * (unpack_q4(kw,j)-8)
        total += float(acc)*q.scales_f32[qb]*f16_to_f32_bits(k.scales_f16_bits[qb])
    return total

def scalar_qk_expanded_i8(q,k):
    ki8=expand_packed8_to_i8_words(k)
    total=0.0
    for qb in range(QBLOCKS):
        acc=0
        for g in range(qb*(QK//4),(qb+1)*(QK//4)):
            for i in range(4): acc += unpack_i8(q.payload_i32[g], i) * unpack_i8(ki8[g], i)
        total += float(acc)*q.scales_f32[qb]*f16_to_f32_bits(k.scales_f16_bits[qb])
    return total

def pack_q_from_f32(vals):
    payload=[]; scales=[]
    for qb in range(QBLOCKS):
        block=vals[qb*QK:(qb+1)*QK]
        amax=max(abs(x) for x in block)
        s=amax/127.0 if amax else 1.0
        scales.append(s)
        for g in range(QK//4):
            payload.append(pack_i8x4(max(-127,min(127,round(block[g*4+i]/s))) for i in range(4)))
    return QPackRow(payload, scales)

def pack_k_packed8_from_f32(vals):
    payload=[]; scales=[]
    for qb in range(QBLOCKS):
        block=vals[qb*QK:(qb+1)*QK]
        maxv=max(block, key=lambda x: abs(x))
        d=maxv/-8.0
        scales.append(f32_to_f16_bits(d))
        inv=1.0/d if d else 0.0
        codes=[8 if inv == 0.0 else min(15,max(0,int(x*inv+8.5))) for x in block]
        for w in range(4): payload.append(pack_q4x8(codes[w*8:(w+1)*8]))
    return Packed8KRow(payload, scales)

def attention(qs, ks, v, causal, qk_fn):
    out=[]
    for qi,q in enumerate(qs):
        logits=[]; m=-math.inf
        for ki,k in enumerate(ks):
            l=qk_fn(q,k)
            if causal and ki > qi: l=-math.inf
            logits.append(l); m=max(m,l)
        probs=[0.0 if not math.isfinite(x) else math.exp(x-m) for x in logits]
        denom=sum(probs) or 1.0
        probs=[p/denom for p in probs]
        vd=len(v[0]); row=[0.0]*vd
        for p,vr in zip(probs,v):
            for d,x in enumerate(vr): row[d]+=p*x
        out.append(row)
    return out

def self_test():
    q=QPackRow([pack_i8x4([1,1,1,1]) for _ in range(D//4)], [1.0]*QBLOCKS)
    k=Packed8KRow([pack_q4x8([9]*8) for _ in range(D//8)], [f32_to_f16_bits(1.0)]*QBLOCKS)
    assert scalar_qk_packed8(q,k) == 256.0
    assert scalar_qk_expanded_i8(q,k) == 256.0
    q=QPackRow([pack_i8x4([-2,-2,-2,-2]) for _ in range(D//4)], [1.0]*QBLOCKS)
    k=Packed8KRow([pack_q4x8([7]*8) for _ in range(D//8)], [f32_to_f16_bits(1.0)]*QBLOCKS)
    assert scalar_qk_packed8(q,k) == 512.0
    assert scalar_qk_expanded_i8(q,k) == 512.0
    rng=random.Random(0x5eed1234)
    for _ in range(200):
        q=QPackRow([pack_i8x4([rng.randint(-127,127) for _ in range(4)]) for _ in range(D//4)], [rng.random()+0.001 for _ in range(QBLOCKS)])
        k=Packed8KRow([pack_q4x8([rng.randrange(16) for _ in range(8)]) for _ in range(D//8)], [f32_to_f16_bits(rng.uniform(-0.2,0.2) or -0.03125) for _ in range(QBLOCKS)])
        a=scalar_qk_packed8(q,k); b=scalar_qk_expanded_i8(q,k)
        assert abs(a-b) <= 1e-5*max(1.0,abs(a)), (a,b)
    qs=[pack_q_from_f32([rng.uniform(-2,2) for _ in range(D)]) for _ in range(4)]
    ks=[pack_k_packed8_from_f32([rng.uniform(-2,2) for _ in range(D)]) for _ in range(5)]
    v=[[rng.uniform(-1,1) for _ in range(17)] for _ in range(5)]
    a=attention(qs,ks,v,True,scalar_qk_packed8)
    b=attention(qs,ks,v,True,scalar_qk_expanded_i8)
    for ra,rb in zip(a,b):
        for x,y in zip(ra,rb): assert abs(x-y) <= 1e-6*max(1.0,abs(x)), (x,y)
    print('pdmq-k-sibling-oracle: packed8 self-test PASS')
if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test', action='store_true'); a=ap.parse_args()
    if a.self_test: self_test()
    else: ap.error('only --self-test is implemented')
