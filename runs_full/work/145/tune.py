import torch, triton
import triton.language as tl
import triton.testing as tt
import reference as ref
from sol import _linear_act

r=ref.Actor(4,4,4).cuda()
x=torch.rand(4,4,4,4).cuda().reshape(-1,4).contiguous()
W=[(r.fc1.weight,r.fc1.bias,1),(r.fc2.weight,r.fc2.bias,1),(r.fc3.weight,r.fc3.bias,2)]

def run(BN, nw, ns):
    def f():
        h=x
        for w,b,act in W:
            M,K=h.shape; N=w.shape[0]
            o=torch.empty((M,N),device=h.device,dtype=torch.float32)
            grid=(triton.cdiv(M,64),triton.cdiv(N,BN))
            _linear_act[grid](h,w,b,o,M,N,K,h.stride(0),h.stride(1),w.stride(0),w.stride(1),o.stride(0),o.stride(1),ACT=act,BLOCK_M=64,BLOCK_N=BN,BLOCK_K=16,num_warps=nw,num_stages=ns)
            h=o
        return h
    return tt.do_bench(f)

for BN in [64,128,256,512]:
    for nw in [2,4,8]:
        for ns in [1,2,3]:
            try:
                print(BN,nw,ns, round(run(BN,nw,ns),4))
            except Exception as e:
                print(BN,nw,ns,'ERR')
