import torch, triton
import triton.language as tl
import triton.testing as tt
import importlib, sol
from reference import resblock, get_inputs, get_init_inputs

# monkeypatch _run_mfm to accept num_warps/BLOCK
def make_run(nw, BLOCK):
    def run(filt, x, out, res, add_res):
        N, IC, H, W = x.shape
        OC = filt.out_channels // 2
        w = filt.weight.contiguous(); b = filt.bias.contiguous()
        total = N*OC*H*W
        grid = (triton.cdiv(total, BLOCK),)
        res_arg = res if res is not None else x
        sol._mfm_conv_kernel[grid](x,w,b,res_arg,out,N,IC,OC,H,W,add_res,BLOCK=BLOCK,num_warps=nw)
        return out
    return run

a,k=get_init_inputs(); ref=resblock(*a,**k).cuda()
new=sol.resblockNew(*a,**k).cuda(); new.load_state_dict(ref.state_dict())
x=get_inputs()[0].cuda()
tref=tt.do_bench(lambda: ref(x))
for nw in [1,2,4,8]:
    for BLOCK in [64,128,256]:
        sol._run_mfm = make_run(nw, BLOCK)
        t=tt.do_bench(lambda: new(x))
        print(f'nw={nw} BLOCK={BLOCK} speedup={tref/t:.3f}')
