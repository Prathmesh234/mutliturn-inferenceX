import torch, triton
import triton.language as tl

@triton.jit
def k(x_ptr,w_ptr,b_ptr,out_ptr,N,IC:tl.constexpr,OC:tl.constexpr,H:tl.constexpr,W:tl.constexpr,add_res:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); total=N*OC*H*W; mask=offs<total
    ow=offs%W; oh=(offs//W)%H; oc=(offs//(W*H))%OC; n=offs//(W*H*OC)
    acc_a=tl.load(b_ptr+oc,mask=mask,other=0.0); acc_b=tl.load(b_ptr+(oc+OC),mask=mask,other=0.0)
    for ci in range(IC):
        for kh in range(3):
            ih=oh+kh-1; vy=(ih>=0)&(ih<H)
            for kw in range(3):
                iw=ow+kw-1; vx=(iw>=0)&(iw<W); valid=mask&vy&vx
                xoff=((n*IC+ci)*H+ih)*W+iw; xoff=tl.where(valid,xoff,0)
                xv=tl.load(x_ptr+xoff,mask=valid,other=0.0)
                wa=tl.load(w_ptr+(((oc*IC+ci)*3+kh)*3+kw),mask=mask,other=0.0)
                wb=tl.load(w_ptr+((((oc+OC)*IC+ci)*3+kh)*3+kw),mask=mask,other=0.0)
                acc_a+=xv*wa; acc_b+=xv*wb
    res=tl.maximum(acc_a,acc_b)
    if add_res:
        r=tl.load(out_ptr+offs,mask=mask,other=0.0); res=res+r
    tl.store(out_ptr+offs,res,mask=mask)

from reference import resblock, get_inputs, get_init_inputs
a,kk=get_init_inputs(); ref=resblock(*a,**kk).cuda()
x=get_inputs()[0].cuda().contiguous()
raw=ref.conv1.filter(x); mx=torch.maximum(raw[:,0:4],raw[:,4:8])
N,IC,H,W=x.shape; OC=4
o=torch.empty(N,OC,H,W,device='cuda')
w=ref.conv1.filter.weight.contiguous(); b=ref.conv1.filter.bias.contiguous()
k[(1,)](x,w,b,o,N,IC,OC,H,W,False,BLOCK=256,num_warps=4)
print('err nw4', (mx-o).abs().max().item())
o2=torch.empty(N,OC,H,W,device='cuda')
k[(1,)](x,w,b,o2,N,IC,OC,H,W,False,BLOCK=256)
print('err default', (mx-o2).abs().max().item())

import sol
o3=torch.empty(N,OC,H,W,device='cuda')
sol._run_mfm(ref.conv1.filter, x, o3, False)
print('sol._run_mfm err', (mx-o3).abs().max().item())
print('sol kernel is k?', sol._mfm_conv_kernel.fn is k.fn if hasattr(sol._mfm_conv_kernel,'fn') else 'n/a')
import inspect
print('sol src lines', len(inspect.getsource(sol._mfm_conv_kernel).splitlines()))
