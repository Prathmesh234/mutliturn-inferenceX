import torch, triton
import triton.language as tl

@triton.jit
def k(x_ptr,w_ptr,b_ptr,o_ptr,N,IC:tl.constexpr,OC:tl.constexpr,H:tl.constexpr,W:tl.constexpr,BLOCK:tl.constexpr):
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
    tl.store(o_ptr+offs,res,mask=mask)

from reference import resblock, get_inputs, get_init_inputs
a,kk=get_init_inputs(); ref=resblock(*a,**kk).cuda()
x=get_inputs()[0].cuda().contiguous()
raw=ref.conv1.filter(x); mx=torch.maximum(raw[:,0:4],raw[:,4:8])
N,IC,H,W=x.shape; OC=4
o=torch.empty(N,OC,H,W,device='cuda')
w=ref.conv1.filter.weight.contiguous(); b=ref.conv1.filter.bias.contiguous()
k[(triton.cdiv(N*OC*H*W,256),)](x,w,b,o,N,IC,OC,H,W,BLOCK=256)
print('err', (mx-o).abs().max().item())
