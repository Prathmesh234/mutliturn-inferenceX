import torch, triton, sol
from reference import resblock, get_inputs, get_init_inputs
a,kk=get_init_inputs(); ref=resblock(*a,**kk).cuda()
x=get_inputs()[0].cuda().contiguous()
raw=ref.conv1.filter(x); mx=torch.maximum(raw[:,0:4],raw[:,4:8])
N,IC,H,W=x.shape; OC=4
w=ref.conv1.filter.weight.contiguous(); b=ref.conv1.filter.bias.contiguous()

# launch sol's kernel directly, dbg3 style
o=torch.empty(N,OC,H,W,device='cuda')
sol._mfm_conv_kernel[(1,)](x,w,b,o,N,IC,OC,H,W,False,BLOCK=256,num_warps=4)
print('direct sol kernel err', (mx-o).abs().max().item())

# now via _run_mfm
o2=torch.empty(N,OC,H,W,device='cuda')
sol._run_mfm(ref.conv1.filter,x,o2,False)
print('via _run_mfm err', (mx-o2).abs().max().item())
print('o vs o2 diff', (o-o2).abs().max().item())
