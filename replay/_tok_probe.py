import json, statistics as st
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/mnt/vast/models/dsv4/tokenizer.json")

def ntok(s): return len(tok.encode(s).ids)

sess=[json.loads(l) for l in open("batch_long.replay.jsonl")]
# per-message token counts, cumulative -> ISL per turn = sum of tokens of messages[:prefix_len]
isl=[]
# small per-message role overhead approximation
OVH=4
for s in sess:
    msgs=s["messages"]
    cum=[]; run=0
    for m in msgs:
        run += ntok(m["content"])+OVH
        cum.append(run)
    for t in s["turns"]:
        isl.append(cum[t["prefix_len"]-1])
print("batch_long turns",len(isl))
print("dsv4-tokenizer ISL: median %d  mean %d  p95 %d  max %d"%(
    int(st.median(isl)), int(st.mean(isl)), int(sorted(isl)[int(len(isl)*0.95)]), max(isl)))
# also measure current system prompt size
sp=open("/mnt/home/ppbhatt500/gpumode-triton/system_prompt.md").read()
print("current system_prompt.md tokens:", ntok(sp))
