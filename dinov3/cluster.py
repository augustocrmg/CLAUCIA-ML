import pandas as pd, numpy as np, collections, os, re
S=os.path.dirname(os.path.abspath(__file__))+'/'
df=pd.read_parquet(S+'index_all.parquet').reset_index(drop=True)
B=np.array([np.frombuffer(bytes.fromhex(h),dtype=np.uint8) for h in df.dhcanon])
n=len(df); par=np.arange(n)
def f(i):
    while par[i]!=i: par[i]=par[par[i]]; i=par[i]
    return i
def u(i,j):
    a,b=f(i),f(j)
    if a!=b: par[a]=b
TH=6
chunks=B.reshape(n,8,4)
cand=set()
for c in range(8):
    keys=collections.defaultdict(list)
    for i,k in enumerate(map(bytes,chunks[:,c,:])): keys[k].append(i)
    for idx in keys.values():
        if 1<len(idx)<400:
            idx=np.array(idx); X=np.unpackbits(np.bitwise_xor(B[idx][:,None],B[idx][None]),axis=2).sum(2)
            for a,b in np.argwhere(np.triu(X<=TH,1)): u(idx[a],idx[b])
# md5 too
for idx in df.groupby('md5').indices.values():
    for j in idx[1:]: u(idx[0],j)
df['cluster']=[f(i) for i in range(n)]
# roboflow stem grouping (augmented exports): name like X_jpg.rf.HASH.jpg
def stem(p):
    b=os.path.basename(p); m=re.match(r'(.+?)_(jpg|jpeg|png)\.rf\.',b,re.I); return m.group(1) if m else None
df['rf_stem']=df.path.map(stem)
for s,g in df[df.rf_stem.notna()].groupby(['source','rf_stem']):
    idx=g.index.values
    for j in idx[1:]: u(idx[0],j)
df['cluster']=[f(i) for i in range(n)]
df.to_parquet(S+'index_clustered.parquet')
print('clusters totais:',df.cluster.nunique())
for s,g in df.groupby('source'): print(f'{s:15s} arquivos={len(g):6d} clusters={g.cluster.nunique():6d}')
