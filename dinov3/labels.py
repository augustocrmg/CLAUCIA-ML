import pandas as pd, re, collections, os
S=os.path.dirname(os.path.abspath(__file__))+'/'
df=pd.read_parquet(S+'index_clustered.parquet')
def lab(r):
    f=r.folder.replace('\\','/'); s=r.source; last=f.split('/')[-1]
    m=re.search(r'(grade|grau|stage)\s*0?(\d)',f,re.I)
    if s in('v4','kg_wagner','archive'):
        if 'normal' in f.lower(): return 'Normal_patch'
        if 'False-case' in f: return 'other_wound'
        return f'W{m.group(2)}' if m else last
    if s in('kg_adpm','roboflow_local','kg_stages') or (s=='kg_pran' and 'Severity' in f): return f'W{m.group(2)}' if m else last
    if s=='mendeley': return 'Normal_foot' if 'Nomal' in f else 'wound_ungraded'
    if s=='kg_laith': return 'Normal_patch' if 'Normal' in f else ('ulcer_patch' if 'Abnormal' in f else 'dfu_misc')
    if re.search(r'healthy|normal',f,re.I): return 'healthy'
    if re.search(r'ordinary|wound$',f,re.I): return 'other_wound'
    return 'dfu_ungraded'
df['label']=df.apply(lab,axis=1)
df.to_parquet(S+'index_labeled.parquet')
print(df.groupby(['source','label']).cluster.nunique().to_string())
# v4 cluster provenance
v4c=set(df[df.source=='v4'].cluster)
print('\n== De onde vêm os clusters do v4:')
for s,g in df[df.cluster.isin(v4c)&(df.source!='v4')].groupby('source'): print(f'  {s:15s} {g.cluster.nunique()}')
# label agreement v4 vs adpm/archive
v4lab=df[df.source=='v4'].groupby('cluster').label.agg(lambda x:'/'.join(sorted(set(x))))
for s in ['kg_adpm','archive','kg_stages','kg_pran','kg_wagner','kg_laith']:
    o=df[(df.source==s)&df.cluster.isin(v4c)].groupby('cluster').label.agg(lambda x:'/'.join(sorted(set(x))))
    ct=pd.crosstab(v4lab.reindex(o.index),o)
    print(f'\n== v4 (linhas) x {s} (colunas), em clusters compartilhados'); print(ct.to_string())
# ADPM vs stages vs pran relation
a=set(df[df.source=='kg_adpm'].cluster)
for s in ['kg_stages','kg_pran','roboflow_local','archive','kg_dfu3','kg_laith','mendeley']:
    c=set(df[df.source==s].cluster); print(f'{s}: clusters={len(c)} em comum com ADPM={len(c&a)} com v4={len(c&v4c)}')
