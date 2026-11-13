import os, re, hashlib, numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
R=os.path.expanduser('~/Repos/CLAUCIA-ML/datasets/')
SRC={  # nome -> pasta relativa
 'v4':'dataset_wagner_v4','archive':'Archive','roboflow_local':'dataset_wagner_v3',
 'kg_wagner':'externos/purushomohan_dfu-wagners-classification',
 'kg_adpm':'externos/khalidsiddiqui2003_dfu-dataset-annotated-into-4-classes',
 'kg_stages':'externos/srivatsanmk2004_dfu-stages-dataset',
 'kg_dfu3':'externos/srivatsanmk2004_dfu-dataset',
 'kg_laith':'externos/laithjj_diabetic-foot-ulcer-dfu',
 'kg_pran':'externos/pran84_diabetic-footulcer-dataset',
 'kg_footdis':'externos/mohamedmahmoud153_foot-diseases',
 'kg_dedup':'externos/tanmayy29_dfu-cleaned-dataset-deduplicated',
 'kg_footulcer':'externos/adrijadas267_footulcer',
 'kg_avanish':'externos/avanishparab_diabetic-foot-ulcers',
 'mendeley':'externos/mendeley_lower_limb_feet',
}
SKIP=re.compile(r'(wound_mask|/labels|Thermal)',re.I)
def feat(p):
    try:
        im=Image.open(p); im.load(); w,h=im.size; g=im.convert('L')
        hs=[]
        for t in [None,Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM,Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270,Image.TRANSPOSE,Image.TRANSVERSE]:
            x=g if t is None else g.transpose(t)
            a=np.asarray(x.resize((17,16),Image.BILINEAR),dtype=np.int16)
            hs.append(np.packbits((a[:,1:]>a[:,:-1]).flatten()).tobytes().hex())
        md5=hashlib.md5(open(p,'rb').read()).hexdigest()
        return dict(path=p,w=w,h=h,md5=md5,dh0=hs[0],dhcanon=min(hs))
    except Exception as e: return dict(path=p,err=str(e))
if __name__=='__main__':
    rows=[]
    for s,d in SRC.items():
        for root,_,fs in os.walk(R+d):
            for f in fs:
                p=os.path.join(root,f)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')) and not SKIP.search(p):
                    rows.append((s,p,os.path.relpath(root,R+d)))
    paths=[r[1] for r in rows]
    with ProcessPoolExecutor(18) as ex: F=list(ex.map(feat,paths,chunksize=64))
    df=pd.DataFrame(F); df['source']=[r[0] for r in rows]; df['folder']=[r[2] for r in rows]
    out=os.path.dirname(os.path.abspath(__file__))+'/index_all.parquet'
    df.to_parquet(out); print(len(df), df.err.notna().sum() if 'err' in df else 0); print(df.groupby('source').size())
