from __future__ import annotations
import os, tarfile, urllib.request, json
from pathlib import Path
import numpy as np
import pandas as pd

TAG=os.getenv('QLIB_RELEASE_TAG','2026-07-29')
ROOT=Path('qlib_data_20y')
OUT=Path('results_20y_probe'); OUT.mkdir(exist_ok=True)

def download_extract():
    if (ROOT/'calendars/day.txt').exists(): return
    ROOT.mkdir(exist_ok=True)
    url=f'https://github.com/chenditc/investment_data/releases/download/{TAG}/qlib_bin.tar.gz'
    arc=Path('qlib_bin_20y.tar.gz')
    print('DOWNLOAD',url,flush=True)
    urllib.request.urlretrieve(url,arc)
    print('ARCHIVE_BYTES',arc.stat().st_size,flush=True)
    with tarfile.open(arc,'r:gz') as tf:
        members=tf.getmembers(); tops={Path(m.name).parts[0] for m in members if Path(m.name).parts}
        if len(tops)==1:
            for m in members:
                parts=Path(m.name).parts
                if len(parts)<=1: continue
                m.name=str(Path(*parts[1:])); tf.extract(m,ROOT)
        else: tf.extractall(ROOT)
    arc.unlink(missing_ok=True)
    if not (ROOT/'calendars/day.txt').exists():
        hits=list(ROOT.rglob('calendars/day.txt'))
        if len(hits)!=1: raise RuntimeError(f'calendar hits={hits}')
        b=hits[0].parent.parent
        for child in b.iterdir():
            tgt=ROOT/child.name
            if not tgt.exists(): child.rename(tgt)

def read_bin(folder:Path,field:str,cal:pd.DatetimeIndex):
    p=folder/f'{field}.day.bin'
    if not p.exists(): return None
    a=np.fromfile(p,dtype='<f4')
    if len(a)<2: return None
    st=int(a[0]); n=min(len(a)-1,len(cal)-st)
    return st,a[1:1+n]

def main():
    download_extract()
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ROOT/'calendars/day.txt',header=None)[0]))
    feats=ROOT/'features'
    dirs=sorted([p for p in feats.iterdir() if p.is_dir()])
    ash=[p for p in dirs if p.name.startswith(('sh6','sz0','sz3','bj'))]
    print('CALENDAR',cal.min(),cal.max(),'N',len(cal),flush=True)
    print('FEATURE_DIRS',len(dirs),'A_SHARE_PREFIX_DIRS',len(ash),flush=True)
    sample=[]
    for p in ash[:20]+ash[-20:]:
        sample.append({'code':p.name,'fields':';'.join(sorted(x.name.replace('.day.bin','') for x in p.glob('*.day.bin')))})
    pd.DataFrame(sample).to_csv(OUT/'sample_fields.csv',index=False)
    # field frequency across every stock directory
    freq={}
    for p in ash:
        for x in p.glob('*.day.bin'):
            f=x.name.replace('.day.bin',''); freq[f]=freq.get(f,0)+1
    pd.DataFrame(sorted(freq.items(),key=lambda x:(-x[1],x[0])),columns=['field','stock_dirs']).to_csv(OUT/'field_frequency.csv',index=False)
    print('FIELD_FREQ_TOP',sorted(freq.items(),key=lambda x:-x[1])[:30],flush=True)
    dates=['2006-01-04','2008-01-02','2010-01-04','2014-01-02','2016-07-29','2020-01-02','2026-07-29']
    cov=[]
    for ds in dates:
        d=pd.Timestamp(ds)
        if d not in cal: d=cal[cal.get_indexer([d],method='bfill')[0]]
        ix=cal.get_loc(d); active=0; close_ok=0; factor_ok=0; amount_ok=0
        for p in ash:
            rb=read_bin(p,'close',cal)
            if rb is None: continue
            st,v=rb
            j=ix-st
            if 0<=j<len(v) and np.isfinite(v[j]) and v[j]>0:
                active+=1; close_ok+=1
                rf=read_bin(p,'factor',cal)
                if rf is not None:
                    s2,v2=rf; q=ix-s2
                    if 0<=q<len(v2) and np.isfinite(v2[q]) and v2[q]>0: factor_ok+=1
                ra=read_bin(p,'amount',cal)
                if ra is not None:
                    s3,v3=ra; q=ix-s3
                    if 0<=q<len(v3) and np.isfinite(v3[q]) and v3[q]>=0: amount_ok+=1
        cov.append({'requested_date':ds,'trade_date':d.date(),'active_close':active,'factor_ok':factor_ok,'amount_ok':amount_ok})
        print('COVERAGE',cov[-1],flush=True)
    pd.DataFrame(cov).to_csv(OUT/'date_coverage.csv',index=False)
    inst=[]
    if (ROOT/'instruments').exists():
        for p in sorted((ROOT/'instruments').glob('*.txt')):
            lines=sum(1 for _ in p.open(errors='ignore'))
            inst.append({'file':p.name,'lines':lines})
    pd.DataFrame(inst).to_csv(OUT/'instrument_files.csv',index=False)
    print('INSTRUMENTS',inst,flush=True)
    # survivors/delisted proxy: histories ending well before release
    hist=[]
    for p in ash:
        rb=read_bin(p,'close',cal)
        if rb is None: continue
        st,v=rb; good=np.where(np.isfinite(v)&(v>0))[0]
        if not len(good): continue
        hist.append({'code':p.name.upper(),'first_date':cal[st+good[0]].date(),'last_date':cal[st+good[-1]].date(),'obs':len(good)})
    h=pd.DataFrame(hist)
    h.to_csv(OUT/'history_ranges.csv',index=False)
    print('HISTORIES',len(h),'END_BEFORE_2016',int((pd.to_datetime(h.last_date)<'2016-01-01').sum()),'END_BEFORE_2020',int((pd.to_datetime(h.last_date)<'2020-01-01').sum()),flush=True)
    summary={'tag':TAG,'calendar_start':str(cal.min().date()),'calendar_end':str(cal.max().date()),'calendar_n':len(cal),'feature_dirs':len(dirs),'ashare_prefix_dirs':len(ash),'field_frequency':freq,'history_count':len(h)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
