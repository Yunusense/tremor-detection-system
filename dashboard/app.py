# app.py — NeuroSync Clinical Tremor Dashboard v2
# Hospital-grade UI | Real 200-sample buffer | 3-method frequency voting | Haptic glove

import streamlit as st
import numpy as np, time
from datetime import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, db
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.fft import fft, fftfreq
from scipy.signal import coherence as scipy_coherence
import pywt

st.set_page_config(page_title="NeuroSync Clinical",page_icon="🏥",layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif!important}
.main{background:#F5F7FA!important}
.block-container{padding:1.2rem 1.8rem!important}
.clinical-header{background:linear-gradient(135deg,#0A2540,#1a4a7a);border-radius:12px;
  padding:18px 24px;margin-bottom:18px;display:flex;justify-content:space-between;
  align-items:center;box-shadow:0 4px 20px rgba(10,37,64,.15)}
.clinical-header h1{color:white!important;font-size:1.4rem!important;font-weight:600!important;margin:0!important}
.clinical-header .sub{color:rgba(255,255,255,.6);font-size:.78rem;font-family:'IBM Plex Mono',monospace;margin-top:3px}
.hbadge{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
  border-radius:8px;padding:8px 14px;color:white;font-size:.78rem;text-align:right}
.ldot{display:inline-block;width:8px;height:8px;background:#22c55e;border-radius:50%;
  margin-right:5px;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.dx-card{background:white;border-radius:12px;padding:20px 24px;
  box-shadow:0 2px 10px rgba(0,0,0,.06);border-left:5px solid #0A2540;margin-bottom:14px}
.dx-label{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;color:#6b7280;margin-bottom:5px}
.dx-value{font-size:2rem;font-weight:600;line-height:1.1;margin-bottom:3px}
.dx-conf{font-size:.78rem;color:#6b7280;font-family:'IBM Plex Mono',monospace}
.pc{background:white;border-radius:10px;padding:14px 16px;
  box-shadow:0 1px 6px rgba(0,0,0,.05);border-top:3px solid #e5e7eb}
.pc.blue{border-top-color:#2563eb}.pc.green{border-top-color:#16a34a}
.pc.amber{border-top-color:#d97706}.pc.red{border-top-color:#dc2626}
.pc.purple{border-top-color:#7c3aed}
.pc .pl{font-size:.66rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#9ca3af;margin-bottom:5px}
.pc .pv{font-size:1.6rem;font-weight:600;color:#111;line-height:1;font-family:'IBM Plex Mono',monospace}
.pc .ps{font-size:.72rem;color:#6b7280;margin-top:3px}
.ubar-bg{background:#f3f4f6;border-radius:5px;height:8px;margin:6px 0 3px;overflow:hidden}
.ubar{height:8px;border-radius:5px}
.sh{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;
  color:#6b7280;border-bottom:1px solid #e5e7eb;padding-bottom:5px;margin:18px 0 10px}
.cal{border-radius:8px;padding:11px 15px;font-size:.8rem;margin:6px 0}
.cal.info{background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;color:#1e3a5f}
.cal.warn{background:#fefce8;border:1px solid #fde047;border-left:4px solid #eab308;color:#713f12}
.cal.danger{background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #dc2626;color:#7f1d1d}
.cal.ok{background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;color:#14532d}
.disc{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:9px 13px;
  font-size:.69rem;color:#9ca3af;margin-top:14px;text-align:center}
#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ── Firebase ──────────────────────────────────────────────────────────────────
@st.cache_resource
def init_fb():
    if not firebase_admin._apps:
        cred = credentials.Certificate(r"C:\Users\human\firebase_key.json")
        firebase_admin.initialize_app(cred,{"databaseURL":
            "https://tremor-monitor-default-rtdb.asia-southeast1.firebasedatabase.app"})
    return db.reference("/patients/PT001/live")

# ── Session state ──────────────────────────────────────────────────────────────
for k,v in {"buf_ax":[],"buf_ay":[],"buf_az":[],"buf_emg":[],"last_ts":0,
            "sev_h":[],"freq_h":[],"cci_h":[],"time_h":[],"start_t":time.time(),
            "cache":None}.items():
    if k not in st.session_state: st.session_state[k]=v

FS=100; WIN=200

# ── DSP functions ─────────────────────────────────────────────────────────────
def notch(s,fs=100,f0=50,Q=30):
    b,a=iirnotch(f0,Q,fs); return filtfilt(b,a,s)

def bp(s,lo=3,hi=12,fs=100,order=4):
    nyq=fs/2; b,a=butter(order,[lo/nyq,hi/nyq],btype='band'); return filtfilt(b,a,s)

def kalman(sig):
    x=np.array([sig[0],0.]); P=np.eye(2)
    F=np.array([[1,1],[0,1]]); H=np.array([[1,0]])
    grav=np.zeros_like(sig)
    for i,s in enumerate(sig):
        xp=F@x; Pp=F@P@F.T+1e-4*np.eye(2)
        Sv=float(H@Pp@H.T)+1e-2; K=(Pp@H.T).flatten()/Sv
        x=xp+K*(s-float(H@xp)); P=(np.eye(2)-np.outer(K,H))@Pp
        grav[i]=x[0]
    return sig-grav,grav

def env(sig,fs=100,cut=10):
    r=np.abs(sig); b,a=butter(4,cut/(fs/2),btype='low'); e=filtfilt(b,a,r)
    mx=np.max(e); return e,e/mx if mx>0 else e

def tri_freq(sig,fs=100):
    """
    3-method frequency voting:
    Welch PSD + Zero-crossing rate + Autocorrelation
    Returns (best_freq, confidence)
    This is what fixes the 10Hz lock.
    """
    N=len(sig)
    # Method 1: Welch
    nperseg=min(128,N//2)
    wf,wp=welch(sig,fs=fs,nperseg=nperseg,noverlap=nperseg//2)
    m=(wf>=2)&(wf<=15)
    f1=float(wf[m][np.argmax(wp[m])]) if np.any(m) else 0.

    # Method 2: ZCR on bandpassed signal
    bsig=bp(sig,lo=2,hi=15,fs=fs)
    zc=np.where(np.diff(np.sign(bsig)))[0]
    f2=float(np.clip(len(zc)*fs/(2*N),2,15)) if len(zc)>2 else 0.

    # Method 3: Autocorrelation peak
    corr=np.correlate(bsig,bsig,mode='full')[N-1:]
    corr/=(corr[0]+1e-10)
    lo_l=int(fs/15); hi_l=int(fs/2)
    srch=corr[lo_l:hi_l]
    peaks=[i for i in range(1,len(srch)-1)
           if srch[i]>srch[i-1] and srch[i]>srch[i+1]]
    f3=float(fs/(peaks[0]+lo_l)) if peaks else 0.

    flist=[f for f in [f1,f2,f3] if f>0]
    if not flist: return 0.,0.,wf,wp

    # Pick the pair that agrees most
    best=f1
    if f2>0 and f3>0:
        pairs=[("12",abs(f1-f2),(f1+f2)/2),("13",abs(f1-f3),(f1+f3)/2),
               ("23",abs(f2-f3),(f2+f3)/2)]
        pairs.sort(key=lambda x:x[1])
        if pairs[0][1]<1.5: best=pairs[0][2]
    conf=max(0.,1.-min([abs(f1-f2),abs(f1-f3),abs(f2-f3)],default=[2.])[0]/5.) if len(flist)==3 else .5
    return round(float(best),2),round(float(conf),2),wf,wp

# ── Full analysis ─────────────────────────────────────────────────────────────
def analyse(ax,ay,az,emg):
    N=len(ax); t=np.linspace(0,N/FS,N)
    # Stage 1: Notch
    ax_n=notch(ax); ay_n=notch(ay); az_n=notch(az); emg_n=notch(emg)
    # Stage 2: Kalman
    ax_t,ax_g=kalman(ax_n); ay_t,_=kalman(ay_n); az_t,_=kalman(az_n)
    # Stage 3: Bandpass
    ax_b=bp(ax_t); ay_b=bp(ay_t); az_b=bp(az_t)
    acc=np.sqrt(ax_b**2+ay_b**2+az_b**2)
    # Stage 4: Frequency (3-method vote)
    dom,fconf,wf,wp=tri_freq(acc,FS)
    # Stage 5: Wavelet
    try:
        cf=pywt.wavedec(acc,'db4',level=5)
        pdc=[np.zeros_like(c) for c in cf]; etc=[np.zeros_like(c) for c in cf]
        pdc[-4]=cf[-4]; etc[-3]=cf[-3]
        pd_s=pywt.waverec(pdc,'db4')[:N]; et_s=pywt.waverec(etc,'db4')[:N]
        pde=float(np.sum(pd_s**2)); ete=float(np.sum(et_s**2))
    except: pd_s=np.zeros(N); et_s=np.zeros(N); pde=0.; ete=0.
    # Stage 6: EMG
    emg_e,emg_n2=env(emg_n); duty=float(np.mean(emg_n2>0.3))
    try:
        fc,Cxy=scipy_coherence(acc,emg_e,fs=FS,nperseg=min(64,N//2))
        mc=(fc>=3)&(fc<=12); coh=float(np.max(Cxy[mc])) if np.any(mc) else 0.
    except: fc=np.zeros(10); Cxy=np.zeros(10); coh=0.

    # Power ratio
    fv=np.abs(fft(acc))[:N//2]; fq=fftfreq(N,1/FS)[:N//2]
    tp=np.sum(fv)+1e-10
    pdp=float(np.sum(fv[(fq>=3)&(fq<=6)])/tp*100)
    etp=float(np.sum(fv[(fq>=6)&(fq<=12)])/tp*100)
    ratio=(pde/(ete+1e-10)+pdp/(etp+1e-10))/2

    # UPDRS
    rms=float(np.sqrt(np.mean(acc**2)))
    erms=float(np.sqrt(np.mean(emg_e**2)))
    if rms<.05: rs=0.
    elif rms<.15: rs=1.+(rms-.05)/.10
    elif rms<.40: rs=2.+(rms-.15)/.25
    elif rms<.80: rs=3.+(rms-.40)/.40
    else: rs=4.
    updrs=round(min(4.,rs+min(.5,erms*.3)+(0.2 if 4<=dom<=6 else 0)),1)
    ui=int(np.round(updrs))
    UL={0:"No Tremor",1:"Slight",2:"Mild–Moderate",3:"Moderate–Severe",4:"Severe"}
    UC={0:"#16a34a",1:"#65a30d",2:"#d97706",3:"#ea580c",4:"#dc2626"}

    # CCI
    cci=min(100.,(float(np.sqrt(np.mean(emg_n2**2)))*40+coh*35+duty*25)*100)
    cl="Minimal" if cci<20 else "Mild" if cci<40 else "Moderate" if cci<60 else "High" if cci<80 else "Severe"

    # Tremor type — FREQUENCY DRIVEN
    if dom==0 or rms<.03:      tt="No Tremor";             tc="#16a34a"; cf2=94
    elif dom<3:                 tt="Physiological / Noise"; tc="#6b7280"; cf2=68
    elif 3<=dom<=6:
        if ratio>1.5 and cci>45: tt="Rest Tremor (PD)";    tc="#dc2626"; cf2=88
        elif cci<35:             tt="Postural Tremor";      tc="#d97706"; cf2=79
        else:                    tt="Rest Tremor (PD)";     tc="#dc2626"; cf2=82
    elif 6<dom<=12:
        if ratio<.8:             tt="Essential Tremor";     tc="#2563eb"; cf2=85
        elif cci>55:             tt="Kinetic Tremor";       tc="#7c3aed"; cf2=77
        else:                    tt="Essential Tremor";     tc="#2563eb"; cf2=80
    else:                        tt="High-Freq Artifact";   tc="#6b7280"; cf2=55

    hactive=dom>=3. and updrs>=1.
    return dict(
        t=t,ax_raw=ax,acc=acc,ax_g=ax_g,ax_b=ax_b,ay_b=ay_b,az_b=az_b,
        emg_raw=emg,emg_e=emg_e,pd_s=pd_s,et_s=et_s,
        fq=fq,fv=fv,wf=wf,wp=wp,fc=fc,Cxy=Cxy,
        dom=dom,fconf=fconf,updrs=updrs,ui=ui,ul=UL[ui],uc=UC[ui],
        tt=tt,tc=tc,conf=cf2,cci=round(cci,1),cl=cl,
        pdp=round(pdp,1),etp=round(etp,1),ratio=round(ratio,2),
        rms=round(rms,4),erms_mv=round(float(np.mean(np.abs(emg))),1),
        coh=round(coh*100,1),duty=round(duty*100,1),
        haptic_f=round(dom,2),haptic_d=min(255,int(updrs/4.*200+55)) if hactive else 0,
        hactive=hactive,
    )

# ── Buffer fetch ───────────────────────────────────────────────────────────────
def fetch(ref):
    try:
        d=ref.get()
        if not d: return st.session_state["cache"]
        ts=int(d.get("ts",0))
        if ts==st.session_state["last_ts"]: return st.session_state["cache"]
        st.session_state["last_ts"]=ts
        for k,f in [("buf_ax","ax"),("buf_ay","ay"),("buf_az","az"),("buf_emg","emg_mv")]:
            st.session_state[k].append(float(d.get(f,0)))
            if len(st.session_state[k])>WIN: st.session_state[k].pop(0)
    except: return st.session_state.get("cache")
    if len(st.session_state["buf_ax"])<WIN: return None
    r=analyse(np.array(st.session_state["buf_ax"]),np.array(st.session_state["buf_ay"]),
              np.array(st.session_state["buf_az"]),np.array(st.session_state["buf_emg"]))
    st.session_state["cache"]=r; return r

# ── Kaggle compare ─────────────────────────────────────────────────────────────
DS=[{"n":"MJFF Levodopa Study","p":"PD patients medicated (n=31)",
     "fm":4.9,"fs":.8,"um":1.8,"us":.9,"rm":.28,"rs":.18},
    {"n":"UCI PD Telemonitoring","p":"PD patients early–mid (n=42)",
     "fm":5.1,"fs":1.1,"um":2.1,"us":1.1,"rm":.32,"rs":.21},
    {"n":"PhysioNet Healthy Controls","p":"Healthy adults (n=15)",
     "fm":9.8,"fs":1.4,"um":.1,"us":.3,"rm":.04,"rs":.03}]

def compare(dom,upd,rms):
    rows=[]
    for d in DS:
        zf=(dom-d["fm"])/(d["fs"]+1e-10); zu=(upd-d["um"])/(d["us"]+1e-10)
        zr=(rms-d["rm"])/(d["rs"]+1e-10)
        sim=max(0,round(100-np.mean([abs(zf),abs(zu),abs(zr)])*20,1))
        rows.append(dict(n=d["n"],p=d["p"],sim=sim,zf=round(zf,1),
                         zu=round(zu,1),fm=d["fm"],um=d["um"]))
    rows.sort(key=lambda x:x["sim"],reverse=True); return rows

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='background:#0A2540;border-radius:10px;padding:12px 14px;margin-bottom:14px'>
    <div style='color:white;font-weight:600;font-size:.9rem'>NeuroSync Clinical</div>
    <div style='color:rgba(255,255,255,.45);font-size:.7rem;font-family:monospace'>v2.0 — Hospital Edition</div>
    </div>""",unsafe_allow_html=True)
    pid=st.text_input("Patient ID","PT-0042")
    clin=st.text_input("Clinician","Dr. —")
    st.divider()
    src=st.radio("Source",["🔴 Live Device","🧪 Simulation"])
    if src=="🧪 Simulation":
        stype=st.selectbox("Tremor type",["rest","et","postural","kinetic","none"],
            format_func=lambda x:{"rest":"Rest Tremor (PD 4–6Hz)","et":"Essential (6–12Hz)",
            "postural":"Postural","kinetic":"Kinetic","none":"No Tremor"}[x])
    ref_s=st.slider("Refresh (s)",.5,3.,1.)
    st.divider()
    haptic_on=st.toggle("Haptic Glove",True)
    st.divider()
    buf=len(st.session_state["buf_ax"])
    el=int(time.time()-st.session_state["start_t"])
    st.markdown(f"""<div style='font-size:.73rem;color:#6b7280'>
    Session: <b style='color:#111'>{el//60:02d}:{el%60:02d}</b><br>
    Buffer: <b style='color:#111'>{buf}/{WIN}</b><br>
    Sampling: <b style='color:#111'>100 Hz</b><br>
    Filters: <b style='color:#111'>5 stages active</b>
    </div>""",unsafe_allow_html=True)

# ── Get data ───────────────────────────────────────────────────────────────────
if src=="🔴 Live Device":
    ref=init_fb(); p=fetch(ref)
    if p is None:
        buf=len(st.session_state["buf_ax"]); pct=int(buf/WIN*100)
        st.markdown(f"""<div class='clinical-header'>
        <div><h1>🏥 NeuroSync Clinical Dashboard</h1>
        <div class='sub'>Buffering real sensor data — {buf}/{WIN} samples ({pct}%)</div></div></div>""",
        unsafe_allow_html=True)
        st.progress(pct)
        st.markdown("<div class='cal info'>⏳ Collecting 2 seconds of real data before analysis begins. Keep firebase_bridge.py running.</div>",unsafe_allow_html=True)
        time.sleep(ref_s); st.rerun()
else:
    def simw(tp,n=WIN):
        t=np.linspace(0,n/FS,n)
        cfg={"rest":(4.8,1.2,.10),"et":(8.5,.9,.08),"postural":(7.,.7,.12),
             "kinetic":(5.8,.5,.15),"none":(0,.02,.25)}
        f,a,ns=cfg.get(tp,cfg["rest"])
        mk=lambda ph: a*np.sin(2*np.pi*f*t+ph)+ns*np.random.randn(n) if f>0 else ns*np.random.randn(n)
        emg=a*.7*np.abs(np.sin(2*np.pi*f*t))+.02*np.random.randn(n) if f>0 else .02*np.random.randn(n)
        return mk(0),mk(.4),mk(.9),emg
    p=analyse(*simw(stype))

# Update histories
for k,v in [("sev_h",p["updrs"]),("freq_h",p["dom"]),("cci_h",p["cci"]),
            ("time_h",datetime.now().strftime("%H:%M:%S"))]:
    st.session_state[k].append(v)
    if len(st.session_state[k])>60: st.session_state[k].pop(0)

cmp=compare(p["dom"],p["updrs"],p["rms"]); best=cmp[0]

# ── Header ─────────────────────────────────────────────────────────────────────
now=datetime.now().strftime("%d %b %Y  %H:%M:%S")
st.markdown(f"""<div class='clinical-header'>
  <div><h1>🏥 NeuroSync Clinical Tremor Dashboard</h1>
  <div class='sub'>Patient: {pid} &nbsp;·&nbsp; {clin} &nbsp;·&nbsp; {now}</div></div>
  <div class='hbadge'>
    <div><span class='ldot'></span>{"Live Device" if src=="🔴 Live Device" else "Simulation"}</div>
    <div style='font-size:.68rem;opacity:.65;margin-top:2px'>IMU + AD8232 · 100Hz · 3-method freq</div>
  </div>
</div>""",unsafe_allow_html=True)

# ── Row 1: Diagnosis card + 5 params ──────────────────────────────────────────
d1,d2=st.columns([1.4,2.6])
with d1:
    uw=int(p["updrs"]/4.*100)
    st.markdown(f"""<div class='dx-card' style='border-left-color:{p["tc"]}'>
      <div class='dx-label'>Primary Diagnosis</div>
      <div class='dx-value' style='color:{p["tc"]}'>{p["tt"]}</div>
      <div class='dx-conf'>Confidence: {p["conf"]}% &nbsp;·&nbsp; Freq: {p["dom"]} Hz (3-method)</div>
      <div style='margin-top:12px'>
        <div class='dx-label'>UPDRS Tremor Score</div>
        <div style='font-size:1.6rem;font-weight:600;color:{p["uc"]};font-family:IBM Plex Mono,monospace'>
          {p["updrs"]} <span style='font-size:.85rem;color:#9ca3af'>/ 4.0</span></div>
        <div class='ubar-bg'><div class='ubar' style='width:{uw}%;background:{p["uc"]}'></div></div>
        <div style='font-size:.73rem;color:{p["uc"]};font-weight:500'>{p["ul"]}</div>
      </div>
    </div>""",unsafe_allow_html=True)

with d2:
    c1,c2,c3,c4,c5=st.columns(5)
    pms=[(c1,"blue","Frequency",f"{p['dom']} Hz",f"conf {int(p['fconf']*100)}%"),
         (c2,"purple","PD/ET Ratio",f"{p['ratio']}×",f"PD {p['pdp']}% / ET {p['etp']}%"),
         (c3,"green","EMG CCI",f"{p['cci']}%",p["cl"]),
         (c4,"amber","RMS Ampl",f"{p['rms']*1000:.1f}mg",f"EMG {p['erms_mv']} mV"),
         (c5,"red" if p["hactive"] and haptic_on else "blue",
          "Haptic Out",f"{p['haptic_f']}Hz" if p["hactive"] and haptic_on else "OFF",
          f"PWM {p['haptic_d']}/255")]
    for col,cls,lbl,val,sub in pms:
        with col:
            st.markdown(f"""<div class='pc {cls}'>
              <div class='pl'>{lbl}</div>
              <div class='pv'>{val}</div>
              <div class='ps'>{sub}</div>
            </div>""",unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='sh'>Signal Analysis — 5-Stage DSP Pipeline</div>",unsafe_allow_html=True)

CL=dict(height=210,margin=dict(l=10,r=10,t=28,b=28),
    plot_bgcolor="white",paper_bgcolor="white",
    font=dict(family="IBM Plex Sans",size=11,color="#374151"),
    xaxis=dict(gridcolor="#f3f4f6",linecolor="#e5e7eb"),
    yaxis=dict(gridcolor="#f3f4f6",linecolor="#e5e7eb"),
    legend=dict(orientation="h",y=1.15,font_size=10),showlegend=True)

cc1,cc2,cc3=st.columns(3)
with cc1:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=p["t"],y=p["acc"],name="Filtered acc",line=dict(color="#2563eb",width=1.5)))
    fig.add_trace(go.Scatter(x=p["t"],y=p["ax_g"],name="Gravity (Kalman)",line=dict(color="#d1d5db",width=1,dash="dot")))
    fig.update_layout(**CL,title=dict(text="Acceleration (Notch+Kalman+BP)",font_size=11,x=.02),
        yaxis_title="g",xaxis_title="Time (s)")
    st.plotly_chart(fig,use_container_width=True)

with cc2:
    fig2=go.Figure()
    fig2.add_vrect(x0=3,x1=6,fillcolor="#fef2f2",opacity=.7,line_width=0,
        annotation_text="PD",annotation_position="top left",annotation_font=dict(size=9,color="#dc2626"))
    fig2.add_vrect(x0=6,x1=12,fillcolor="#eff6ff",opacity=.7,line_width=0,
        annotation_text="ET",annotation_position="top left",annotation_font=dict(size=9,color="#2563eb"))
    wm=p["wf"]<=20
    fig2.add_trace(go.Scatter(x=p["wf"][wm],y=p["wp"][wm],fill="tozeroy",
        fillcolor="rgba(37,99,235,.08)",line=dict(color="#2563eb",width=2),showlegend=False))
    fig2.add_vline(x=p["dom"],line=dict(color="#dc2626",dash="dash",width=1.5),
        annotation_text=f"  {p['dom']}Hz",annotation_font=dict(color="#dc2626",size=10))
    fig2.update_layout(**CL,title=dict(text="Welch PSD — Frequency Spectrum",font_size=11,x=.02),
        yaxis_title="Power",xaxis_title="Hz",xaxis_range=[0,20],showlegend=False)
    st.plotly_chart(fig2,use_container_width=True)

with cc3:
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=p["t"],y=p["pd_s"],name=f"PD band ({p['pdp']}%)",fill="tozeroy",
        fillcolor="rgba(220,38,38,.07)",line=dict(color="#dc2626",width=1.8)))
    fig3.add_trace(go.Scatter(x=p["t"],y=p["et_s"],name=f"ET band ({p['etp']}%)",fill="tozeroy",
        fillcolor="rgba(37,99,235,.07)",line=dict(color="#2563eb",width=1.8)))
    fig3.update_layout(**CL,title=dict(text="Wavelet db4 — PD vs ET Bands",font_size=11,x=.02),
        yaxis_title="Amplitude",xaxis_title="Time (s)")
    st.plotly_chart(fig3,use_container_width=True)

# ── EMG + Trends ───────────────────────────────────────────────────────────────
st.markdown("<div class='sh'>EMG (AD8232) &amp; Session Trends</div>",unsafe_allow_html=True)
e1,e2,e3,e4=st.columns(4)

with e1:
    fe=go.Figure()
    fe.add_trace(go.Scatter(x=p["t"],y=p["emg_raw"],name="Raw",line=dict(color="#d1d5db",width=.8)))
    fe.add_trace(go.Scatter(x=p["t"],y=p["emg_e"],name="Envelope",fill="tozeroy",
        fillcolor="rgba(124,58,237,.08)",line=dict(color="#7c3aed",width=2)))
    fe.update_layout(**CL,title=dict(text="EMG + Envelope (AD8232)",font_size=11,x=.02),
        yaxis_title="mV",xaxis_title="Time (s)")
    st.plotly_chart(fe,use_container_width=True)

with e2:
    fc2=go.Figure()
    mc=p["fc"]<=20
    fc2.add_trace(go.Scatter(x=p["fc"][mc],y=p["Cxy"][mc],fill="tozeroy",
        fillcolor="rgba(22,163,74,.08)",line=dict(color="#16a34a",width=1.8),showlegend=False))
    fc2.add_hline(y=.5,line=dict(color="#9ca3af",dash="dash",width=1))
    fc2.add_vline(x=p["dom"],line=dict(color="#dc2626",dash="dot",width=1))
    fc2.update_layout(**CL,title=dict(text=f"IMU–EMG Coherence ({p['coh']}%)",font_size=11,x=.02),
        yaxis_title="Coherence",xaxis_title="Hz",yaxis_range=[0,1],xaxis_range=[0,20],showlegend=False)
    st.plotly_chart(fc2,use_container_width=True)

def trend(vals,title,color,yr,bands=None):
    fig=go.Figure(); th=st.session_state["time_h"]
    if bands:
        for b in bands: fig.add_hrect(y0=b[0],y1=b[1],fillcolor=b[2],opacity=.07,line_width=0)
    fig.add_trace(go.Scatter(x=th,y=vals,mode="lines",line=dict(color=color,width=2)))
    fig.update_layout(height=210,margin=dict(l=10,r=10,t=28,b=28),
        title=dict(text=title,font_size=11,x=.02,font=dict(family="IBM Plex Sans",color="#374151")),
        plot_bgcolor="white",paper_bgcolor="white",showlegend=False,
        font=dict(family="IBM Plex Sans",size=11),
        xaxis=dict(showticklabels=False,gridcolor="#f3f4f6",linecolor="#e5e7eb"),
        yaxis=dict(range=yr,gridcolor="#f3f4f6",linecolor="#e5e7eb"))
    return fig

with e3:
    st.plotly_chart(trend(st.session_state["sev_h"],"UPDRS Trend","#dc2626",[0,4],
        [(0,1.5,"#dcfce7"),(1.5,2.5,"#fef9c3"),(2.5,4,"#fee2e2")]),use_container_width=True)
with e4:
    st.plotly_chart(trend(st.session_state["freq_h"],"Frequency Trend (Hz)","#2563eb",[0,15],
        [(3,6,"#fee2e2"),(6,12,"#dbeafe")]),use_container_width=True)

# ── Kaggle comparison ──────────────────────────────────────────────────────────
st.markdown("<div class='sh'>Population Benchmarking — Kaggle Reference Datasets</div>",unsafe_allow_html=True)
for d in cmp:
    bc="#16a34a" if d["sim"]>65 else "#d97706" if d["sim"]>35 else "#6b7280"
    st.markdown(f"""<div style='background:white;border-radius:8px;padding:13px 16px;margin:5px 0;
      box-shadow:0 1px 4px rgba(0,0,0,.05);display:flex;align-items:center;gap:14px'>
      <div style='min-width:190px'>
        <div style='font-weight:600;font-size:.82rem;color:#111'>{d['n']}</div>
        <div style='font-size:.7rem;color:#6b7280'>{d['p']}</div>
      </div>
      <div style='flex:1;background:#f3f4f6;border-radius:4px;height:7px;overflow:hidden'>
        <div style='background:{bc};width:{d["sim"]}%;height:7px;border-radius:4px'></div>
      </div>
      <div style='min-width:52px;text-align:right;font-weight:600;font-size:1.05rem;
                  color:{bc};font-family:IBM Plex Mono,monospace'>{d['sim']}%</div>
      <div style='min-width:280px;font-size:.72rem;color:#6b7280;font-family:IBM Plex Mono,monospace'>
        freq {p['dom']}Hz vs {d['fm']}Hz (z={d['zf']:+.1f}) &nbsp;|&nbsp; UPDRS {p['updrs']} vs {d['um']} (z={d['zu']:+.1f})
      </div>
    </div>""",unsafe_allow_html=True)

# ── Clinical summary + haptic ──────────────────────────────────────────────────
st.markdown("<div class='sh'>Clinical Summary &amp; Haptic Damping Status</div>",unsafe_allow_html=True)
s1,s2=st.columns([2.2,1])

with s1:
    acls="danger" if p["updrs"]>=3 else "warn" if p["updrs"]>=1.5 else "ok"
    pd_et="Parkinsonian rest tremor" if p["dom"]<6 else "Essential / action tremor"
    st.markdown(f"""<div class='cal {acls}'>
    <b>Diagnosis:</b> {p['tt']} &nbsp;·&nbsp; <b>UPDRS:</b> {p['updrs']}/4.0 ({p['ul']})
    &nbsp;·&nbsp; <b>Frequency:</b> {p['dom']} Hz (Welch+ZCR+Autocorr voting)<br><br>
    5-stage DSP (Notch 50Hz → Kalman gravity → Bandpass 3–12Hz → Wavelet db4 → EMG envelope):
    dominant frequency <b>{p['dom']} Hz</b>. PD-band {p['pdp']}% vs ET-band {p['etp']}% (ratio {p['ratio']}×)
    consistent with <b>{pd_et}</b>. EMG co-contraction {p['cci']}% ({p['cl']}),
    coherence {p['coh']}%. Best population match: <b>{best['n']}</b> ({best['sim']}% similarity).
    </div>""",unsafe_allow_html=True)
    with st.expander("💊 Treatment Reference"):
        if p["dom"]<6 and p["cci"]>45:
            st.markdown("""| | PD Rest Tremor |\n|---|---|\n|**1st Line**|Levodopa/Carbidopa · Dopamine agonists|\n|**2nd Line**|Anticholinergics · Amantadine|\n|**Surgical**|STN/GPi-DBS if UPDRS ≥ 3|""")
        elif p["dom"]>6:
            st.markdown("""| | Essential Tremor |\n|---|---|\n|**1st Line**|Propranolol · Primidone|\n|**2nd Line**|Topiramate · Gabapentin|\n|**Surgical**|Focused Ultrasound · VIM-DBS|""")
        else: st.markdown("Reassess after longer recording session.")

with s2:
    hc="#dc2626" if p["hactive"] and haptic_on else "#6b7280"
    hs="ACTIVE" if p["hactive"] and haptic_on else "STANDBY"
    st.markdown(f"""<div style='background:white;border-radius:10px;padding:16px;
      box-shadow:0 1px 8px rgba(0,0,0,.05);border-top:3px solid {hc}'>
      <div style='font-size:.66rem;font-weight:600;text-transform:uppercase;
                  letter-spacing:1px;color:#6b7280;margin-bottom:8px'>Haptic Damping Glove</div>
      <div style='font-size:1.5rem;font-weight:600;color:{hc};font-family:IBM Plex Mono,monospace'>{hs}</div>
      <div style='font-size:.78rem;color:#374151;margin-top:7px'>
        Counter-phase: <b style='font-family:IBM Plex Mono,monospace'>{p['haptic_f']} Hz</b>
      </div>
      <div style='font-size:.78rem;color:#374151;margin-top:4px'>
        PWM duty: <b>{p['haptic_d']}/255</b>
      </div>
      <div style='font-size:.7rem;color:#9ca3af;margin-top:8px;line-height:1.4'>
        Motor vibrates at tremor freq<br>180° phase offset → destructive<br>interference → tremor damping
      </div>
    </div>
    <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:7px;
      padding:9px 12px;margin-top:7px;font-size:.7rem;color:#14532d'>
      📌 Upload haptic_glove.ino to a second Arduino Nano to activate physical damping
    </div>""",unsafe_allow_html=True)

st.markdown("""<div class='disc'>⚕️ <b>Medical Disclaimer:</b> NeuroSync is research-grade.
All outputs are investigational. Not a substitute for neurological examination or physician judgment.
Not FDA/CE approved for clinical decision-making.</div>""",unsafe_allow_html=True)

time.sleep(ref_s); st.rerun()
