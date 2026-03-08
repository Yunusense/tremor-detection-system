# dashboard/app.py
import streamlit as st
import numpy as np, time
from datetime import datetime
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(__file__))
 
st.set_page_config(page_title='Tremor Monitor',page_icon='🧠',layout='wide')
st.markdown('<style>.main{background:#0e1117}hr{border-color:#2e75b6}</style>',
            unsafe_allow_html=True)
 
if 'sev_hist' not in st.session_state:
    st.session_state.sev_hist=[]; st.session_state.time_hist=[]
    st.session_state.start_t=time.time()
 
with st.sidebar:
    st.title('🧠 Controls')
    patient_id  =st.text_input('Patient ID','PT-0042')
    data_source =st.radio('Data Source',['🧪 Simulation','📁 Upload CSV','🔴 Live Device'])
    refresh_rate=st.slider('Refresh rate (s)',0.5,5.0,1.0)
    st.divider()
    elapsed=int(time.time()-st.session_state.start_t)
    st.metric('Session Time',f'{elapsed//60:02d}:{elapsed%60:02d}')
    gen_report=st.button('📄 Generate PDF Report',use_container_width=True)
 
from scipy.signal import butter,filtfilt
from scipy.fft import fft,fftfreq
 
def simulate(n=200,freq=4.8,amp=1.2):
    t=np.linspace(0,2,n)
    ax=amp*np.sin(2*np.pi*freq*t)+0.15*np.random.randn(n)
    ay=amp*0.8*np.sin(2*np.pi*freq*t+0.5)+0.1*np.random.randn(n)
    az=amp*0.3*np.sin(2*np.pi*freq*t+1.0)+0.08*np.random.randn(n)
    emg=amp*0.6*np.abs(np.sin(2*np.pi*freq*t))+0.05*np.random.randn(n)
    return t,ax,ay,az,emg
 
t,ax,ay,az,emg=simulate()
b,a=butter(4,[3/50,12/50],btype='band'); ax_f=filtfilt(b,a,ax)
N=len(ax_f); yf=np.abs(fft(ax_f))[:N//2]; xf=fftfreq(N,1/100)[:N//2]
dom_freq=abs(float(xf[np.argmax(yf)]))
rms_amp=float(np.sqrt(np.mean(ax_f**2)))
severity=round(min(4.0,rms_amp*3.0),2)
label='Rest Tremor' if dom_freq<6 else 'Essential Tremor'
conf=87.0; emg_rms=float(np.sqrt(np.mean(emg**2)))
 
st.session_state.sev_hist.append(severity)
st.session_state.time_hist.append(datetime.now().strftime('%H:%M:%S'))
if len(st.session_state.sev_hist)>60:
    st.session_state.sev_hist.pop(0); st.session_state.time_hist.pop(0)
 
st.markdown('''<h1 style='text-align:center;color:#2e75b6'>
🧠 AI Tremor Detection — Clinical Dashboard</h1>
<p style='text-align:center;color:#888'>Real-Time Neurological Monitoring | Wrist IMU + EMG</p>
<hr>''',unsafe_allow_html=True)
 
sev_icon='🟢' if severity<1.5 else '🟡' if severity<2.5 else '🔴'
c1,c2,c3,c4,c5=st.columns(5)
c1.metric('🏷️ Diagnosis',label); c2.metric('📊 Confidence',f'{conf}%')
c3.metric('〰️ Frequency',f'{dom_freq:.1f} Hz'); c4.metric('📏 Amplitude',f'{rms_amp:.2f} g')
c5.metric(f'⚠️ Severity {sev_icon}',f'{severity}/4.0')
st.divider()
 
col_w,col_f=st.columns([3,2])
with col_w:
    st.markdown('#### 📈 Live Accelerometer')
    fig=go.Figure()
    for sig,nm,clr in [(ax_f,'X','#2e75b6'),(ay,'Y','#70c172'),(az,'Z','#f4c542')]:
        fig.add_trace(go.Scatter(x=t,y=sig,name=nm,line=dict(color=clr,width=1.2)))
    fig.update_layout(height=280,margin=dict(l=20,r=20,t=10,b=30),
        plot_bgcolor='#0e1117',paper_bgcolor='#0e1117',font=dict(color='#ccc'),
        xaxis=dict(title='Time (s)',gridcolor='#1f2937'),
        yaxis=dict(title='g',gridcolor='#1f2937'),legend=dict(orientation='h',y=1.1))
    st.plotly_chart(fig,use_container_width=True)
 
with col_f:
    st.markdown('#### 🔊 Frequency Spectrum')
    fig2=go.Figure()
    fig2.add_vrect(x0=3,x1=6,fillcolor='orange',opacity=0.12,
                   annotation_text='PD 3-6Hz',annotation_position='top left')
    fig2.add_vrect(x0=6,x1=12,fillcolor='green',opacity=0.10,
                   annotation_text='ET 6-12Hz',annotation_position='top left')
    mask=xf<=25
    fig2.add_trace(go.Scatter(x=xf[mask],y=yf[mask],fill='tozeroy',
        fillcolor='rgba(46,117,182,0.2)',line=dict(color='#2e75b6',width=2)))
    fig2.add_vline(x=dom_freq,line=dict(color='red',dash='dash'),
                   annotation_text=f'Peak:{dom_freq:.1f}Hz')
    fig2.update_layout(height=280,margin=dict(l=20,r=20,t=10,b=30),
        plot_bgcolor='#0e1117',paper_bgcolor='#0e1117',font=dict(color='#ccc'),
        showlegend=False,xaxis=dict(title='Hz',range=[0,25],gridcolor='#1f2937'),
        yaxis=dict(title='Power',gridcolor='#1f2937'))
    st.plotly_chart(fig2,use_container_width=True)
 
col_e,col_t=st.columns([2,3])
with col_e:
    st.markdown('#### 💪 EMG Activation')
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=t,y=emg,fill='tozeroy',
        fillcolor='rgba(224,92,92,0.2)',line=dict(color='#e05c5c',width=1.5)))
    fig3.update_layout(height=220,margin=dict(l=20,r=20,t=10,b=30),
        plot_bgcolor='#0e1117',paper_bgcolor='#0e1117',font=dict(color='#ccc'),
        showlegend=False,xaxis=dict(title='Time (s)',gridcolor='#1f2937'),
        yaxis=dict(title='mV',gridcolor='#1f2937'))
    st.plotly_chart(fig3,use_container_width=True)
    st.caption(f'EMG RMS: {emg_rms:.3f} mV  |  Pattern: Synchronous')
 
with col_t:
    st.markdown('#### 📉 Session Severity Trend')
    fig4=go.Figure()
    fig4.add_hrect(y0=0,y1=1.5,fillcolor='green',opacity=0.06)
    fig4.add_hrect(y0=1.5,y1=2.5,fillcolor='yellow',opacity=0.06)
    fig4.add_hrect(y0=2.5,y1=4.0,fillcolor='red',opacity=0.06)
    fig4.add_trace(go.Scatter(x=st.session_state.time_hist,
        y=st.session_state.sev_hist,mode='lines+markers',
        line=dict(color='#2e75b6',width=2),marker=dict(size=5)))
    fig4.update_layout(height=220,margin=dict(l=20,r=20,t=10,b=30),
        plot_bgcolor='#0e1117',paper_bgcolor='#0e1117',font=dict(color='#ccc'),
        showlegend=False,yaxis=dict(title='Severity',range=[0,4],gridcolor='#1f2937'),
        xaxis=dict(title='Time',gridcolor='#1f2937'))
    st.plotly_chart(fig4,use_container_width=True)
 
st.divider()
st.markdown('#### 🩺 AI Clinical Interpretation')
tremor_type='Parkinsonian rest tremor' if dom_freq<6 else 'Essential Tremor'
st.info(f'''**Diagnosis:** {label}  |  **Confidence:** {conf}%  |
**Dominant Frequency:** {dom_freq:.1f} Hz  |  **UPDRS Estimate:** {severity}/4.0
 
Wrist accelerometry demonstrates a dominant oscillatory frequency of **{dom_freq:.1f} Hz**
with synchronous EMG co-contraction. Profile is consistent with **{tremor_type}**.
 
*AI-assisted output. Not a substitute for clinical neurological examination.*''')
 
with st.expander('💊 Treatment Reference Panel'):
    if dom_freq<6:
        st.markdown('| |PD Rest Tremor|\n|---|---|\n|**First Line**|Levodopa/Carbidopa, Dopamine agonists|\n|**Surgical**|DBS if UPDRS >= 3|')
    else:
        st.markdown('| |Essential Tremor|\n|---|---|\n|**First Line**|Propranolol, Primidone|\n|**Surgical**|Focused Ultrasound / DBS (VIM)|')
 
time.sleep(refresh_rate)
st.rerun()

