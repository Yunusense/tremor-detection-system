# dashboard/utils/inference.py
import joblib, numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
 
class TremorEngine:
    def __init__(self):
        self.model  = joblib.load('models/ensemble_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.le     = joblib.load('models/label_encoder.pkl')
        self.fs     = 100
 
    def _features(self,ax_win,emg_win):
        b,a=butter(4,[3/50,12/50],btype='band')
        ax_f=filtfilt(b,a,ax_win)
        N=len(ax_f); yf=np.abs(fft(ax_f))[:N//2]; xf=fftfreq(N,1/self.fs)[:N//2]
        return np.array([[
            float(np.sqrt(np.mean(ax_f**2))),float(np.ptp(ax_f)),
            float(np.sum(np.diff(np.sign(ax_f))!=0)/len(ax_f)),
            float(np.sum(np.abs(np.diff(ax_f)))),
            float(skew(ax_f)),float(kurtosis(ax_f)),float(np.var(ax_f)),
            float(xf[np.argmax(yf)]),
            float(np.sum(xf*yf)/(np.sum(yf)+1e-8)),
            float(np.sum(yf[(xf>=3)&(xf<=6)])/(np.sum(yf)+1e-8)),
            float(np.sum(yf[(xf>=6)&(xf<=12)])/(np.sum(yf)+1e-8)),
            float(np.sum(yf[(xf>=3)&(xf<=12)])/(np.sum(yf)+1e-8)),
            float(np.sqrt(np.mean(emg_win**2))),float(np.var(emg_win)),
            float(np.sum(np.diff(np.sign(emg_win))!=0)/len(emg_win)),
        ]])
 
    def predict(self,ax_win,emg_win):
        feat=self._features(ax_win,emg_win)
        feat_s=self.scaler.transform(feat)
        probs=self.model.predict_proba(feat_s)[0]
        idx=int(np.argmax(probs)); conf=float(probs[idx])
        label=self.le.classes_[idx] if conf>=0.65 else 'indeterminate'
        N=len(ax_win); yf=np.abs(fft(ax_win))[:N//2]; xf=fftfreq(N,1/self.fs)[:N//2]
        dom_f=abs(float(xf[np.argmax(yf)]))
        sev=round(min(4.0,float(np.sqrt(np.mean(ax_win**2)))*3.0),2)
        return{'label':label,'confidence':round(conf*100,1),
               'dom_freq':round(dom_f,2),'severity':sev,'updrs':int(np.ceil(sev)),
               'probs':dict(zip(self.le.classes_,[round(p*100,1) for p in probs.tolist()]))}

