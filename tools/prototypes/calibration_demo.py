"""Test cosmic.calibration (P02 lead novelty) end-to-end on a synthetic field+cluster
mixture with a deliberately MISCALIBRATED p-tilde. Shows the reliability diagram catches
it + isotonic recalibration fixes it. Visual: reliability curve before/after."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cosmic.calibration as cal
rng=np.random.default_rng(7)
N=2000
# true membership: latent score -> true prob -> label
score=rng.normal(0,1.5,N); p_true=1/(1+np.exp(-score)); y=(rng.random(N)<p_true).astype(int)
# miscalibrated p-tilde: OVERCONFIDENT (sharpened) -- like an uncalibrated ranking statistic
ptilde=np.clip(1/(1+np.exp(-1.8*score)),1e-4,1-1e-4)   # steeper slope -> overconfident
# diagnostics BEFORE
rd=cal.reliability_diagram(ptilde,y,n_bins=10); hl=cal.hosmer_lemeshow(ptilde,y)
b0=cal.brier_score(ptilde,y); ece0=float(np.ravel(cal.expected_calibration_error(ptilde,y))[0])
# recalibrate (isotonic) + diagnostics AFTER
rec=cal.fit_isotonic(ptilde,y); pcal=np.asarray(rec.transform(ptilde) if hasattr(rec,'transform') else rec(ptilde))
rd2=cal.reliability_diagram(pcal,y,n_bins=10); hl2=cal.hosmer_lemeshow(pcal,y)
b1=cal.brier_score(pcal,y); ece1=float(np.ravel(cal.expected_calibration_error(pcal,y))[0])
print(f"BEFORE: Brier={b0:.4f} ECE={ece0:.4f} H-L stat={hl.statistic:.1f} p={hl.pvalue:.4f} -> {'MISCALIBRATED' if hl.pvalue<0.05 else 'ok'}")
print(f"AFTER (isotonic): Brier={b1:.4f} ECE={ece1:.4f} H-L stat={hl2.statistic:.1f} p={hl2.pvalue:.4f} -> {'ok' if hl2.pvalue>0.05 else 'still off'}")
# reliability plot: use bin predicted vs observed
def curve(rd):
    for a in ('bin_confidence','mean_predicted','bin_predicted','predicted'):
        if hasattr(rd,a): px=np.asarray(getattr(rd,a)); break
    for a in ('bin_accuracy','observed_frequency','bin_observed','observed','accuracy'):
        if hasattr(rd,a): py=np.asarray(getattr(rd,a)); break
    return px,py
px0,py0=curve(rd); px1,py1=curve(rd2)
fig,ax=plt.subplots(figsize=(6,6))
ax.plot([0,1],[0,1],'k--',lw=1,label='perfect calibration')
ax.plot(px0,py0,'o-',c='C3',label=f'p̃ before (H-L p={hl.pvalue:.3f}, Brier {b0:.3f})')
ax.plot(px1,py1,'s-',c='C0',label=f'after isotonic (p={hl2.pvalue:.3f}, Brier {b1:.3f})')
ax.set_xlabel('predicted membership probability'); ax.set_ylabel('observed member fraction')
ax.set_title('cosmic.calibration: reliability diagram\n(synthetic mixture, overconfident p̃ -> isotonic recalibration)')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect('equal')
out="/private/tmp/claude-501/-Users-notluquis-phd/9a0f9b71-83f6-46ec-ba5e-2fc967bc6851/scratchpad/reliability.png"
plt.tight_layout(); plt.savefig(out,dpi=110); print("saved",out)
