"""Offline geometry and demo checks for this task (no GPU). Run in the yam-isaac-lab-arena environment:
    python check_offline.py
"""
import sys, itertools, numpy as np; import pathlib; HERE=pathlib.Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from yam_arena import task_file, ik, station
from yam_arena.demo import ArmPlan, grip_rotation
t=task_file.load(HERE/"seat_sfp_module.py")
import seat_sfp_module as T, sfp_seat_scoring as sc
card=[(np.array(o.pos),T.R_CARD,np.array(o.size)/2) for o in t.objects if o.name.startswith("card_") and o.name!="card_base"]
mod=T._module_boxes()
def sat_gap(c1,R1,h1,c2,R2,h2):
    """Separation (m) of two oriented boxes: >0 apart, <0 penetration depth (SAT, min over 15 axes of signed overlap)."""
    d=c2-c1; best=-1e9
    axes=[R1[:,i] for i in range(3)]+[R2[:,i] for i in range(3)]+[np.cross(R1[:,i],R2[:,j]) for i in range(3) for j in range(3)]
    for a in axes:
        n=np.linalg.norm(a)
        if n<1e-9: continue
        a=a/n
        r1=sum(h1[i]*abs(a@R1[:,i]) for i in range(3)); r2=sum(h2[i]*abs(a@R2[:,i]) for i in range(3))
        best=max(best,abs(a@d)-r1-r2)
    return best
def min_gap(origin,R):
    g=1e9
    for c,s,_ in mod:
        cm=origin+R@np.array(c); hm=np.array(s)/2
        for cc,Rc,hc in card:
            if np.linalg.norm(cm-cc)>0.2: continue
            g=min(g,sat_gap(cm,R,hm,cc,Rc,hc))
    return g
d0=T.R_HOLD@np.array([0,0,T.HEAD_Z])+np.array([0,0,T.GRIP_UP])
def at(pos,rot):
    Rz=rot@np.linalg.inv(grip_rotation(0.0)); return pos-Rz@d0, Rz@T.R_HOLD
tipf=lambda dy,d: T.CARD_POS+T.R_CARD@np.array([sc.BAY_X,sc.BAY_Y[0]+dy,sc.MOUTH_Z-d])
org=lambda tip: tip-T.R_CARD@np.array([0,0,sc.TIP_Z])

tipf=lambda dy,dx,d: T.CARD_POS+T.R_CARD@np.array([sc.BAY_X+dx,sc.BAY_Y[0]+dy,sc.MOUTH_Z-d])
org=lambda tip: tip-T.R_CARD@np.array([0,0,sc.TIP_Z])
print("centred gap to the port along the push (mm) at depth 0/10/20/30/40:",[round(float(1000*min_gap(org(tipf(0,0,d)),T.R_CARD)),2) for d in (0,.01,.02,.03,.04)])
print("control, 3 mm sideways at 30 mm (must be negative):",round(float(1000*min_gap(org(tipf(0.003,0,0.03)),T.R_CARD)),2))
plan=T.demo()["left"]; arm=ArmPlan(plan, station.BASES["left"]); per={}
acts=np.array([arm.action7(x)[0] for x in np.arange(0,arm.duration,0.05)])
print("demo %.1f s, worst IK %.2f mm %.2f deg, joints within limits: %s"%(arm.duration,arm.worst_ik_mm,arm.worst_ik_deg,bool(np.all(acts[:,:6]>=ik.LOWER) and np.all(acts[:,:6]<=ik.UPPER))))
arm2=ArmPlan(plan, station.BASES["left"])
for x in np.arange(0,arm2.duration,0.05):
    pos,rot,fin,label=arm2.target(x)
    if label in ("release","retreat along the axis","done"): continue
    o,R=at(pos,rot); per[label]=min(per.get(label,1e9),min_gap(o,R))
print("module-to-card clearance per demo segment (mm):",{k:round(float(v*1000),1) for k,v in per.items()})
