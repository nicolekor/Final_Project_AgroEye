# -*- coding: utf-8 -*-
"""
LeafEnsemble Serving (v4.6.6 rules port)
- 코랩 스크립트의 핵심 규칙(게이트/가드/엔트로피 완화/TTA/Rice expert 하이브리드/override 등) 이식
- 단일 이미지 추론 서빙 파이프라인에 맞게 배치·플롯·샘플링 제거
- 현재 프로젝트 구조(스크린샷)에 맞춘 경로 기본값
"""

from __future__ import annotations
import os, json, time, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision

# -------- Optional cv2 ----------
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ===================== PATHS / CONFIG =====================
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent)).resolve()
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", MODEL_DIR / "weights")).resolve()

CKPT_MN = Path(os.getenv("CKPT_MN", WEIGHTS_DIR / "MN" / "best.pth"))
CKPT_RN = Path(os.getenv("CKPT_RN", WEIGHTS_DIR / "RN" / "best.pth"))
CKPT_RN_RICE_EXPERT = Path(os.getenv("CKPT_RN_RICE_EXPERT", WEIGHTS_DIR / "RN" / "rn_rice_ft_best.pth"))

CLASS_TO_IDX_JSON = Path(os.getenv("CLASS_TO_IDX_JSON", WEIGHTS_DIR / "class_to_idx.json"))

TEMP_CLASSWISE_JSON = Path(os.getenv("TEMP_CLASSWISE_JSON", MODEL_DIR / "calibration" / "temperature_classwise_v1.fixed.json"))
TEMP_SCALAR_JSON    = Path(os.getenv("TEMP_SCALAR_JSON",    MODEL_DIR / "calibration" / "temperature_scalar.json"))  # optional

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

def _select_device() -> str:
    """CUDA 가능 시 GPU, 아니면 CPU 선택 + 1회 확인 메시지 출력"""
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA Device 0"
        print(f"[LeafEnsemble] Using GPU: {name}")
        return "cuda"
    print("[LeafEnsemble] Using CPU (CUDA not available)")
    return "cpu"

DEVICE = _select_device()

# Ensemble
ENSEMBLE = dict(mode=os.getenv("ENSEMBLE_MODE", "prob"), w_mn=float(os.getenv("ENSEMBLE_W_MN", "0.25")), w_rn=float(os.getenv("ENSEMBLE_W_RN", "0.75")))

# Temperature floor
T_FLOOR = float(os.getenv("T_FLOOR", "1.00"))

# ======= v4.6.6 RULE CONFIGS (코랩과 동일/동등) =======
ENTROPY = dict(
    use_quantile=False,        # 서빙에선 검증셋이 없으므로 False, 기본 1.5에서 완화 규칙만 적용
    q=0.92, pad=0.03, k=1.5,
    relax_when_leaf_rn_strong=1.35,
    relax_when_bigleaf_rn_mid=1.40
)
RN_PREF = dict(conf_min=0.75, margin_min=0.20, dconf_tta=0.30)
LEAF_GATE = dict(area_min=0.035, exg_min=-0.30, rescue_gy_min=0.03, rescue_sat_min=0.15)
NECROSIS = dict(
    use=True, edge_tau=0.035,
    lab_a_min=-5.0, lab_a_max=18.0, lab_b_min=5.0, lab_b_max=40.0,
    sat_max=0.40, gy_max=0.15, rn_conf_min=0.80, rn_margin_min=0.20,
    class_whitelist=[
        "Corn___Northern_Leaf_Blight",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Tomato___Late_blight",
        "Tomato___Target_Spot",
        "Potato___Late_blight",
        "Rice___Brown_Spot",
        "Tomato___Early_blight",
    ],
)
GLOBAL_OOD = dict(leaf_min=0.10, gy_min=0.07, sat_min=0.05, hi_max=0.55, conf_min=0.60, agree_conf=0.85, agree_margin=0.12, H_k=1.5)
GUARD_CFG = {
    "Orange___Haunglongbing_(Citrus_greening)": dict(
        leaf_min=0.32, gy_min=0.20, sat_min=0.08, hi_max=0.18,
        conf_cap=0.42, margin_min=0.25, dconf_cap=0.22, require_leaf=True,
        rn_min=0.84, rn_margin_min=0.22, leaf_min_override=0.10, gy_min_override=0.10,
        hard_no_leaf=True, hard_leaf_min=0.08, hard_gy_min=0.10
    ),
    "Rice___Sheath_Blight": dict(
        leaf_min=0.20, gy_min=0.10, sat_min=0.06, hi_max=0.25,
        conf_cap=0.50, margin_min=0.12, dconf_cap=0.30
    ),
    "Tomato___Late_blight": dict(
        leaf_min=0.30, gy_min=0.20, sat_min=0.06, hi_max=0.18,
        conf_cap=0.45, margin_min=0.25, dconf_cap=0.22,
        require_leaf=True, rn_min=0.88, rn_margin_min=0.35,
        necrosis_relax=True, nec_sat_max=0.46, nec_gy_max=0.15,
        leaf_min_override=0.02, gy_min_override=0.05
    ),
    "Cucumber___Powdery_mildew": dict(
        leaf_min=0.20, gy_min=0.09, sat_min=0.05, hi_max=0.35,
        conf_cap=0.92, margin_min=0.10, dconf_cap=0.30
    ),
    "Strawberry___Leaf_scorch": dict(
        leaf_min=0.25, gy_min=0.20, sat_min=0.15, hi_max=0.35,
        conf_cap=0.52, margin_min=0.10, dconf_cap=0.30,
        require_leaf=True, hard_no_leaf=True, hard_leaf_min=0.10, hard_gy_min=0.12
    ),
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": dict(
        leaf_min=0.10, gy_min=0.12, sat_min=0.08, hi_max=0.35,
        conf_cap=0.88, margin_min=0.15, dconf_cap=0.30
    ),
    "Corn___Common_rust": dict(
        leaf_min=0.12, gy_min=0.16, sat_min=0.06, hi_max=0.35,
        conf_cap=0.80, margin_min=0.20, dconf_cap=0.30,
        require_leaf=True, hard_no_leaf=True, hard_leaf_min=0.08, hard_gy_min=0.12
    ),
}
CLASS_ENTROPY_RELAX = {
    "Potato___Late_blight": 1.45,
    "Tomato___Spider_mites Two-spotted_spider_mite": 1.70,
    "Corn___Northern_Leaf_Blight": 1.50,
    "Grape___healthy": 1.65,
    "Orange___Haunglongbing_(Citrus_greening)": 1.85,
    "Rice___Leaf_Blast": 1.75,
    "Tomato___Septoria_leaf_spot": 1.80,
}
RELAX_RULES = {
    "Potato___Late_blight": dict(rn_conf_min=0.88, rn_margin_min=0.20, require_leaf=True),
    "Tomato___Spider_mites Two-spotted_spider_mite": dict(rn_conf_min=0.80, rn_margin_min=0.25, require_leaf=True, leaf_min=0.18, gy_min=0.18),
    "Corn___Northern_Leaf_Blight": dict(rn_conf_min=0.82, rn_margin_min=0.20, require_leaf=True, leaf_min=0.15, gy_min=0.20),
    "Grape___healthy": dict(rn_conf_min=0.82, rn_margin_min=0.20, require_leaf=True, leaf_min=0.35, gy_min=0.32),
    "Orange___Haunglongbing_(Citrus_greening)": dict(rn_conf_min=0.78, rn_margin_min=0.20, require_leaf=True, leaf_min=0.16, gy_min=0.16),
    "Rice___Leaf_Blast": dict(rn_conf_min=0.80, rn_margin_min=0.20, require_leaf=True, leaf_min=0.25, gy_min=0.35),
    "Tomato___Septoria_leaf_spot": dict(rn_conf_min=0.85, rn_margin_min=0.25, require_leaf=True, leaf_min=0.18, gy_min=0.18, hi_max=0.10),
}
OVERRIDE = dict(
    leaf_big_rn_mid=dict(leaf_min=0.65, rn_min=0.70, rn_margin_min=0.18),
    guard_override=dict(rn_min=0.90, rn_margin_min=0.40, require_leaf=True,
                        leaf_min_override=0.10, gy_min_override=0.08),
    guard_override_deny=["Strawberry___Leaf_scorch"],
    ensemble_sameclass=dict(conf_min=0.85)
)
CONSENSUS = dict(
    leaf_min=0.25, gy_min=0.15, mean_conf_min=0.80, min_margin_min=0.25,
    entropy_cap=1.40, edge_min=0.02, max_aspect=6.0
)
RNHIGH = dict(
    leaf_min=0.22, gy_min=0.20, sat_min=0.08, same_or_abs=True,
    cr_abs=0.95, mr_abs=0.30, guard_abs_cr=0.92, guard_abs_mr=0.35,
    entropy_cap=1.45, edge_min=0.02
)
RICE = dict(
    gate_leaf_min=0.30, gate_gy_min=0.35, gate_entropy_max=1.65,
    blend_alpha=float(os.getenv("RICE_BLEND_ALPHA", "0.60")),
    accept_conf=0.75, accept_margin=0.25,
    veto_edge_min=0.02, max_aspect=6.0,
    water_veto_h_lo=70, water_veto_h_hi=110, water_veto_s_lo=100, water_veto_frac=0.10,
    gate_alt_leaf_min=0.22, gate_alt_aspect_min=3.0, gate_alt_mr_max=0.30, gate_alt_cr_max=0.60,
)

# ===================== UTILS / METRICS =====================
def _excess_green(arr_rgb):
    r = arr_rgb[...,0].astype(np.float32)
    g = arr_rgb[...,1].astype(np.float32)
    b = arr_rgb[...,2].astype(np.float32)
    exg = 2*g - r - b
    exg_norm = (exg - exg.mean())/(exg.std()+1e-6)
    return exg_norm

def _red_ratio(arr_rgb, r_thresh=150, g_max=160, b_max=160, mask=None):
    r = arr_rgb[...,0]; g = arr_rgb[...,1]; b = arr_rgb[...,2]
    red_mask = (r >= r_thresh) & (g <= g_max) & (b <= b_max)
    if mask is not None:
        red_mask = red_mask & mask.astype(bool)
        denom = max(1, int(mask.sum()))
        return float(red_mask.sum()/denom)
    return float(red_mask.mean())

def _bbox_from_mask(mask, pad=4):
    ys, xs = np.where(mask>0)
    if len(ys)==0: return None
    y1, y2 = ys.min(), ys.max(); x1, x2 = xs.min(), xs.max()
    y1 = max(0, y1-pad); x1 = max(0, x1-pad)
    y2 = min(mask.shape[0]-1, y2+pad); x2 = min(mask.shape[1]-1, x2+pad)
    return (y1,x1,y2,x2)

def _edge_density(mask_uint8):
    if not _HAS_CV2 or mask_uint8 is None: return 0.0
    edges = cv2.Canny(mask_uint8, 60, 120)
    return float(edges.mean())

def _lab_stats(arr_rgb, mask=None):
    if not _HAS_CV2: return 0.0,0.0
    lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[...,1]-128.0; b = lab[...,2]-128.0
    if mask is not None and mask.sum()>0:
        a = a[mask.astype(bool)]; b = b[mask.astype(bool)]
    return float(np.mean(a)), float(np.mean(b))

def _adaptive_hsv_mask(arr):
    if not _HAS_CV2:
        g = arr[...,1].astype(np.int32); r = arr[...,0].astype(np.int32); b = arr[...,2].astype(np.int32)
        return (g - ((r+b)//2) > 18).astype(np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Vc = clahe.apply(V)
    S_low = int(max(50, np.percentile(S, 20)))
    V_low = int(max(45, np.percentile(Vc, 15)))
    mask = ((H>=30)&(H<=95)&(S>=S_low)&(Vc>=V_low)).astype(np.uint8)
    k = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask

def leaf_metrics(img_pil: Image.Image):
    arr = np.array(img_pil.convert("RGB"))
    h,w,_ = arr.shape
    mask0 = _adaptive_hsv_mask(arr)
    if _HAS_CV2:
        num, labels = cv2.connectedComponents(mask0)
        max_area=0; max_lab=0
        for lab in range(1,num):
            area = int((labels==lab).sum())
            if area>max_area: max_area=area; max_lab=lab
        mask_largest = (labels==max_lab).astype(np.uint8) if max_lab>0 else (mask0*0)
        leaf_area = max_area/float(h*w)
    else:
        mask_largest = mask0; leaf_area = float(mask0.mean())

    gy_ratio = float(mask0.mean())
    exg = _excess_green(arr); exg_mean = float(exg.mean())
    bbox = _bbox_from_mask(mask_largest, pad=4)

    if bbox is not None:
        y1,x1,y2,x2 = bbox
        box = arr[y1:y2+1, x1:x2+1, :]
        mask_box = mask_largest[y1:y2+1, x1:x2+1]
        exg_mean_box = float(_excess_green(box)[mask_box.astype(bool)].mean()) if mask_box.sum()>0 else -1.0
        red_frac_box = _red_ratio(box, mask=mask_box)
        edge_den_box = _edge_density((mask_box*255).astype(np.uint8))
        lab_a, lab_b = _lab_stats(box, mask_box)
        hh = max(1, y2-y1+1); ww = max(1, x2-x1+1)
        aspect = max(hh, ww)/max(1, min(hh, ww))
    else:
        exg_mean_box = exg_mean
        red_frac_box = _red_ratio(arr)
        edge_den_box = 0.0
        lab_a, lab_b = _lab_stats(arr, None)
        aspect = 1.0

    return (leaf_area, exg_mean, gy_ratio, bbox, exg_mean_box,
            red_frac_box, mask_largest, edge_den_box, lab_a, lab_b, aspect)

def highlight_ratio(img_pil, thresh=240):
    im = np.asarray(img_pil.convert("RGB"))
    gray = (0.299*im[...,0] + 0.587*im[...,1] + 0.114*im[...,2])
    return float((gray >= thresh).mean())

def saturation_ratio(img_pil, s_thresh=60):
    arr = np.array(img_pil.convert("RGB"))
    if _HAS_CV2:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        S = hsv[...,1]
        return float((S >= s_thresh).mean())
    r = arr[...,0].astype(np.float32); g = arr[...,1].astype(np.float32); b = arr[...,2].astype(np.float32)
    mean = (r+g+b)/3.0
    sat = (np.abs(r-mean)+np.abs(g-mean)+np.abs(b-mean))/3.0
    return float((sat >= 30).mean())

def water_like_ratio(img_pil, h_lo=70, h_hi=110, s_lo=100):
    arr = np.array(img_pil.convert("RGB"))
    if not _HAS_CV2: return 0.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    mask = (H >= h_lo) & (H <= h_hi) & (S >= s_lo)
    return float(mask.mean())

# ===================== MODEL / TEMPERATURE =====================
def _canon(s: str):
    s = s.replace("___","_").replace(",", "").replace(" ","_")
    s = s.replace("(","_").replace(")","_").replace("/","_").replace("\\","_")
    while "__" in s: s = s.replace("__","_")
    return s.strip().lower()

def _load_classes_from_class_to_idx(p: Path) -> List[str]:
    if not p.is_file():
        raise RuntimeError(f"class_to_idx.json not found: {p}")
    js = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(js, dict) and "classes" in js and isinstance(js["classes"], dict):
        js = js["classes"]
    if isinstance(js, dict):
        kt = {type(k) for k in js.keys()}
        vt = {type(v) for v in js.values()}
        if kt == {str} and vt <= {int, float}:
            pairs = [(int(v), k) for k, v in js.items()]
        elif kt <= {str, int} and vt == {str}:
            pairs = [(int(k), v) for k, v in js.items()]
        else:
            raise RuntimeError("Unsupported class_to_idx.json format")
        pairs.sort(key=lambda x: x[0])
        return [name for _, name in pairs]
    raise RuntimeError("Unsupported class_to_idx.json format")

def robust_load_T_classwise(json_path: Path, classes: List[str], model_key: str) -> List[float]:
    js = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(js, dict) and "classes" in js and model_key in js:
        keys = js["classes"]; blob = js[model_key]
        if isinstance(blob, list) and len(blob)==len(keys):
            m={}
            for k,v in zip(keys, blob):
                if isinstance(v, dict) and "T" in v: m[_canon(k)] = float(v["T"])
                elif isinstance(v, (int,float)):    m[_canon(k)] = float(v)
            return [max(m.get(_canon(c),1.0), T_FLOOR) for c in classes]
    blob = js.get(model_key) if isinstance(js, dict) else None
    if isinstance(blob, dict) and "T" in blob and isinstance(blob["T"], list):
        return [max(float(x), T_FLOOR) for x in blob["T"]]
    if isinstance(blob, list):
        return [max(float(x), T_FLOOR) for x in blob]
    if isinstance(blob, dict):
        m={}
        for k,v in blob.items():
            if isinstance(v, dict) and "T" in v: m[_canon(k)] = float(v["T"])
            elif isinstance(v, (int,float)):    m[_canon(k)] = float(v)
        if m:
            return [max(m.get(_canon(c),1.0), T_FLOOR) for c in classes]
    raise ValueError("Unsupported class-wise T JSON format")

def load_scalar_T(path: Path) -> Tuple[float, float]:
    if not path.is_file(): return 1.0, 1.0
    try:
        js=json.loads(path.read_text(encoding="utf-8"))
        return float(js.get("mn",1.0)), float(js.get("rn",1.0))
    except Exception:
        return 1.0, 1.0

# ===================== INFER / TTA / ENSEMBLE =====================
tfm_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@torch.inference_mode()
def infer_batch(model, images: List[Image.Image], T_vec=None, return_logits=True):
    x = torch.stack([tfm_eval(im) for im in images]).to(DEVICE, non_blocking=(DEVICE=="cuda"))
    t0 = time.time()
    # 최신 API 사용, CPU에서는 비활성화하여 경고 제거
    if DEVICE == "cuda":
        amp = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        amp = torch.amp.autocast("cpu", enabled=False)
    with amp:
        z = model(x)
        if T_vec is not None:
            if not torch.is_tensor(T_vec):
                T_vec = torch.tensor(T_vec, dtype=z.dtype, device=z.device)
            z = z / T_vec[None,:]
        p = F.softmax(z, dim=1)
        top2 = p.topk(2, dim=1)
        conf = top2.values[:,0]
        margin = top2.values[:,0] - top2.values[:,1]
        idx = p.argmax(1)
    out = {"probs":p, "conf":conf, "margin":margin, "idx":idx, "time":time.time()-t0}
    if return_logits: out["logits"]=z
    return out

def entropy(p: torch.Tensor) -> float:
    return float((-p.clamp_min(1e-12) * (p.clamp_min(1e-12)).log()).sum().item())

def ensemble_probs(pm, pr):
    if ENSEMBLE["mode"]=="logit":
        z = torch.log(pm.clamp_min(1e-12))*ENSEMBLE["w_mn"] + torch.log(pr.clamp_min(1e-12))*ENSEMBLE["w_rn"]
        pe = torch.softmax(z, dim=0)
    else:
        pe = pm*ENSEMBLE["w_mn"] + pr*ENSEMBLE["w_rn"]
        pe = pe/pe.sum()
    return pe

def _tta2_views(img_pil: Image.Image):
    w,h = img_pil.size; crops=[]
    s = int(min(w,h)*0.90); left=(w-s)//2; top=(h-s)//2
    crops.append(img_pil.crop((left,top,left+s,top+s)))
    s2 = int(min(w,h)*0.85)
    crops.append(img_pil.crop((0,0,s2,s2)))
    crops.append(img_pil.crop((w-s2,0,w,s2)))
    crops.append(img_pil.crop((0,h-s2,s2,h)))
    crops.append(img_pil.crop((w-s2,h-s2,w,h)))
    views = crops + [c.transpose(Image.FLIP_LEFT_RIGHT) for c in crops]
    return views

@torch.inference_mode()
def tta2_predict(mn, rn, img_pil, Tmn_vec, Trn_vec):
    views = _tta2_views(img_pil)
    out_mn = infer_batch(mn, views, Tmn_vec, return_logits=False)
    out_rn = infer_batch(rn, views, Trn_vec, return_logits=False)
    pm = out_mn["probs"].mean(dim=0); pr = out_rn["probs"].mean(dim=0)
    def _cm(pv):
        top2 = pv.topk(2); return float(top2.values[0]), float(top2.values[0]-top2.values[1]), int(top2.indices[0])
    cm, mm, km = _cm(pm); cr, mr, kr = _cm(pr)
    return pm, pr, cm, cr, mm, mr, km, kr

@torch.inference_mode()
def tta_quick_predict(mn, rn, img_pil, Tmn_vec, Trn_vec):
    v1 = img_pil
    v2 = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    out_mn = infer_batch(mn, [v1, v2], Tmn_vec, return_logits=False)
    out_rn = infer_batch(rn, [v1, v2], Trn_vec, return_logits=False)
    pm = out_mn["probs"].mean(dim=0); pr = out_rn["probs"].mean(dim=0)
    tm = pm.topk(2); tr = pr.topk(2)
    cm = float(tm.values[0]); mm = float(tm.values[0] - tm.values[1])
    cr = float(tr.values[0]); mr = float(tr.values[0] - tr.values[1])
    return pm, pr, cm, cr, mm, mr

# ===================== CLASS HELPERS =====================
def is_rice_label(lbl: str) -> bool:
    return lbl.startswith("Rice___")

# ===================== MODEL WRAPPER =====================
class LeafEnsemble:
    def __init__(self):
        self.device = DEVICE
        # classes
        self.classes = _load_classes_from_class_to_idx(CLASS_TO_IDX_JSON)
        self.num_classes = len(self.classes)

        # models
        self.mn = torchvision.models.mobilenet_v2(num_classes=self.num_classes)
        self.rn = torchvision.models.resnet50(num_classes=self.num_classes)
        self.mn.load_state_dict(torch.load(str(CKPT_MN), map_location="cpu"))
        self.rn.load_state_dict(torch.load(str(CKPT_RN), map_location="cpu"))
        self.mn.to(self.device).eval(); self.rn.to(self.device).eval()

        self.rn_rice = None
        if CKPT_RN_RICE_EXPERT.is_file():
            self.rn_rice = torchvision.models.resnet50(num_classes=self.num_classes)
            state = torch.load(str(CKPT_RN_RICE_EXPERT), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
            self.rn_rice.load_state_dict(state)
            self.rn_rice.to(self.device).eval()

        # T vec
        try:
            if TEMP_CLASSWISE_JSON.is_file():
                Tmn = robust_load_T_classwise(TEMP_CLASSWISE_JSON, self.classes, "mn")
                Trn = robust_load_T_classwise(TEMP_CLASSWISE_JSON, self.classes, "rn")
                self.Tmn_vec = torch.tensor(Tmn, dtype=torch.float32, device=self.device)
                self.Trn_vec = torch.tensor(Trn, dtype=torch.float32, device=self.device)
            else:
                smn, srn = load_scalar_T(TEMP_SCALAR_JSON)
                self.Tmn_vec = torch.full((self.num_classes,), max(smn, T_FLOOR), device=self.device)
                self.Trn_vec = torch.full((self.num_classes,), max(srn, T_FLOOR), device=self.device)
        except Exception:
            smn, srn = load_scalar_T(TEMP_SCALAR_JSON)
            self.Tmn_vec = torch.full((self.num_classes,), max(smn, T_FLOOR), device=self.device)
            self.Trn_vec = torch.full((self.num_classes,), max(srn, T_FLOOR), device=self.device)

        if not hasattr(self, "_class_guard"):
            self._class_guard = None

    @torch.inference_mode()
    def predict_one(self, im: Image.Image) -> Dict:
        """
        코랩 규칙을 단일 이미지 서빙에 맞게 적용.
        반환 포맷은 backend/services/classifier.py 가 기대하는 구조를 따름.
        """
        # ---------- 1) 기본 추론 ----------
        out_mn = infer_batch(self.mn, [im], self.Tmn_vec)
        out_rn = infer_batch(self.rn, [im], self.Trn_vec)
        pm0, pr0 = out_mn["probs"][0], out_rn["probs"][0]
        cm0, cr0 = float(out_mn["conf"][0]), float(out_rn["conf"][0])
        mm0, mr0 = float(out_mn["margin"][0]), float(out_rn["margin"][0])
        km0, kr0 = int(out_mn["idx"][0]), int(out_rn["idx"][0])
        lbl_mn0, lbl_rn0 = self.classes[km0], self.classes[kr0]

        # ---------- 2) 이미지 메트릭 ----------
        (leaf_area, exg_mean, gy, bbox, exg_mean_box,
         red_frac_box, mask_largest, edge_den_box, lab_a, lab_b, aspect) = leaf_metrics(im)
        sat = saturation_ratio(im); hi = highlight_ratio(im)
        p_avg0 = 0.5*(pm0+pr0)

        # water veto
        water_frac = water_like_ratio(im, h_lo=RICE["water_veto_h_lo"], h_hi=RICE["water_veto_h_hi"], s_lo=RICE["water_veto_s_lo"])

        # leaf 판단
        is_leaf = (leaf_area >= LEAF_GATE["area_min"]) and (exg_mean >= LEAF_GATE["exg_min"])
        rn_strong0 = (cr0 >= RN_PREF["conf_min"]) and (mr0 >= RN_PREF["margin_min"])
        red_dominant = (red_frac_box > 0.30 and exg_mean_box < -0.10)

        # RN-rescue (edge/green/sat 최소 보증 + red-dominant 차단)
        if (
            (not is_leaf) and rn_strong0 and (gy >= LEAF_GATE["rescue_gy_min"]) and
            (sat >= LEAF_GATE["rescue_sat_min"]) and (leaf_area >= 0.02) and
            (edge_den_box >= 0.020) and (exg_mean_box >= -0.05) and (not red_dominant)
        ):
            is_leaf = True

        # Necrosis rescue
        if NECROSIS["use"] and (not is_leaf):
            in_white = (lbl_rn0 in NECROSIS["class_whitelist"])
            low_chroma = (sat <= NECROSIS["sat_max"]) and (gy <= NECROSIS["gy_max"])
            rn_ok = (cr0 >= NECROSIS["rn_conf_min"]) and (mr0 >= NECROSIS["rn_margin_min"])
            edge_ok = (edge_den_box >= NECROSIS["edge_tau"])
            lab_ok = (NECROSIS["lab_a_min"] <= lab_a <= NECROSIS["lab_a_max"]) and (NECROSIS["lab_b_min"] <= lab_b <= NECROSIS["lab_b_max"])
            if in_white and low_chroma and rn_ok and (edge_ok or lab_ok):
                is_leaf = True

        # Food veto
        food_veto = (red_frac_box > 0.30 and exg_mean_box < -0.15)

        # Strawberry OOD 가드(예외 허용 조건)
        if lbl_rn0 == "Strawberry___Leaf_scorch":
            leafish = (leaf_area >= 0.12) and (gy >= 0.15) and (edge_den_box >= 0.02) and (exg_mean_box >= -0.05)
            rn_overrule = (cr0 >= 0.93 and mr0 >= 0.45) and leafish and (not red_dominant)
            if (not rn_overrule) and (not is_leaf or (leaf_area < 0.15 or gy < 0.15)):
                return self._pack_unknown(im_path=None, reason="Guard[Strawberry_OOD_Veto]",
                                          raw=locals(), out_mn=out_mn, out_rn=out_rn)

        # Powdery-mildew 전용 구제
        powdery_classes = {"Cherry___Powdery_mildew"}
        powdery_candidate = ((lbl_rn0 in powdery_classes) and (cr0 >= 0.90) and (mr0 >= 0.30) and
                             (edge_den_box >= 0.020) and (aspect <= 6.0))
        if (not is_leaf) and powdery_candidate:
            is_leaf = True

        # healthy 소형 잎 구제
        healthy_labels = {
            "Peach___healthy", "Apple___healthy", "Grape___healthy",
            "Soybean___healthy", "Blueberry___healthy", "Pepper,_bell___healthy",
        }
        healthy_smallleaf_rescue = (
            ((lbl_rn0 in healthy_labels) or (lbl_mn0 in healthy_labels)) and
            (cr0 >= 0.96 and mr0 >= 0.40) and
            (leaf_area >= 0.015) and (gy >= 0.06) and (sat >= 0.12) and
            (edge_den_box >= 0.020) and (exg_mean_box >= -0.02)
        )
        if (not is_leaf) and healthy_smallleaf_rescue:
            is_leaf = True

        # very low leaf veto
        no_leaf_veto = (leaf_area < 0.05) or (gy < 0.05) or (sat < 0.10)
        if not is_leaf and no_leaf_veto:
            return self._pack_unknown(im_path=None, reason="Global[NoLeafVeto]",
                                      raw=locals(), out_mn=out_mn, out_rn=out_rn)

        # water veto for rice overconfidence
        pe0 = ensemble_probs(pm0, pr0)
        rice_suspect = (is_rice_label(lbl_rn0) or self._rice_in_topk(pm0, k=3) or self._rice_in_topk(pe0, k=3))
        if water_frac >= RICE["water_veto_frac"] and rice_suspect:
            return self._pack_unknown(im_path=None, reason=f"Global[WaterVeto wf={water_frac:.2f}]",
                                      raw=locals(), out_mn=out_mn, out_rn=out_rn)

        # Rice 전문가 하이브리드 트리거
        trigger_rice_expert = False
        if self.rn_rice is not None:
            trigger_rice_expert = (
                is_leaf and (leaf_area >= RICE["gate_leaf_min"]) and (gy >= RICE["gate_gy_min"]) and
                (entropy(p_avg0) <= RICE["gate_entropy_max"]) and
                ( is_rice_label(lbl_rn0) or self._rice_in_topk(pe0, k=2) ) and
                self._rice_in_topk(pm0, k=3) and
                (water_frac < RICE["water_veto_frac"])
            )
            alt_rice_trigger = (
                is_leaf and (leaf_area >= RICE["gate_alt_leaf_min"]) and (gy >= RICE["gate_gy_min"]) and
                (aspect >= RICE["gate_alt_aspect_min"]) and (mr0 <= RICE["gate_alt_mr_max"]) and (cr0 <= RICE["gate_alt_cr_max"]) and
                (entropy(p_avg0) <= RICE["gate_entropy_max"]) and
                ( is_rice_label(lbl_rn0) or self._rice_in_topk(pe0, k=3) ) and
                self._rice_in_topk(pm0, k=3) and (water_frac < RICE["water_veto_frac"])
            )
            trigger_rice_expert = trigger_rice_expert or alt_rice_trigger

        # 전문가 적용(부분치환 대신 안전한 soft blend)
        pr_used = pr0.clone()
        if trigger_rice_expert:
            pmE, prE, cmE, crE, mmE, mrE, kmE, krE = tta2_predict(self.mn, self.rn_rice, im, self.Tmn_vec, self.Trn_vec)
            alpha = RICE["blend_alpha"]
            pr_used = (1.0 - alpha) * pr0 + alpha * prE
            pr_used = pr_used / pr_used.sum()

        # RN 지표 갱신
        pr = pr_used
        v2, i2 = pr.topk(2)
        cr = float(v2[0]); mr = float(v2[0]-v2[1]); kr = int(i2[0])
        lbl_rn = self.classes[kr]

        # Orange HLB grass-like veto v3
        if lbl_rn == "Orange___Haunglongbing_(Citrus_greening)":
            grass_like_v3 = (sat >= 0.65) and (0.12 <= gy <= 0.24) and (leaf_area <= 0.15)
            if grass_like_v3:
                return self._pack_unknown(im_path=None, reason="Guard[Orange_Grass_Veto_v3]",
                                          raw=locals(), out_mn=out_mn, out_rn=out_rn, pr_used=pr)

        # RN leaf-override
        if is_leaf and (leaf_area >= OVERRIDE["leaf_big_rn_mid"]["leaf_min"]) and (cr >= OVERRIDE["leaf_big_rn_mid"]["rn_min"]) and (mr >= OVERRIDE["leaf_big_rn_mid"]["rn_margin_min"]):
            final_lbl, final_conf = lbl_rn, cr
            return self._pack_final(final_lbl, final_conf, im, out_mn, out_rn, pm0, pr, reason="RN_leaf_override",
                                    extra=dict(leaf=leaf_area, gy=gy, sat=sat, hi=hi, cm=cm0, cr=cr, mm=mm0, mr=mr))

        # ---------- 3) 글로벌 엔트로피 가드 + TTA ----------
        H_th = 1.50  # 서빙 기본값
        H_th_eff = H_th
        if is_leaf and rn_strong0:
            H_th_eff = max(H_th_eff, ENTROPY["relax_when_leaf_rn_strong"])
        if is_leaf and (leaf_area >= OVERRIDE["leaf_big_rn_mid"]["leaf_min"]):
            H_th_eff = max(H_th_eff, ENTROPY["relax_when_bigleaf_rn_mid"])

        # per-class relax
        label_for_relax = lbl_rn
        if label_for_relax in RELAX_RULES:
            rule = RELAX_RULES[label_for_relax]
            ok = True
            if rule.get("require_leaf", False): ok &= bool(is_leaf)
            if "rn_conf_min" in rule: ok &= bool(cr >= rule["rn_conf_min"])
            if "rn_margin_min" in rule: ok &= bool(mr >= rule["rn_margin_min"])
            if "leaf_min" in rule: ok &= bool(leaf_area >= rule["leaf_min"])
            if "gy_min" in rule:   ok &= bool(gy        >= rule["gy_min"])
            if "hi_max" in rule:   ok &= bool(hi        <= rule["hi_max"])
            if ok and (label_for_relax in CLASS_ENTROPY_RELAX):
                H_th_eff = max(H_th_eff, CLASS_ENTROPY_RELAX[label_for_relax])

        p_avg_used = 0.5*(pm0 + pr)
        H_cur = entropy(p_avg_used)
        if H_cur > H_th_eff:
            if H_cur < 1.20 * H_th_eff:
                pmQ, prQ, cmQ, crQ, mmQ, mrQ = tta_quick_predict(self.mn, self.rn, im, self.Tmn_vec, self.Trn_vec)
                if entropy(0.5*(pmQ+prQ)) <= H_th_eff:
                    pm0, pr = pmQ, prQ
                    cm0, cr, mm0, mr = cmQ, crQ, mmQ, mrQ
                    lbl_mn0 = self.classes[int(pm0.argmax())]
                    lbl_rn  = self.classes[int(pr.argmax())]
                else:
                    pm2, pr2, cm2, cr2, mm2, mr2, _, _ = tta2_predict(self.mn, self.rn, im, self.Tmn_vec, self.Trn_vec)
                    if entropy(0.5*(pm2+pr2)) <= H_th_eff:
                        pm0, pr = pm2, pr2
                        cm0, cr, mm0, mr = cm2, cr2, mm2, mr2
                        lbl_mn0 = self.classes[int(pm0.argmax())]
                        lbl_rn  = self.classes[int(pr.argmax())]
                    else:
                        return self._pack_unknown(im_path=None, reason=f"HighEntropy({H_cur:.2f}>{H_th_eff:.2f})",
                                                  raw=locals(), out_mn=out_mn, out_rn=out_rn, pr_used=pr)

        # ---------- 4) CONSENSUS / RNHIGH ----------
        same_class0 = (int(pm0.argmax()) == int(pr.argmax()))
        mean_conf0 = 0.5*(cm0 + cr)
        min_margin0 = min(mm0, mr)
        if is_leaf and same_class0 and (mean_conf0 >= CONSENSUS["mean_conf_min"]) and (min_margin0 >= CONSENSUS["min_margin_min"]) \
           and (entropy(0.5*(pm0+pr)) <= CONSENSUS["entropy_cap"]) and (gy >= CONSENSUS["gy_min"]) \
           and (edge_den_box >= CONSENSUS["edge_min"]) and (aspect <= CONSENSUS["max_aspect"]):
            final_lbl, final_conf = self.classes[int(pm0.argmax())], mean_conf0
            return self._pack_final(final_lbl, final_conf, im, out_mn, out_rn, pm0, pr, reason="CONSENSUS",
                                    extra=dict(leaf=leaf_area, gy=gy, sat=sat, hi=hi, cm=cm0, cr=cr, mm=mm0, mr=mr))

        if (not is_rice_label(lbl_rn)) and is_leaf and (leaf_area >= RNHIGH["leaf_min"]) and (gy >= RNHIGH["gy_min"]) and \
           (sat >= RNHIGH["sat_min"]) and (edge_den_box >= RNHIGH["edge_min"]):
            rn_abs = (cr >= RNHIGH["cr_abs"]) and (mr >= RNHIGH["mr_abs"])
            if (same_class0 or (RNHIGH["same_or_abs"] and rn_abs)) and (entropy(0.5*(pm0+pr)) <= RNHIGH["entropy_cap"]):
                final_lbl, final_conf = lbl_rn, cr
                return self._pack_final(final_lbl, final_conf, im, out_mn, out_rn, pm0, pr, reason="RNHIGH",
                                        extra=dict(leaf=leaf_area, gy=gy, sat=sat, hi=hi, cm=cm0, cr=cr, mm=mm0, mr=mr))

        # Rice 최종 안전장치
        if is_rice_label(lbl_rn):
            if not ((cr >= RICE["accept_conf"]) and (mr >= RICE["accept_margin"]) and
                    (edge_den_box >= RICE["veto_edge_min"]) and (aspect <= RICE["max_aspect"]) and (not food_veto)):
                # 그냥 계속 진행(클래스 가드에서 걸릴 수 있음)
                pass

        # MN 우세 보정
        pe_base = ensemble_probs(pm0, pr)
        top_idx = int(torch.argmax(pe_base))
        if is_leaf and is_rice_label(self.classes[top_idx]) and (not is_rice_label(lbl_mn0)) and \
           (cm0 >= cr + 0.15) and (mm0 >= 0.25) and (entropy(0.5*(pm0+pr)) >= 1.60):
            pe = pm0*0.60 + pr*0.40; pe = pe/pe.sum()
        else:
            pe = pe_base

        # ---------- 5) 클래스 가드 ----------
        ke = int(torch.argmax(pe)); pred_lbl = self.classes[ke]; conf_e = float(pe[ke])

        # 가드에 올바른 파라미터 전달
        is_unknown, reason, info = self._class_guard(
            mn_label=lbl_mn0, mn_conf=cm0, 
            rn_label=lbl_rn, rn_conf=cr, 
            ens_label=pred_lbl, ens_conf=conf_e,
            picked_model="Ensemble", picked_label=pred_lbl, picked_conf=conf_e
        )

        deny = (lbl_rn in OVERRIDE.get("guard_override_deny", []))
        cfg  = GUARD_CFG.get(lbl_rn, {})
        hard_no_leaf_hit = cfg.get("require_leaf", False) and cfg.get("hard_no_leaf", False) and \
            ((leaf_area < cfg.get("hard_leaf_min", 0.0)) or (gy < cfg.get("hard_gy_min", 0.0)))
        min_leaf_ok = (leaf_area >= OVERRIDE["guard_override"].get("leaf_min_override", 0.10))
        min_gy_ok   = (gy        >= OVERRIDE["guard_override"].get("gy_min_override",   0.08))
        red_block   = (red_frac_box > 0.25 and exg_mean_box < -0.10)

        if is_unknown and is_leaf and \
           (cr >= OVERRIDE["guard_override"]["rn_min"]) and (mr >= OVERRIDE["guard_override"]["rn_margin_min"]) and \
           (not deny) and min_leaf_ok and min_gy_ok and (not hard_no_leaf_hit) and (not red_block):
            final_lbl, final_conf = lbl_rn, cr
            return self._pack_final(final_lbl, final_conf, im, out_mn, out_rn, pm0, pr, reason="GuardOverride",
                                    extra=dict(leaf=leaf_area, gy=gy, sat=sat, hi=hi, cm=cm0, cr=cr, mm=mm0, mr=mr))

        if is_unknown:
            return self._pack_unknown(im_path=None, reason=reason, raw=locals(), out_mn=out_mn, out_rn=out_rn, pr_used=pr)

        # 최종
        final_lbl, final_conf = pred_lbl, conf_e
        return self._pack_final(final_lbl, final_conf, im, out_mn, out_rn, pm0, pr, reason=None,
                                extra=dict(leaf=leaf_area, gy=gy, sat=sat, hi=hi, cm=cm0, cr=cr, mm=mm0, mr=mr))

    # --------- helpers for serving packs ----------
    def _pack_final(self, label, conf, im, out_mn, out_rn, pm, pr, reason=None, extra=None):
        p_mn = out_mn["probs"][0]; p_rn = out_rn["probs"][0]
        return {
            "mobilenet": {"label": self.classes[int(p_mn.argmax())], "confidence": float(p_mn.max())},
            "resnet50":  {"label": self.classes[int(p_rn.argmax())], "confidence": float(p_rn.max())},
            "ensemble":  {"label": label, "confidence": conf, "weights": {"mn": ENSEMBLE["w_mn"], "rn": ENSEMBLE["w_rn"]}},
            "picked":    {"model": "Ensemble", "label": label, "confidence": conf},
            "meta": {
                "entropy": {
                    "mobilenet": entropy(p_mn),
                    "resnet50":  entropy(p_rn),
                    "ensemble":  entropy(ensemble_probs(pm, pr)),
                },
                "inference_ms": (out_mn["time"] + out_rn["time"]) * 1000.0,
                "reason": reason,
                "signals": extra or {}
            }
        }

    def _pack_unknown(self, im_path, reason, raw, out_mn, out_rn, pr_used=None):
        p_mn = out_mn["probs"][0]; p_rn = out_rn["probs"][0]
        return {
            "mobilenet": {"label": self.classes[int(p_mn.argmax())], "confidence": float(p_mn.max())},
            "resnet50":  {"label": self.classes[int(p_rn.argmax())], "confidence": float(p_rn.max())},
            "ensemble":  {"label": "Unknown", "confidence": 0.0, "weights": {"mn": ENSEMBLE["w_mn"], "rn": ENSEMBLE["w_rn"]}},
            "picked":    {"model": "Unknown", "label": "Unknown", "confidence": 0.0},
            "meta": {
                "entropy": {
                    "mobilenet": entropy(p_mn),
                    "resnet50":  entropy(p_rn),
                    "ensemble":  entropy(ensemble_probs(p_mn, pr_used if pr_used is not None else p_rn)),
                },
                "inference_ms": (out_mn["time"] + out_rn["time"]) * 1000.0,
                "reason": reason,
                "signals": {
                    "leaf": raw.get("leaf_area", None),
                    "gy": raw.get("gy", None),
                    "sat": raw.get("sat", None),
                    "hi": raw.get("hi", None),
                }
            }
        }

    def _rice_in_topk(self, prob_vec: torch.Tensor, k=3) -> bool:
        v, idx = prob_vec.topk(min(k, prob_vec.numel()))
        for ii in idx.tolist():
            if is_rice_label(self.classes[int(ii)]):
                return True
        return False

# ============== FACTORY (singleton) ==============
_model_singleton: Optional[LeafEnsemble] = None
def get_model() -> LeafEnsemble:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = LeafEnsemble()
    return _model_singleton
