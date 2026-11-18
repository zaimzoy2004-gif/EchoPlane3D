"""Hybrid offline tutor for EchoPlane3D.

Two engines answer a guide turn:

1. **Deterministic spatial tutor** (always available, zero extra RAM/VRAM): a
   keyword / fuzzy intent matcher over a small clinical knowledge base.  Every
   number it states is computed by ``simpsons_math`` from the current study.
2. **Optional local LLM through Ollama** (``http://localhost:11434/api/chat``):
   used only for the wording of an explanation.  Its output must be a JSON
   object whose actions pass a Pydantic whitelist; anything else — malformed
   JSON, an unknown action, a time-out after 4 s, Ollama not running — falls
   back to the deterministic reply instantly.

Separation of concerns: quiz questions, grading, UI actions decided by a
recognised intent, and every calculation always come from the deterministic
engine.  LLM text is labelled as an explanation and travels next to (never
instead of) the ``verified`` calculation block.

No cloud API is ever contacted.  EchoPlane3D is an educational and research
prototype, not a medical device.
"""
from __future__ import annotations

import difflib
import json
import re
import time
from typing import Any, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field, ValidationError

from . import config
from .simpsons_math import simulate_apical_foreshortening

DISCLAIMER = "Educational prototype — not a medical device and not a diagnosis."
STRUCTURE_NAMES = ("LV", "IVS", "RV", "LA", "RA", "Aorta", "PA", "SVC", "IVC")
OVERLAYS = ("mask", "axis", "chords", "fan", "cutaway", "labels", "beat")

# =========================================================================== #
# Knowledge base
# =========================================================================== #
VIEWS: dict[str, dict[str, Any]] = {
    "A4C": {
        "label": "Apical 4-Chamber (A4C)",
        "keywords": [r"\ba4c\b", r"\b4c\b", r"four[\s-]*chamber", r"4[\s-]*chamber", r"apical\s*4"],
        "probe": "Probe on the apex (5th intercostal space, mid-clavicular line), index marker toward the "
                 "patient's left (~3 o'clock), beam aimed at the right shoulder.",
        "see": "All four chambers. LV on screen right, RV on screen left, atria at the bottom; the crux of the "
               "heart (septum meeting both AV valves) sits in the centre.",
        "landmarks": "LV apex at the top of the sector, interventricular septum, mitral and tricuspid valves "
                     "(the tricuspid inserts slightly more apically), interatrial septum.",
        "pitfall": "Most prone to foreshortening: if the LV looks short and round with a thick, 'beating' apex, "
                   "you are cutting through the anterior wall — slide one rib-space lower / more lateral.",
        "highlight": ["LV", "IVS", "RV", "LA", "RA"],
    },
    "A2C": {
        "label": "Apical 2-Chamber (A2C)",
        "keywords": [r"\ba2c\b", r"\b2c\b", r"two[\s-]*chamber", r"2[\s-]*chamber", r"apical\s*2"],
        "probe": "From A4C keep the apex centred and rotate the probe ~60° counter-clockwise (index marker "
                 "toward ~12 o'clock) until the right heart disappears.",
        "see": "Only the LV and LA. Anterior wall on screen right, inferior wall on screen left.",
        "landmarks": "Mitral valve between LV and LA, LA appendage (sometimes) on the right, coronary sinus in "
                     "the inferior AV groove.",
        "pitfall": "If any RV or aortic outflow shows up you have under- or over-rotated. Biplane Simpson needs "
                   "A2C and A4C long-axis lengths to agree within 10 %.",
        "highlight": ["LV", "LA"],
    },
    "PLAX": {
        "label": "Parasternal Long-Axis (PLAX)",
        "keywords": [r"\bplax\b", r"parasternal\s*long", r"long[\s-]*axis\s+view", r"long[\s-]*axis$"],
        "probe": "Left sternal border, 3rd–4th intercostal space, index marker toward the right shoulder "
                 "(~10 o'clock); patient in left lateral decubitus.",
        "see": "RV outflow tract nearest the probe, then septum, LV cavity, aortic root and valve on screen "
               "right, with the LA behind the aorta. The apex is off-screen to the left.",
        "landmarks": "Aortic–mitral continuity, anterior mitral leaflet, posterior LV wall, descending aorta "
                     "behind the LA (pericardial vs pleural effusion landmark).",
        "pitfall": "LV volumes are NOT measured here — the true apex is never in the PLAX plane. Use it for "
                   "linear dimensions (LVIDd, IVSd, LVOT diameter).",
        "highlight": ["LV", "IVS", "RV", "LA", "Aorta"],
    },
    "PSAX": {
        "label": "Parasternal Short-Axis (PSAX, mid-papillary)",
        "keywords": [r"\bpsax\b", r"\bsax\b", r"short[\s-]*axis"],
        "probe": "From PLAX rotate 90° clockwise (index marker toward the left shoulder, ~2 o'clock) and tilt "
                 "slightly toward the apex until both papillary muscles appear.",
        "see": "The LV as a circle ('doughnut') with the septum and the crescent-shaped RV wrapped around it at "
               "the top-left of the screen.",
        "landmarks": "Anterolateral and posteromedial papillary muscles, all six mid-ventricular wall segments, "
                     "interventricular septum.",
        "pitfall": "An oval LV means the cut is oblique — correct the tilt until it is circular. A D-shaped "
                   "septum signals RV pressure or volume overload.",
        "highlight": ["LV", "IVS", "RV"],
    },
}

ANATOMY: dict[str, dict[str, Any]] = {
    "LV": {
        "keywords": [r"\blv\b", r"left\s+ventric"],
        "text": "**Left ventricle (LV).** Thick-walled, bullet-shaped systemic pump. Blood enters through the "
                "mitral valve and leaves through the LV outflow tract and aortic valve. Its volumes (EDV, ESV) "
                "and ejection fraction are what Simpson's method measures from the apical views.",
        "view": "A4C",
    },
    "RV": {
        "keywords": [r"\brv\b", r"right\s+ventric", r"\brvot\b", r"infundibulum"],
        "text": "**Right ventricle (RV).** Thin-walled and crescent-shaped in cross-section, wrapped around the "
                "septum; it is the most anterior chamber. Its outflow portion (infundibulum / RVOT) leads to the "
                "pulmonary valve. In A4C it lies on screen left; in PLAX the RVOT is the structure nearest the probe.",
        "view": "A4C",
    },
    "IVS": {
        "keywords": [r"\bivs\b", r"septum", r"septal"],
        "text": "**Interventricular septum (IVS).** The muscular wall shared by both ventricles. It normally "
                "bows toward the RV and contracts with the LV. Flattening into a 'D' shape in short axis suggests "
                "RV pressure or volume overload. Its thickness (IVSd) is measured in PLAX.",
        "view": "PSAX",
    },
    "LA": {
        "keywords": [r"\bla\b", r"left\s+atri", r"\blaa\b", r"appendage"],
        "text": "**Left atrium (LA).** Posterior, superior chamber receiving the pulmonary veins. The left atrial "
                "appendage (LAA) is a finger-like pouch on its anterolateral side — the commonest site of thrombus "
                "in atrial fibrillation; it is sometimes seen at the edge of the A2C view. In PLAX the LA lies "
                "behind the aortic root and is about the same size.",
        "view": "A2C",
    },
    "RA": {
        "keywords": [r"\bra\b", r"right\s+atri"],
        "text": "**Right atrium (RA).** Receives systemic venous blood from the SVC, the IVC and the coronary "
                "sinus. In A4C it sits below the RV on screen left, separated from the LA by the interatrial septum.",
        "view": "A4C",
    },
    "Aorta": {
        "keywords": [r"aort", r"\bao\b", r"\blvot\b"],
        "text": "**Aorta.** The aortic root arises from the LV outflow tract, in fibrous continuity with the "
                "anterior mitral leaflet. It continues as the ascending aorta, the arch (giving the "
                "brachiocephalic, left common carotid and left subclavian arteries) and the descending thoracic "
                "aorta, which is seen in cross-section behind the LA in PLAX.",
        "view": "PLAX",
    },
    "PA": {
        "keywords": [r"pulmonary\s+(artery|trunk)", r"\bpa\b", r"pulmonic"],
        "text": "**Pulmonary artery (PA).** The main trunk leaves the RV outflow tract in front of and to the left "
                "of the aortic root, then divides under the aortic arch into right and left pulmonary arteries. "
                "It is best seen in the basal short-axis and RV outflow views, not in the four views modelled here.",
        "view": None,
    },
    "SVC": {
        "keywords": [r"\bsvc\b", r"superior\s+vena"],
        "text": "**Superior vena cava (SVC).** Drains the head, neck and arms into the top of the right atrium, "
                "running vertically to the right of the ascending aorta. Seen in subcostal and suprasternal views.",
        "view": None,
    },
    "IVC": {
        "keywords": [r"\bivc\b", r"inferior\s+vena"],
        "text": "**Inferior vena cava (IVC).** Enters the floor of the right atrium from below. Its diameter and "
                "inspiratory collapse in the subcostal view are used to estimate right-atrial pressure.",
        "view": None,
    },
}

# Landmark quiz bank — fixed order, fixed answers.
QUIZ: dict[str, list[dict[str, Any]]] = {
    "A4C": [
        {
            "question": "In a standard A4C image, which chamber sits at the top-right of the screen, closest to the probe?",
            "options": ["Right atrium", "Left ventricle", "Left atrium", "Right ventricle"],
            "answer": 1,
            "explanation": "The probe sits on the apex, so the ventricles are at the top; by convention the LV is on screen right.",
            "highlight": ["LV"],
        },
        {
            "question": "Which AV valve inserts slightly closer to the apex — a landmark for telling the ventricles apart?",
            "options": ["Mitral valve", "Aortic valve", "Tricuspid valve", "Pulmonary valve"],
            "answer": 2,
            "explanation": "The septal tricuspid leaflet inserts ~5–10 mm more apically than the mitral valve; the ventricle attached to it is the RV.",
            "highlight": ["RV", "RA"],
        },
        {
            "question": "The LV looks short and round and the 'apex' thickens vigorously in systole. What happened?",
            "options": ["Normal variant", "Apical foreshortening", "Too much gain", "Pericardial effusion"],
            "answer": 1,
            "explanation": "The plane misses the true apex and cuts the anterior wall obliquely. Volumes are under-estimated and LVEF tends to be over-estimated.",
            "highlight": ["LV"],
        },
        {
            "question": "Which structure separates the two ventricles in the centre of the A4C image?",
            "options": ["Interatrial septum", "Interventricular septum", "Moderator band", "Mitral annulus"],
            "answer": 1,
            "explanation": "The interventricular septum runs from the apex to the crux of the heart between LV and RV.",
            "highlight": ["IVS"],
        },
    ],
    "A2C": [
        {
            "question": "Which two LV walls are seen in the apical 2-chamber view?",
            "options": ["Septal and lateral", "Anterior and inferior", "Anteroseptal and inferolateral", "Inferior and lateral"],
            "answer": 1,
            "explanation": "A2C shows the anterior wall (screen right) and the inferior wall (screen left).",
            "highlight": ["LV"],
        },
        {
            "question": "Starting from A4C, how do you obtain A2C?",
            "options": ["Slide one rib-space up", "Tilt the tail down", "Rotate ~60° counter-clockwise", "Rotate 90° clockwise"],
            "answer": 2,
            "explanation": "Keep the apex centred and rotate ~60° counter-clockwise until the right heart disappears.",
            "highlight": ["LV", "LA"],
        },
        {
            "question": "Which chambers must NOT be visible in a correct A2C view?",
            "options": ["LV and LA", "RV and RA", "LA only", "LV only"],
            "answer": 1,
            "explanation": "A2C contains only the left heart. Any right-heart structure means the rotation is off.",
            "highlight": ["RV", "RA"],
        },
        {
            "question": "Biplane Simpson combines A4C with A2C. By how much may their long-axis lengths differ?",
            "options": ["Up to 10 %", "Up to 25 %", "Up to 40 %", "It does not matter"],
            "answer": 0,
            "explanation": "A difference above 10 % suggests that one of the two views is foreshortened.",
            "highlight": ["LV"],
        },
    ],
    "PLAX": [
        {
            "question": "In PLAX, which structure lies closest to the probe at the top of the image?",
            "options": ["Left atrium", "Aortic root", "Right ventricular outflow tract", "Descending aorta"],
            "answer": 2,
            "explanation": "The RV is the most anterior chamber, so the RVOT is the first thing the beam meets behind the sternum.",
            "highlight": ["RV"],
        },
        {
            "question": "Which chamber sits directly behind (posterior to) the aortic root in PLAX?",
            "options": ["Left atrium", "Right atrium", "Right ventricle", "Left ventricle"],
            "answer": 0,
            "explanation": "RVOT – aortic root – LA stack from anterior to posterior and are normally about the same size.",
            "highlight": ["LA", "Aorta"],
        },
        {
            "question": "Why is PLAX not used for Simpson's volumes?",
            "options": ["Depth is too shallow", "The true apex is not in the plane", "The mitral valve is hidden", "Frame rate is too low"],
            "answer": 1,
            "explanation": "The LV apex is off-screen in PLAX, so the long axis cannot be measured. Volumes come from apical views.",
            "highlight": ["LV"],
        },
        {
            "question": "A round vessel seen in cross-section behind the left atrium in PLAX is the…",
            "options": ["Pulmonary artery", "Coronary sinus", "Descending thoracic aorta", "Superior vena cava"],
            "answer": 2,
            "explanation": "The descending aorta lies posterior to the LA; fluid in front of it is pericardial, behind it pleural.",
            "highlight": ["Aorta"],
        },
    ],
    "PSAX": [
        {
            "question": "At the mid-papillary PSAX level, what shape should a normal LV have?",
            "options": ["Oval", "D-shaped", "Circular", "Triangular"],
            "answer": 2,
            "explanation": "A circular LV confirms the cut is truly perpendicular to the long axis; oval means oblique.",
            "highlight": ["LV"],
        },
        {
            "question": "Which chamber appears as a crescent wrapped around the septum?",
            "options": ["Left atrium", "Right ventricle", "Right atrium", "Aorta"],
            "answer": 1,
            "explanation": "The thin-walled RV wraps around the anterior and septal side of the LV.",
            "highlight": ["RV"],
        },
        {
            "question": "From PLAX, how do you reach PSAX?",
            "options": ["Rotate 90° clockwise", "Rotate 60° counter-clockwise", "Slide to the apex", "Tilt toward the right hip"],
            "answer": 0,
            "explanation": "Rotate 90° clockwise so the index marker points to the left shoulder, then tilt to choose the level.",
            "highlight": ["LV", "RV"],
        },
        {
            "question": "A flattened, D-shaped septum in short axis suggests overload of which chamber?",
            "options": ["Left atrium", "Left ventricle", "Right ventricle", "Aorta"],
            "answer": 2,
            "explanation": "RV pressure or volume overload pushes the septum toward the LV.",
            "highlight": ["IVS", "RV"],
        },
    ],
}

PRESETS = [
    {"id": "view_a4c", "label": "A4C · four chambers", "message": "Show me the apical 4-chamber view"},
    {"id": "view_a2c", "label": "A2C · two chambers", "message": "Show me the apical 2-chamber view"},
    {"id": "view_plax", "label": "PLAX", "message": "Show me the parasternal long-axis view"},
    {"id": "view_psax", "label": "PSAX", "message": "Show me the parasternal short-axis view"},
    {"id": "explain_foreshortening", "label": "What is foreshortening?", "message": "Explain apical foreshortening"},
    {"id": "demo_foreshortening", "label": "Demo a 20° tilt", "message": "Demonstrate foreshortening with a 20 degree tilt"},
    {"id": "fix_foreshortening", "label": "How do I fix it?", "message": "How do I fix foreshortening?"},
    {"id": "explain_simpson", "label": "Simpson's 20 disks", "message": "Explain Simpson's method of disks"},
    {"id": "biplane", "label": "Single-plane vs biplane", "message": "What is the difference between single-plane and biplane Simpson?"},
    {"id": "interpret", "label": "Interpret this LVEF", "message": "Interpret the current EDV, ESV and LVEF"},
    {"id": "anatomy_ivs", "label": "Anatomy: septum", "message": "Tell me about the interventricular septum"},
    {"id": "prompting", "label": "Mask looks wrong?", "message": "The segmentation mask looks wrong, what can I do?"},
    {"id": "quiz", "label": "Quiz me on landmarks", "message": "Quiz me on landmarks"},
]
_PRESET_BY_ID = {p["id"]: p for p in PRESETS}

WELCOME = (
    "I'm the EchoPlane3D spatial tutor. I can **switch the 3-D probe** to A4C, A2C, PLAX or PSAX, "
    "**explain apical foreshortening** with live numbers, walk through **Simpson's single-plane and biplane "
    "methods**, **interpret the measured EDV / ESV / LVEF**, describe any **chamber or vessel**, toggle overlays, "
    "load a sample, or **quiz you on landmarks**. Pick a preset or type a question."
)

# =========================================================================== #
# Action whitelist (shared by both engines)
# =========================================================================== #
ViewName = Literal["A4C", "A2C", "PLAX", "PSAX"]


class SetView(BaseModel):
    type: Literal["set_view"]
    view: ViewName


class SetTilt(BaseModel):
    type: Literal["set_tilt"]
    tilt_deg: float = Field(ge=0, le=30)


class ToggleOverlay(BaseModel):
    type: Literal["toggle_overlay"]
    overlay: Literal["mask", "axis", "chords", "fan", "cutaway", "labels", "beat"]
    on: bool


class StartQuiz(BaseModel):
    type: Literal["start_quiz"]
    view: Optional[ViewName] = None


class SelectSample(BaseModel):
    type: Literal["select_sample"]
    sample_id: str = Field(min_length=1, max_length=80)


class Highlight(BaseModel):
    type: Literal["highlight"]
    chambers: list[Literal["LV", "IVS", "RV", "LA", "RA", "Aorta", "PA", "SVC", "IVC"]] = Field(max_length=9)


class _ActionEnvelope(BaseModel):
    action: Union[SetView, SetTilt, ToggleOverlay, StartQuiz, SelectSample, Highlight] = Field(discriminator="type")


LLM_ALLOWED_ACTIONS = ("set_view", "set_tilt", "toggle_overlay", "start_quiz", "select_sample")


def validate_actions(raw: Any, sample_ids: Optional[set] = None, allowed: Optional[tuple] = None) -> tuple[list[dict], list[Any]]:
    """Return ``(valid, rejected)``.  An action survives only if it matches the
    whitelist schema exactly, is in ``allowed`` (if given) and — for
    ``select_sample`` — names a sample that exists."""
    valid, rejected = [], []
    if not isinstance(raw, list):
        return valid, [raw] if raw is not None else []
    for item in raw[:8]:
        try:
            act = _ActionEnvelope.model_validate({"action": item}).action
        except ValidationError:
            rejected.append(item)
            continue
        data = act.model_dump()
        if allowed is not None and data["type"] not in allowed:
            rejected.append(item)
        elif data["type"] == "select_sample" and (sample_ids is None or data["sample_id"] not in sample_ids):
            rejected.append(item)
        else:
            valid.append(data)
    return valid, rejected


# =========================================================================== #
# Deterministic engine
# =========================================================================== #
_RE_FORESHORTEN = re.compile(r"foreshort|fore-short|off[\s-]*axis|cut(ting)?\s+the\s+apex|true\s+apex|false\s+apex", re.I)
_RE_TILT = re.compile(r"\btilt", re.I)
_RE_FIX = re.compile(r"\bfix|correct|avoid|prevent|optimi[sz]e|improve|get rid", re.I)
_RE_DEMO = re.compile(r"\bdemo|demonstrate|simulate|show\s+me\s+what|\btry\b|\bset\b", re.I)
_RE_QUIZ = re.compile(r"\bquiz|test\s+me|landmark|ask\s+me|next\s+question|another\s+question", re.I)
_RE_BIPLANE = re.compile(r"biplane|bi-plane|single[\s-]*plane|two\s+planes|difference.*simpson", re.I)
_RE_SIMPSON = re.compile(r"simpson|\bdis[ck]s?\b|method\s+of\s+dis|how.*(volume|measure)", re.I)
_RE_INTERPRET = re.compile(r"interpret|what\s+does.*(mean|lvef|ef)|is\s+(this|that|it|my).*(normal|reduced|bad|ok)|"
                           r"\blvef\b|ejection\s+fraction|\bedv\b|\besv\b|stroke\s+volume|hfref|hfpef|heart\s+failure", re.I)
_RE_PROMPT = re.compile(r"mask|segment|click|prompt|snap|multi[\s-]*point|track(ing)?\b|wrong|leak|sam\s*2?", re.I)
_RE_PROBE = re.compile(r"index\s+marker|orientation\s+marker|notch|hold\s+the\s+probe|probe\s+(position|orientation)|"
                       r"which\s+way.*probe", re.I)
_RE_ABOUT = re.compile(r"who\s+are\s+you|are\s+you\s+(an?\s+)?(ai|llm|chatgpt|bot)|diagnos|medical\s+device|clinical\s+use|"
                       r"\bfda\b|\bce\b|offline|internet|privacy|can\s+i\s+trust", re.I)
_RE_HELP = re.compile(r"^\s*$|\bhelp\b|what\s+can\s+you|^\s*(hi|hello|hey)\b|\bmenu\b", re.I)
_RE_DEGREES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|deg|degree)", re.I)
_RE_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?)\b")
_RE_SHOW = re.compile(r"\b(show|display|turn\s+on|enable|unhide|bring\s+back)\b", re.I)
_RE_HIDE = re.compile(r"\b(hide|remove|turn\s+off|disable|without|no)\b", re.I)
_RE_LOAD = re.compile(r"\b(load|open|select|switch\s+to|use|go\s+to|next|previous|another)\b.*\b(sample|patient|phantom|clip|"
                      r"case|example|study|subject)\b|\bpatient\s*0*\d+\b|\b(hfref|normal)\s+phantom\b", re.I)
_OVERLAY_WORDS = {
    "mask": [r"\bmask\b", r"segmentation\s+overlay", r"green"],
    "axis": [r"long\s*axis\s+(line|overlay)", r"\baxis\s+(line|overlay)\b", r"yellow\s+line"],
    "chords": [r"\bchords?\b", r"\bdis[ck]s\s+overlay\b", r"simpson\s+(lines|chords|overlay)", r"cyan"],
    "fan": [r"\bfan\b", r"sector"],
    "cutaway": [r"cut[\s-]*away", r"\bclip(ping)?\b", r"cross[\s-]*section"],
    "labels": [r"\blabels?\b", r"\bnames\b"],
    "beat": [r"\bbeat(ing)?\b", r"contraction", r"animation"],
}
_VOCAB = [
    "foreshortening", "foreshortened", "simpson", "biplane", "ejection", "fraction", "ventricle", "atrium", "septum",
    "aorta", "pulmonary", "parasternal", "apical", "chamber", "quiz", "landmark", "tilt", "overlay", "mask",
    "segmentation", "interpret", "phantom", "patient", "volume", "appendage", "disks", "probe", "marker",
]


def _normalise(text: str) -> str:
    """Lower-case and repair obvious typos against the tutor's vocabulary."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:\.\d+)?|[^\sa-z0-9]", text.lower()):
        if tok.isalpha() and len(tok) >= 6 and tok not in _VOCAB:
            close = difflib.get_close_matches(tok, _VOCAB, n=1, cutoff=0.84)
            if close:
                tok = close[0]
        out.append(tok)
    return " ".join(out)


def _detect_view(text: str) -> Optional[str]:
    for key, v in VIEWS.items():
        if any(re.search(k, text, re.I) for k in v["keywords"]):
            return key
    return None


def _detect_structure(text: str) -> Optional[str]:
    for name, a in ANATOMY.items():
        if any(re.search(k, text, re.I) for k in a["keywords"]):
            return name
    return None


def _detect_overlay(text: str) -> Optional[str]:
    for name, pats in _OVERLAY_WORDS.items():
        if any(re.search(p, text, re.I) for p in pats):
            return name
    return None


def _view_reply(view: str) -> str:
    v = VIEWS[view]
    return (
        f"**{v['label']}**\n"
        f"**Probe:** {v['probe']}\n"
        f"**You see:** {v['see']}\n"
        f"**Landmarks:** {v['landmarks']}\n"
        f"**Pitfall:** {v['pitfall']}"
    )


def _volumes_from_context(ctx: dict) -> tuple[float, float, str, bool]:
    try:
        edv = float(ctx.get("edv_ml"))
        esv = float(ctx.get("esv_ml"))
        if edv > 0 and 0 < esv <= edv:
            return edv, esv, "the current study", True
    except (TypeError, ValueError):
        pass
    return 120.0, 48.0, "a reference ventricle (EDV 120 mL, ESV 48 mL)", False


def _foreshortening_numbers(ctx: dict, tilt: float) -> str:
    edv, esv, src, _ = _volumes_from_context(ctx)
    r = simulate_apical_foreshortening(edv, esv, tilt)
    t, f, e = r["true"], r["foreshortened"], r["errors"]
    return (
        f"For {src} at **{tilt:.0f}° tilt** (half-ellipsoid model): apparent LV length {t['ed_length_mm']:.0f} → "
        f"{f['ed_length_mm']:.0f} mm, EDV {t['edv_ml']:.0f} → {f['edv_ml']:.0f} mL ({e['edv_pct']:+.0f} %), "
        f"ESV {t['esv_ml']:.0f} → {f['esv_ml']:.0f} mL ({e['esv_pct']:+.0f} %), "
        f"LVEF {t['lvef_pct']:.1f} → {f['lvef_pct']:.1f} % ({e['lvef_abs_pts']:+.1f} points)."
    )


def lvef_band(lvef: float) -> str:
    """Heart-failure style bands (reduced <= 40 %, mildly reduced 41-49 %, preserved >= 50 %)."""
    if lvef <= 40:
        return "reduced (≤ 40 %)"
    if lvef < 50:
        return "mildly reduced (41–49 %)"
    if lvef < 55:
        return "preserved, low-normal (50–54 %)"
    return "preserved / normal (≥ 55 %)"


def _interpretation(ctx: dict) -> tuple[str, Optional[str]]:
    edv, esv, _, real = _volumes_from_context(ctx)
    if not real:
        return ("Segment a clip first (click the LV cavity or press **Auto-segment**) and I can interpret its "
                "EDV, ESV and LVEF.\n**Definitions:** EDV is the largest cavity volume in the cycle, ESV the "
                "smallest; stroke volume = EDV − ESV; LVEF = (EDV − ESV) / EDV × 100.\n"
                "**Bands used here:** reduced ≤ 40 %, mildly reduced 41–49 %, preserved ≥ 50 %."), None
    lvef = (edv - esv) / edv * 100.0
    unit = "mL" if ctx.get("calibrated", True) else "index units (uncalibrated clip)"
    verified = (f"EDV {edv:.1f} {unit}, ESV {esv:.1f} {unit}, stroke volume {edv - esv:.1f}, "
                f"LVEF = ({edv:.1f} − {esv:.1f}) / {edv:.1f} × 100 = **{lvef:.1f} %** → {lvef_band(lvef)}.")
    notes = []
    if ctx.get("segmented"):
        notes.append("These are **algorithmic estimates** from a SAM 2.1 mask and single-plane Simpson — compare "
                     "them with the reference rows in the *Estimates vs references* panel.")
    else:
        notes.append("These are the sample's **reference values**, not a measurement you made — segment the clip "
                     "to get an estimate.")
    if not ctx.get("calibrated", True):
        notes.append("The clip is uncalibrated, so volumes are a pixel-scale index; LVEF is a ratio and is unaffected.")
    bp = ctx.get("biplane_lvef")
    if isinstance(bp, (int, float)):
        notes.append(f"Biplane LVEF from both apical views is {bp:.1f} %.")
    try:
        if float(ctx.get("tilt_deg", 0)) > 0:
            notes.append("The tilt slider is not at 0° — the *Foreshortened* card shows what an off-axis plane would report.")
    except (TypeError, ValueError):
        pass
    reply = ("**Reading EDV, ESV and LVEF.** A single-plane apical estimate assumes the ventricle is symmetric "
             "about its long axis, depends entirely on where the endocardial border is traced, and is biased by "
             "foreshortening. " + " ".join(notes) + f"\n{DISCLAIMER}")
    return reply, verified


def _quiz_payload(view: str, index: int) -> dict:
    bank = QUIZ[view]
    i = index % len(bank)
    q = bank[i]
    return {"id": f"{view}:{i}", "view": view, "number": i + 1, "total": len(bank),
            "question": q["question"], "options": q["options"]}


def _grade(quiz_id: str, answer: Any) -> Optional[dict]:
    try:
        view, idx = quiz_id.split(":")
        q = QUIZ[view][int(idx)]
        answer = int(answer)
    except (KeyError, ValueError, IndexError, AttributeError, TypeError):
        return None
    correct = answer == q["answer"]
    return {
        "id": quiz_id,
        "correct": correct,
        "your_answer": answer,
        "correct_index": q["answer"],
        "correct_option": q["options"][q["answer"]],
        "explanation": q["explanation"],
        "highlight": q["highlight"],
        "view": view,
        "index": int(idx),
    }


def _find_sample(text: str, samples: list[dict], ctx: dict) -> Optional[dict]:
    """Resolve "load the HFrEF phantom", "patient 38", "next patient" ... to a sample."""
    if not samples:
        return None
    current = ctx.get("sample_id")
    subjects: list[str] = []
    for s in samples:
        if s["subject_id"] not in subjects:
            subjects.append(s["subject_id"])
    want_view = "A2C" if re.search(r"\b(a2c|2c|two[\s-]*chamber)\b", text) else "A4C"

    def pick(subject_id: str) -> Optional[dict]:
        seqs = [s for s in samples if s["subject_id"] == subject_id]
        return next((s for s in seqs if s["view"] == want_view), seqs[0] if seqs else None)

    m = re.search(r"patient\s*0*(\d+)", text)
    if m:
        target = f"patient{int(m.group(1)):04d}"
        return next((pick(sid) for sid in subjects if sid.endswith(target)), None)
    if re.search(r"hfref|reduced|low\s+ef|dilated|heart\s+failure", text):
        if "phantom" in text or not any(s["kind"] == "camus" for s in samples):
            return next((s for s in samples if s["id"] == "phantom_hfref"), None)
        return next((pick(s["subject_id"]) for s in samples
                     if s["kind"] == "camus" and (s.get("ef_class") or "").startswith("reduced")), None)
    if re.search(r"normal|healthy|preserved", text):
        if "phantom" in text or not any(s["kind"] == "camus" for s in samples):
            return next((s for s in samples if s["id"] == "phantom_normal"), None)
        return next((pick(s["subject_id"]) for s in samples
                     if s["kind"] == "camus" and (s.get("ef_class") or "").startswith("normal")), None)
    if "phantom" in text:
        return next((s for s in samples if s["kind"] == "phantom" and s["id"] != current), None)
    if re.search(r"\b(real|camus)\b", text):
        return next((s for s in samples if s["kind"] == "camus" and s["id"] != current), None)
    cur_subject = next((s["subject_id"] for s in samples if s["id"] == current), None)
    if re.search(r"\b(next|another)\b", text) and cur_subject in subjects:
        return pick(subjects[(subjects.index(cur_subject) + 1) % len(subjects)])
    if re.search(r"\bprevious\b", text) and cur_subject in subjects:
        return pick(subjects[(subjects.index(cur_subject) - 1) % len(subjects)])
    return None


def _base(ctx: dict) -> dict:
    return {"intent": "help", "reply": "", "actions": [], "quiz": None, "quiz_result": None, "verified": None,
            "presets": PRESETS, "engine": "deterministic"}


def deterministic_respond(message: str = "", preset: Optional[str] = None, context: Optional[dict] = None,
                          quiz_answer: Optional[dict] = None, samples: Optional[list] = None) -> dict:
    """Pure function of its inputs: the same request always yields the same reply."""
    ctx = dict(context or {})
    samples = samples or []
    current_view = ctx.get("view") if ctx.get("view") in VIEWS else "A4C"
    apical = current_view if current_view in ("A4C", "A2C") else "A4C"
    try:
        current_tilt = max(0.0, min(30.0, float(ctx.get("tilt_deg", 0.0))))
    except (TypeError, ValueError):
        current_tilt = 0.0

    def as_int(key: str) -> int:
        try:
            return max(0, int(ctx.get(key, 0)))
        except (TypeError, ValueError):
            return 0

    quiz_index, asked, correct_so_far = as_int("quiz_index"), as_int("quiz_asked"), as_int("quiz_correct")
    out = _base(ctx)

    # 1) grading a quiz answer takes priority; the next question follows automatically
    if quiz_answer:
        result = _grade(str(quiz_answer.get("id", "")), quiz_answer.get("answer"))
        if result is None:
            out.update(intent="quiz_error", reply="I couldn't find that quiz question — ask me for a new one.")
            return out
        score_c, score_n = correct_so_far + int(result["correct"]), asked + 1
        verdict = "✅ **Correct.**" if result["correct"] else f"❌ **Not quite** — the answer is **{result['correct_option']}**."
        reply = f"{verdict} {result['explanation']}\n**Score so far: {score_c} / {score_n}.**"
        out.update(intent="quiz_result", quiz_result=result)
        out["actions"] = [{"type": "set_view", "view": result["view"]}, {"type": "highlight", "chambers": result["highlight"]}]
        bank = QUIZ[result["view"]]
        if result["index"] + 1 < len(bank):
            nxt = _quiz_payload(result["view"], result["index"] + 1)
            out["quiz"] = nxt
            reply += f"\n**Next (question {nxt['number']} of {nxt['total']}):** {nxt['question']}"
        else:
            other = [v for v in VIEWS if v != result["view"]]
            reply += (f"\nThat was the last {result['view']} question. Say **“quiz me on {other[0]}”** for another view.")
        out["reply"] = reply
        return out

    if preset and preset in _PRESET_BY_ID:
        message = _PRESET_BY_ID[preset]["message"]
    raw = (message or "").strip()
    text = _normalise(raw)

    view = _detect_view(text)
    wants_foreshort = bool(_RE_FORESHORTEN.search(text))
    degrees = _RE_DEGREES.search(raw)

    # 2) landmark quiz (optionally on a named view)
    if _RE_QUIZ.search(text):
        qv = view or current_view
        quiz = _quiz_payload(qv, quiz_index)
        out.update(intent="quiz", quiz=quiz,
                   reply=f"**Landmark quiz · {VIEWS[qv]['label']}** (question {quiz['number']} of {quiz['total']})\n{quiz['question']}")
        out["actions"] = [{"type": "set_view", "view": qv}, {"type": "set_tilt", "tilt_deg": 0}]
        return out

    # 3) foreshortening: fix / demo / explain
    if wants_foreshort:
        if _RE_FIX.search(text):
            out.update(
                intent="fix_foreshortening",
                reply=(
                    "**Fixing a foreshortened apical view**\n"
                    "1. Find the true apex: slide the probe **one rib-space lower and more lateral**; the apex "
                    "should be thin-walled, barely moving, and at the very top of the sector.\n"
                    "2. Maximise LV length — the long axis should be the **longest you can obtain** and the LV "
                    "should look like a bullet, not a ball.\n"
                    "3. Check that A4C and A2C long-axis lengths **agree within 10 %**.\n"
                    "4. Left-lateral decubitus with a cut-out mattress and held expiration helps.\n"
                    "I've reset the tilt to 0° so the fan passes through the true apex again."
                ),
            )
            out["actions"] = [{"type": "set_view", "view": apical}, {"type": "set_tilt", "tilt_deg": 0}]
            return out
        if degrees or _RE_DEMO.search(text):
            tilt = max(0.0, min(30.0, float(degrees.group(1)))) if degrees else 20.0
            out.update(
                intent="demo_foreshortening",
                reply=(f"**Foreshortening demo.** I tilted the apical plane {tilt:.0f}° off the true long axis: watch "
                       "the fan slide off the yellow apex marker and the cut LV become short and round."),
                verified=_foreshortening_numbers(ctx, tilt),
            )
            target = view if view in ("A4C", "A2C") else apical
            out["actions"] = [{"type": "set_view", "view": target}, {"type": "set_tilt", "tilt_deg": tilt}]
            return out
        demo_tilt = current_tilt if current_tilt > 0 else 15.0
        out.update(
            intent="explain_foreshortening",
            reply=(
                "**Apical foreshortening** happens when the imaging plane does not pass through the true LV "
                "apex — usually because the probe sits one rib-space too high. The plane then slices obliquely "
                "through the anterior wall: the LV looks shorter and rounder, and the false 'apex' appears to "
                "contract briskly.\n"
                "**Geometry:** for a cavity of length a and base radius b, a plane through the mitral-annulus "
                "centre tilted by θ sees an apparent length a′ = 1 / √(cos²θ / a² + sin²θ / b²). Simpson's rule "
                "integrates disks along that axis, so the reported volume scales by a′ / a.\n"
                "**Why it matters:** both EDV and ESV are under-estimated. The systolic cavity is more elongated "
                "(larger a / b), so ESV loses proportionally more — and **LVEF is over-estimated**.\n"
                "Drag the **tilt slider** in the 3-D pane to see it happen."
            ),
            verified=_foreshortening_numbers(ctx, demo_tilt),
        )
        out["actions"] = [{"type": "set_view", "view": apical}]
        return out

    # 4) an explicit tilt command ("tilt 15 degrees", "set the tilt to 10")
    if _RE_TILT.search(text):
        num = degrees or _RE_NUMBER.search(raw)
        if num:
            tilt = max(0.0, min(30.0, float(num.group(1))))
            out.update(intent="set_tilt", reply=f"Tilt set to **{tilt:.0f}°** on the {apical} plane.",
                       verified=_foreshortening_numbers(ctx, tilt) if tilt > 0 else None)
            out["actions"] = [{"type": "set_view", "view": apical}, {"type": "set_tilt", "tilt_deg": tilt}]
            return out

    # 5) overlays ("hide the mask", "show the sector fan")
    overlay = _detect_overlay(text)
    if overlay and (_RE_HIDE.search(text) or _RE_SHOW.search(text)) and not _RE_PROMPT.search(text.replace("mask", "")):
        on = not bool(_RE_HIDE.search(text))
        names = {"mask": "segmentation mask", "axis": "long-axis line", "chords": "20 Simpson chords", "fan": "sector fan",
                 "cutaway": "cut-away", "labels": "3-D labels", "beat": "LV beat animation"}
        out.update(intent="toggle_overlay", reply=f"{'Showing' if on else 'Hiding'} the **{names[overlay]}**.")
        out["actions"] = [{"type": "toggle_overlay", "overlay": overlay, "on": on}]
        return out

    # 6) loading a sample
    if _RE_LOAD.search(text):
        target = _find_sample(text, samples, ctx)
        if target:
            out.update(intent="select_sample", reply=f"Loading **{target['name']}** ({target['view']}).")
            out["actions"] = [{"type": "select_sample", "sample_id": target["id"]}]
            if target["view"] in ("A4C", "A2C"):
                out["actions"].append({"type": "set_view", "view": target["view"]})
            return out
        if samples:
            names = ", ".join(sorted({s["subject_id"].replace("camus_", "") for s in samples})[:10])
            out.update(intent="select_sample_unknown", reply=f"I couldn't match that to a sample. Available subjects: {names}.")
            return out

    # 7) probe view switch
    if view and not _RE_BIPLANE.search(text):
        out.update(intent="set_view", reply=_view_reply(view))
        out["actions"] = [{"type": "set_view", "view": view}, {"type": "set_tilt", "tilt_deg": 0},
                          {"type": "highlight", "chambers": VIEWS[view]["highlight"]}]
        return out

    # 8) single-plane vs biplane
    if _RE_BIPLANE.search(text):
        bp = ctx.get("biplane_lvef")
        state = (f"For this subject the biplane LVEF is currently **{bp:.1f} %**." if isinstance(bp, (int, float))
                 else "To get a biplane value here, segment **both** the A4C and the A2C clip of the same CAMUS patient.")
        out.update(
            intent="explain_biplane",
            reply=(
                "**Single-plane vs biplane method of disks** (Lang et al., ASE/EACVI 2015)\n"
                "**Single plane:** V = π/4 · Σ Dᵢ² · L/20 — one apical view, each of the 20 disks assumed circular "
                "with diameter Dᵢ.\n"
                "**Biplane:** V = π/4 · Σ aᵢ · bᵢ · L/20 — disk i is an ellipse whose axes aᵢ and bᵢ are measured "
                "in A4C and A2C; L is the longer of the two long axes.\n"
                "Biplane is the recommended 2-D method because it does not assume circular cross-sections, which "
                "matters in regional wall-motion abnormality. It only helps if both views are on axis: a long-axis "
                "mismatch above 10 % means one of them is foreshortened.\n" + state
            ),
        )
        return out

    # 9) Simpson's method
    if _RE_SIMPSON.search(text):
        out.update(
            intent="explain_simpson",
            reply=(
                "**Modified Simpson's rule (method of disks)**\n"
                "1. Trace the LV endocardium in an apical view (here SAM 2.1 does it — the **green mask**).\n"
                "2. Draw the **long axis** from the apex to the middle of the mitral annulus (**yellow**; "
                "EchoPlane3D uses the mask's principal PCA axis).\n"
                "3. Cut the cavity into **20 disks** of equal height h = L / 20, perpendicular to that axis "
                "(**cyan chords** are the diameters Dᵢ).\n"
                "4. Volume = Σ π/4 · Dᵢ² · h (single plane). Biplane uses π/4 · aᵢ · bᵢ · h with diameters "
                "from A4C and A2C.\n"
                "5. LVEF = (EDV − ESV) / EDV × 100.\n"
                "Everything depends on the long axis being the true one — which is why foreshortening is so "
                "damaging. Ask me about **single-plane vs biplane** for the difference."
            ),
        )
        return out

    # 10) interpretation of the numbers on screen
    if _RE_INTERPRET.search(text):
        reply, verified = _interpretation(ctx)
        out.update(intent="interpret", reply=reply, verified=verified)
        return out

    # 11) anatomy of a named chamber or vessel
    structure = _detect_structure(text)
    if structure:
        a = ANATOMY[structure]
        out.update(intent="anatomy", reply=a["text"])
        out["actions"] = [{"type": "highlight", "chambers": [structure]}]
        if a["view"]:
            out["actions"].insert(0, {"type": "set_view", "view": a["view"]})
        else:
            out["actions"].insert(0, {"type": "toggle_overlay", "overlay": "cutaway", "on": False})
            out["reply"] += "\nI switched the cut-away off so you can see it in the 3-D model."
        return out

    # 12) prompting / segmentation help
    if _RE_PROMPT.search(text):
        out.update(
            intent="prompting_help",
            reply=(
                "**Getting a good LV mask**\n"
                "• Go to **frame 1** and click in the **dark blood pool**, away from the walls and valve.\n"
                "• **Auto-snap** (on by default) moves a click that landed on bright wall into the adjacent "
                "cavity; the dashed violet line shows the move. Switch it off for raw clicks.\n"
                "• **Multi-point:** click = cavity (+), Shift+click = wall or left atrium (−), then *Segment "
                "points*. A negative point on the mitral valve stops leaks into the atrium.\n"
                "• A ⚠ badge means a plausibility check failed (volume jump, non-compact mask, area far from the "
                "reference). On real ultrasound a small foundation model often under-fills hazy cavities — treat "
                "those volumes with suspicion."
            ),
        )
        return out

    # 13) probe handling
    if _RE_PROBE.search(text):
        out.update(
            intent="probe_orientation",
            reply=(
                "**Probe orientation marker.** The ridge or light on the probe corresponds to the side of the "
                "screen carrying the manufacturer's marker — on the right in adult cardiac presets. In the 3-D "
                "pane the **green dot** on the probe is that marker.\n"
                "A4C ≈ 3 o'clock, A2C ≈ 12 o'clock, PLAX ≈ 10 o'clock (right shoulder), PSAX ≈ 2 o'clock (left "
                "shoulder). Switch to **Echo-aligned** to see each cut exactly as the screen shows it."
            ),
        )
        return out

    # 14) about / safety
    if _RE_ABOUT.search(text):
        out.update(
            intent="about",
            reply=(
                "I'm a **rule-based tutor**: a fixed knowledge base plus calculations done by the same maths "
                "engine as the workbench. Everything runs on this machine — no data leaves it, and no cloud "
                "service is contacted. If a local Ollama model is running, it may rephrase explanations; numbers, "
                "quizzes and UI actions still come from the deterministic engine.\n"
                f"**{DISCLAIMER}** Nothing here is validated for patient care."
            ),
        )
        return out

    # 15) help / fallback
    if _RE_HELP.search(raw):
        out.update(intent="help", reply=WELCOME)
    else:
        out.update(
            intent="fallback",
            reply="I'm a rule-based tutor, so I only know a few topics. Try: **“show me PLAX”**, "
                  "**“explain foreshortening”**, **“demo a 20 degree tilt”**, **“single-plane vs biplane”**, "
                  "**“interpret this LVEF”**, **“what is the septum?”**, **“hide the mask”**, "
                  "**“load the HFrEF phantom”** or **“quiz me”**.",
        )
    return out


# =========================================================================== #
# Optional local LLM (Ollama)
# =========================================================================== #
class LLMReply(BaseModel):
    reply: str = Field(min_length=1, max_length=1600)
    actions: list[Any] = Field(default_factory=list, max_length=6)


class OllamaClient:
    """Minimal ``/api/chat`` client with a strict connect/read timeout."""

    def __init__(self, base_url: str = config.OLLAMA_URL, model: str = config.OLLAMA_MODEL,
                 timeout: float = config.OLLAMA_TIMEOUT_SEC, transport: Optional[httpx.BaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self._transport = transport
        self._probe: tuple[float, bool] = (0.0, False)

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout), transport=self._transport)

    def available(self, max_age: float = 20.0) -> bool:
        """Is an Ollama server answering?  Cached so a missing server costs nothing per turn."""
        now = time.monotonic()
        if now - self._probe[0] < max_age and self._probe[0] > 0:
            return self._probe[1]
        ok = False
        try:
            with self._client(min(self.timeout, 1.0)) as c:
                ok = c.get("/api/tags").status_code == 200
        except (httpx.HTTPError, OSError):
            ok = False
        self._probe = (now, ok)
        return ok

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "keep_alive": "2m",                       # release VRAM soon after use; SAM 2.1 shares the 6 GB GPU
            "options": {"temperature": 0.2, "num_predict": 320},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        with self._client(self.timeout) as c:
            res = c.post("/api/chat", json=payload)
            res.raise_for_status()
            return str(res.json()["message"]["content"])


_ollama: Optional[OllamaClient] = None


def get_ollama() -> OllamaClient:
    global _ollama
    if _ollama is None:
        _ollama = OllamaClient()
    return _ollama


def set_ollama(client: Optional[OllamaClient]) -> None:
    """Replace the Ollama client (tests inject one backed by ``httpx.MockTransport``)."""
    global _ollama
    _ollama = client


def engine_status() -> dict:
    """Which engine will answer the next free-text turn."""
    if not config.OLLAMA_ENABLED:
        return {"id": "deterministic", "label": "Offline Deterministic Tutor", "model": None, "reachable": False,
                "ollama_enabled": False}
    client = get_ollama()
    up = client.available()
    return {
        "id": "ollama" if up else "deterministic",
        "label": "Ollama Local LLM" if up else "Offline Deterministic Tutor",
        "model": client.model if up else None,
        "reachable": up,
        "ollama_enabled": True,
    }


def parse_llm_json(content: str) -> Optional[LLMReply]:
    """Extract and validate the JSON object an LLM was asked to produce."""
    if not isinstance(content, str):
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return LLMReply.model_validate(json.loads(text[start:end + 1]))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _llm_system_prompt(ctx: dict, det: dict, samples: list[dict]) -> str:
    sample_ids = ", ".join(s["id"] for s in samples[:24]) or "(none)"
    return (
        "You are the tutor inside EchoPlane3D, an offline echocardiography TRAINING simulator (not a medical "
        "device). Explain concisely (at most 120 words) for a sonography or medical student. Never give a "
        "diagnosis or patient-specific advice. Never state numbers of your own: if numbers are needed, refer to "
        "the 'verified calculation' the app shows next to your answer.\n"
        'Reply with ONE JSON object: {"reply": "<text>", "actions": [<zero or more actions>]}.\n'
        "Allowed actions (anything else is discarded):\n"
        '  {"type":"set_view","view":"A4C|A2C|PLAX|PSAX"}\n'
        '  {"type":"set_tilt","tilt_deg":0-30}\n'
        '  {"type":"toggle_overlay","overlay":"mask|axis|chords|fan|cutaway|labels|beat","on":true|false}\n'
        '  {"type":"start_quiz","view":"A4C|A2C|PLAX|PSAX"}\n'
        '  {"type":"select_sample","sample_id":"<one of: ' + sample_ids + '>"}\n'
        f"Current state: 3-D view {ctx.get('view')}, tilt {ctx.get('tilt_deg')} deg, sample {ctx.get('sample_id')}, "
        f"segmented {bool(ctx.get('segmented'))}.\n"
        f"The deterministic tutor classified this turn as '{det['intent']}' and would answer:\n{det['reply'][:700]}"
    )


def llm_respond(message: str, ctx: dict, det: dict, samples: list[dict]) -> Optional[dict]:
    """One Ollama turn.  Returns None on any failure so the caller falls back."""
    client = get_ollama()
    try:
        content = client.chat(_llm_system_prompt(ctx, det, samples), message)
    except (httpx.HTTPError, OSError, KeyError, ValueError, TypeError):
        client._probe = (time.monotonic(), False)       # do not keep hammering a server that just failed
        return None
    parsed = parse_llm_json(content)
    if parsed is None:
        return None
    actions, rejected = validate_actions(parsed.actions, {s["id"] for s in samples}, LLM_ALLOWED_ACTIONS)
    return {"reply": parsed.reply.strip(), "actions": actions, "rejected_actions": rejected}


# =========================================================================== #
# Public entry point
# =========================================================================== #
_DETERMINISTIC_ONLY = {"quiz", "quiz_result", "quiz_error", "toggle_overlay", "select_sample", "set_tilt", "help"}


def respond(message: str = "", preset: Optional[str] = None, context: Optional[dict] = None,
            quiz_answer: Optional[dict] = None, samples: Optional[list] = None) -> dict:
    """Route one tutor turn through the hybrid engine.

    Returns ``reply`` (light markdown), validated UI ``actions``, optional
    ``quiz`` / ``quiz_result``, ``verified`` (deterministic calculation text or
    None), ``engine`` (``deterministic`` | ``ollama``) and ``engine_status``.
    """
    samples = samples or []
    det = deterministic_respond(message, preset, context, quiz_answer, samples)
    det["actions"], _ = validate_actions(det["actions"], {s["id"] for s in samples} if samples else None)
    status = engine_status()
    det["engine_status"] = status

    text = (message or "").strip()
    use_llm = (status["id"] == "ollama" and not quiz_answer and not preset and text
               and det["intent"] not in _DETERMINISTIC_ONLY)
    if not use_llm:
        return det

    llm = llm_respond(text, dict(context or {}), det, samples)
    if llm is None:
        det["engine_status"] = engine_status()
        det["llm_fallback"] = True
        return det

    out = dict(det)
    out["engine"] = "ollama"
    out["reply"] = llm["reply"]
    # a recognised intent keeps its deterministic actions; otherwise the LLM's validated ones apply
    if det["intent"] == "fallback":
        actions = []
        for a in llm["actions"]:
            if a["type"] == "start_quiz":
                ctx = dict(context or {})
                qv = a.get("view") or (ctx.get("view") if ctx.get("view") in VIEWS else "A4C")
                try:
                    idx = int(ctx.get("quiz_index", 0))
                except (TypeError, ValueError):
                    idx = 0
                out["quiz"] = _quiz_payload(qv, idx)
                out["reply"] += f"\n{out['quiz']['question']}"
                actions.append({"type": "set_view", "view": qv})
            else:
                actions.append(a)
        out["actions"] = actions
        out["intent"] = "llm"
    else:
        # keep the deterministic explanation available as the verified text when it carries no separate numbers
        out["verified"] = det["verified"] or None
    out["rejected_actions"] = llm["rejected_actions"]
    return out
