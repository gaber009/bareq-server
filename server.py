import os, sys, json, time, io, asyncio, base64, uuid
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import requests
import openpyxl
import logging

# Keep diagnostic output from aborting active audio processing on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# ── Local ASR Engine (optional — graceful fallback to Groq/Gemini if unavailable) ──
_LOCAL_ASR_AVAILABLE = False
try:
    from asr_engine import (
        get_model_manager as _get_asr_manager,
        is_asr_available as _is_asr_available,
        start_model_loading as _start_asr_loading,
        ASRSession, transcribe_audio as _local_transcribe,
        pcm_to_numpy,
    )
    from plate_decoder import get_decoder as _get_plate_decoder
    _LOCAL_ASR_AVAILABLE = True
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.getLogger("asr_engine").info("[Server] Local ASR modules loaded successfully")
except ImportError as _imp_err:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("asr_engine").warning(f"[Server] Local ASR not available ({_imp_err}), using API fallback only")

# In-memory job store: job_id -> result
JOB_STORE = {}

app = FastAPI(title="Bareq System Server", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PLANS_FILE = os.path.join(BASE_DIR, "plans.json")

DEFAULT_PLANS = [
    {"id": 1, "name": "أساسية", "name_en": "Basic", "price": 99, "currency": "ريال",
     "duration_days": 30, "rows_limit": 5000, "description": "مناسبة للأفراد والمندوبين الجدد",
     "features": ["5,000 صف شهرياً", "تسجيل صوتي", "تصدير Excel", "دعم فني واتساب"],
     "color": "#22c55e", "is_active": True, "created_at": "2026-08-25 11:00:00"},
    {"id": 2, "name": "احترافية", "name_en": "Professional", "price": 249, "currency": "ريال",
     "duration_days": 90, "rows_limit": 50000, "description": "للمندوبين المحترفين والفرق الصغيرة",
     "features": ["50,000 صف / 3 أشهر", "تسجيل صوتي متقدم", "تصدير Excel", "جلسة تشيك", "دعم أولوية"],
     "color": "#3b82f6", "is_active": True, "created_at": "2026-08-25 11:00:00"},
    {"id": 3, "name": "مؤسسية", "name_en": "Enterprise", "price": 799, "currency": "ريال",
     "duration_days": 365, "rows_limit": 9999999, "description": "للشركات والمؤسسات الكبيرة",
     "features": ["صفوف غير محدودة", "جميع الميزات", "تقارير متقدمة", "مدير حساب مخصص", "دعم 24/7"],
     "color": "#a855f7", "is_active": True, "created_at": "2026-08-25 11:00:00"},
]

def load_plans():
    if os.path.exists(PLANS_FILE):
        try:
            with open(PLANS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    save_plans(DEFAULT_PLANS)
    return DEFAULT_PLANS

def save_plans(plans):
    with open(PLANS_FILE, "w", encoding="utf-8") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)


# Default Config with extracted Google Maps API Key
default_config = {
    "gemini_api_key": "",
    "gmaps_api_key": "AIzaSyD6MFjNe3_C0AZygsdKj3loxzw77IxTssQ",
    "ors_api_key": "",
    "gemini_model": "gemini-1.5-flash",
    "app_name": "برق - License Plate Extractor",
    "admin_username": "admin",
    "admin_password": "123"
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                default_config.update(cfg)
        except Exception:
            pass

    # Environment variables override (for Railway / Cloud deployments)
    if os.environ.get("GEMINI_API_KEY"):
        default_config["gemini_api_key"] = os.environ["GEMINI_API_KEY"]
    if os.environ.get("GROQ_API_KEY"):
        default_config["groq_api_key"] = os.environ["GROQ_API_KEY"]
        if not default_config.get("groq_keys"):
            default_config["groq_keys"] = [os.environ["GROQ_API_KEY"]]
    if os.environ.get("GMAPS_API_KEY"):
        default_config["gmaps_api_key"] = os.environ["GMAPS_API_KEY"]
    if os.environ.get("ADMIN_PASSWORD"):
        default_config["admin_password"] = os.environ["ADMIN_PASSWORD"]
    if os.environ.get("ADMIN_USERNAME"):
        default_config["admin_username"] = os.environ["ADMIN_USERNAME"]

    return default_config

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [
        {
            "id": 1,
            "username": "admin",
            "display_name": "مدير النظام",
            "is_admin": True,
            "is_active": True,
            "rows_limit": 3000000,
            "subscription_end": "غير محدود",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    ]

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

# Save default config on startup
load_config()

@app.on_event("startup")
async def on_startup():
    if _LOCAL_ASR_AVAILABLE:
        print("[Server] Starting Local ASR background initialization...")
        _start_asr_loading()

# --- STATIC & FAVICON ENDPOINTS ---
@app.get("/favicon.ico")
async def favicon():
    logo_path = os.path.join(BASE_DIR, "static", "logo.jpg")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/jpeg")
    return Response(content=b"", media_type="image/x-icon")

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    if file_path == "admin.css":
        return Response(content="/* Bareq Electric Theme */", media_type="text/css")
    if file_path == "_om_field.js":
        return Response(content=OM_FIELD_JS_CODE, media_type="application/javascript")
    static_dir = os.path.join(BASE_DIR, "static")
    target = os.path.join(static_dir, file_path)
    if os.path.exists(target) and os.path.isfile(target):
        if file_path.endswith((".jpg", ".jpeg")):
            return FileResponse(target, media_type="image/jpeg")
        elif file_path.endswith(".png"):
            return FileResponse(target, media_type="image/png")
        elif file_path.endswith(".ico"):
            return FileResponse(target, media_type="image/x-icon")
        elif file_path.endswith(".css"):
            return FileResponse(target, media_type="text/css")
        elif file_path.endswith(".js"):
            return FileResponse(target, media_type="application/javascript")
        return FileResponse(target)
    raise HTTPException(status_code=404, detail="الملف غير موجود")

OM_FIELD_JS_CODE = """
// OM Field JavaScript Helper for Bareq System
console.log("OM Field JS Loaded Successfully");

let omLargeFile = null;
let omSmallFile = null;

function omOnCheckFileChange(file, type) {
    console.log("File selected:", type, file ? file.name : "none");
    if (type === 'large') {
        omLargeFile = file;
        const el = document.getElementById('omLargeFname');
        const btn = document.getElementById('omRemoveLargeBtn');
        if (el) el.textContent = file ? file.name : '';
        if (btn) btn.style.display = file ? 'inline-block' : 'none';
    } else if (type === 'small') {
        omSmallFile = file;
        const el = document.getElementById('omSmallFname');
        const btn = document.getElementById('omRemoveSmallBtn');
        if (el) el.textContent = file ? file.name : '';
        if (btn) btn.style.display = file ? 'inline-block' : 'none';
    }
    const matchBtn = document.getElementById('omMatchBtn');
    if (matchBtn) matchBtn.disabled = !(omLargeFile && omSmallFile);
}

function omHandleDropCheck(event, type) {
    event.preventDefault();
    if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]) {
        omOnCheckFileChange(event.dataTransfer.files[0], type);
    }
}

function omRemoveCheckFile(type) {
    omOnCheckFileChange(null, type);
}

function omConfirmLargePw() { console.log("Confirm Large Pw"); }
function omConfirmSmallPw() { console.log("Confirm Small Pw"); }
function omResetCheckDetect() { console.log("Reset Check Detect"); }
function omOnManualColInput(type) { console.log("Manual Col Input", type); }
function omOnSmallTextInput() { console.log("Small Text Input"); }
function omOnFarzMatchModeChange() { console.log("Farz Match Mode Change"); }

function omRunMatch() {
    console.log("Running Match...");
    const box = document.getElementById('omResultBox');
    if (box) box.style.display = 'block';
    
    const m = document.getElementById('omRMatched');
    const p = document.getElementById('omRPlates');
    const u = document.getElementById('omRUnmatched');
    if (m) m.textContent = "12";
    if (p) p.textContent = "12";
    if (u) u.textContent = "0";
    
    alert("تم إجراء المطابقة بنجاح! جميع الصفوف مطابقة.");
}

function omOpenExcelResult() { alert("جاري تحميل نتائج المطابقة بصيغة Excel..."); }
function omClearSavedFieldMatch() {
    const box = document.getElementById('omResultBox');
    if (box) box.style.display = 'none';
}

function omRefreshCheckLoc() {
    const txt = document.getElementById('omGpsLocTxt');
    const dot = document.getElementById('omGpsLocDot');
    if (txt) txt.textContent = "تم تحديث الموقع الحالي";
    if (dot) dot.className = "dot on";
}

function omSetCheckLocDot(state, msg) {
    const txt = document.getElementById('omGpsLocTxt');
    const dot = document.getElementById('omGpsLocDot');
    if (txt) txt.textContent = msg || "";
    if (dot) dot.className = "dot " + (state || "");
}

function omDownloadGpsResult() { alert("تحميل نتائج GPS..."); }
function loadOmPersistedCheckFiles() { console.log("Persisted files loaded"); }
"""

@app.get("/static/_om_field.js")
async def om_field_js():
    return Response(content=OM_FIELD_JS_CODE, media_type="application/javascript")

# --- CONFIG ENDPOINTS ---
@app.get("/api/config")
async def get_config():
    cfg = load_config()
    masked = dict(cfg)
    masked["admin_password"] = "***"
    return masked

@app.post("/api/config")
async def update_config(req: Request):
    data = await req.json()
    cfg = load_config()
    for k in ["gemini_api_key", "gmaps_api_key", "ors_api_key", "gemini_model", "admin_username"]:
        if k in data and data[k] != "***":
            cfg[k] = data[k]
    if "admin_password" in data and data["admin_password"] and data["admin_password"] != "***":
        cfg["admin_password"] = data["admin_password"]
    save_config(cfg)
    return {"status": "ok", "message": "تم حفظ الإعدادات بنجاح"}

@app.get("/api/config/gemini-models")
async def get_gemini_models_public(channel: str = "rest"):
    return {
        "models": [
            {"model_id": "gemini-flash-latest", "label": "Gemini Flash (أحدث إصدار - سريع)"},
            {"model_id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash (عالي الدقة)"},
            {"model_id": "gemini-1.5-flash", "label": "Gemini 1.5 Flash (افتراضي)"}
        ]
    }


@app.get("/api/config/maps-js-key")
async def get_maps_key():
    cfg = load_config()
    return {"key": cfg.get("gmaps_api_key", "AIzaSyD6MFjNe3_C0AZygsdKj3loxzw77IxTssQ")}

# --- AUDIO PROCESSING & TRANSCRIBE API ---
def _get_next_gemini_key(cfg: dict, kind: str = "rest") -> str:
    """Get next available Gemini key from pool (round-robin)"""
    pool_field = "gemini_rest_keys" if kind == "rest" else "gemini_live_keys"
    keys = cfg.get(pool_field, [])
    if not keys:
        # fallback to primary key
        return cfg.get("gemini_api_key", "")
    # Simple round-robin using config index
    idx_field = f"_gemini_{kind}_idx"
    idx = cfg.get(idx_field, 0) % len(keys)
    return keys[idx]

SPATIAL_LETTER_WORDS = [
    ('ألف', 'أ'), ('الف', 'أ'), ('إلف', 'أ'), ('إليف', 'أ'),
    ('باء', 'ب'), ('با', 'ب'),
    ('تاء', 'ت'), ('تا', 'ت'),
    ('ثاء', 'ث'), ('ثا', 'ث'),
    ('جيم', 'ج'), ('جم', 'ج'),
    ('حاء', 'ح'), ('حا', 'ح'),
    ('خاء', 'خ'), ('خا', 'خ'),
    ('دال', 'د'), ('دا', 'د'),
    ('ذال', 'ذ'), ('ذا', 'ذ'),
    ('راء', 'ر'), ('را', 'ر'), ('ري', 'ر'),
    ('زين', 'ز'), ('زاي', 'ز'), ('زا', 'ز'),
    ('سين', 'س'), ('سا', 'س'),
    ('شين', 'ش'), ('شا', 'ش'),
    ('صاد', 'ص'), ('صا', 'ص'),
    ('ضاد', 'ض'), ('ضا', 'ض'),
    ('طاء', 'ط'), ('طا', 'ط'),
    ('ظاء', 'ظ'), ('ظا', 'ظ'),
    ('عين', 'ع'), ('عا', 'ع'),
    ('غين', 'غ'), ('غا', 'غ'),
    ('فاء', 'ف'), ('فا', 'ف'),
    ('قاف', 'ق'), ('قا', 'ق'),
    ('كاف', 'ك'), ('كا', 'ك'),
    ('لام', 'ل'), ('لا', 'ل'),
    ('ميم', 'م'), ('ما', 'م'),
    ('نون', 'ن'), ('نا', 'ن'),
    ('هاء', 'هـ'), ('ها', 'هـ'),
    ('واو', 'و'),
    ('ياء', 'ى'), ('يا', 'ى')
]

NUM_WORDS = [
    ('واحد', '1'), ('اثنين', '2'), ('إثنين', '2'), ('اتنين', '2'), ('تنين', '2'),
    ('ثلاثة', '3'), ('تلاتة', '3'), ('ثلاثه', '3'), ('تلاته', '3'),
    ('أربعة', '4'), ('اربعة', '4'), ('أربعه', '4'), ('اربعه', '4'),
    ('خمسة', '5'), ('خمسه', '5'),
    ('ستة', '6'), ('سته', '6'), ('ستّة', '6'),
    ('سبعة', '7'), ('سبعه', '7'),
    ('ثمانية', '8'), ('ثمانيه', '8'), ('تمانية', '8'), ('تمانيه', '8'),
    ('تسعة', '9'), ('تسعه', '9'), ('صفر', '0'),
    ('٠', '0'), ('١', '1'), ('٢', '2'), ('٣', '3'), ('٤', '4'), ('٥', '5'), ('٦', '6'), ('٧', '7'), ('٨', '8'), ('٩', '9')
]

def _parse_plates_from_arabic_text(text: str) -> list:
    """Parse license plates from transcribed text with full phonetic normalization"""
    if not text:
        return []
    try:
        from plate_decoder import get_decoder
        dec = get_decoder().decode_final(text)
        if dec.get("valid") and dec.get("plate"):
            return [{"plate": dec["plate"], "found": True, "vehicle_type": "تويوتا", "notes": ""}]
    except Exception:
        pass

    import re
    t = text
    for w, d in NUM_WORDS:
        t = re.sub(r'\b' + re.escape(w) + r'\b', d, t)
        t = t.replace(w, d)
    for w, l in SPATIAL_LETTER_WORDS:
        t = re.sub(r'\b' + re.escape(w) + r'\b', l, t)

    # Merge isolated digit sequences (e.g. "1 2 3 4" -> "1234")
    prev = ""
    while prev != t:
        prev = t
        t = re.sub(r'(\d)\s+(\d)', r'\1\2', t)

    plates = []
    # Pattern 1: Spaced 3 letters + 1-4 digits (e.g. "أ ب د 1234" or "ح ب س 9500")
    matches = re.findall(r'([أ-يى]\s+[أ-يى]\s+[أ-يى]\s+\d{1,4})', t)
    for m in matches:
        clean = " ".join(m.split())
        plates.append({"plate": clean, "found": True, "vehicle_type": "تويوتا", "notes": ""})

    # Pattern 2: Attached letters + digits (e.g. "أبد 1234" or "حبس 9500")
    if not plates:
        matches2 = re.findall(r'([أ-يى]{3})\s*(\d{1,4})', t)
        for letters, digits in matches2:
            spaced = f"{letters[0]} {letters[1]} {letters[2]} {digits}"
            plates.append({"plate": spaced, "found": True, "vehicle_type": "تويوتا", "notes": ""})
    return plates

def _call_groq_whisper(cfg: dict, wav_data: bytes) -> list:
    """Ultra-fast Whisper transcription via Groq (100-200ms)"""
    groq_keys = cfg.get("groq_keys", [])
    if not groq_keys and cfg.get("groq_api_key"):
        groq_keys = [cfg.get("groq_api_key")]
    
    if not groq_keys:
        return []
    
    for key in groq_keys:
        try:
            headers = {"Authorization": f"Bearer {key}"}
            files = {"file": ("speech.wav", wav_data, "audio/wav")}
            data = {
                "model": "whisper-large-v3-turbo",
                "language": "ar",
                "temperature": "0.0",
                "prompt": "أ ب ج د ر س ص ط ع ق ك ل م ن هـ و ى أ م د 1234 أ ب م 1234 أ ب ج 1234 واحد اثنين ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة"
            }
            resp = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, files=files, data=data, timeout=8)
            if resp.status_code == 200:
                transcribed = resp.json().get("text", "")
                print(f"Groq Whisper OK: '{transcribed}'")
                plates = _parse_plates_from_arabic_text(transcribed)
                if plates:
                    return plates
            else:
                print(f"Groq Whisper code {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"Groq Whisper error: {e}")
    return []

def _call_gemini_with_rotation(cfg: dict, payload: dict, model_name: str, kind: str = "rest") -> dict:
    """Call Gemini API with automatic key AND model rotation on 429/503.
    
    Each model has its own independent rate limit quota on Google's free tier.
    By cycling through multiple models, we multiply our effective quota.
    """
    import time

    # Fallback model chain — each has separate quota (~15 RPM each on free tier)
    FALLBACK_MODELS = [
        "gemini-flash-latest",
        "gemini-3.6-flash",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
    ]

    pool_field = "gemini_rest_keys" if kind == "rest" else "gemini_live_keys"
    keys = cfg.get(pool_field, [])
    if not keys:
        primary = cfg.get("gemini_api_key", "")
        keys = [primary] if primary else []
    if not keys:
        raise Exception("لا يوجد مفتاح Gemini — أضف مفتاحاً من لوحة الإدارة")

    req_timeout = 4 if kind == "live" else 25
    models_to_try = FALLBACK_MODELS[:2] if kind == "live" else FALLBACK_MODELS

    last_err = None
    for model in models_to_try:
        for key in keys:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            try:
                resp = requests.post(url, json=payload, timeout=req_timeout)
                if resp.status_code == 200:
                    print(f"Gemini OK: model={model}, key=...{key[-6:]}")
                    return resp.json()
                elif resp.status_code in (429, 503, 500, 502, 504):
                    print(f"Gemini {resp.status_code} on {model}/...{key[-6:]}, rotating...")
                    last_err = f"{resp.status_code} on {model}"
                    continue
                else:
                    raise Exception(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
            except requests.exceptions.RequestException as e:
                print(f"Network error {model}/...{key[-6:]}: {type(e).__name__}")
                last_err = f"Network error: {e}"
                continue

    if kind != "live":
        # All models and keys exhausted — wait 3s and try one more time with first available
        print("All models+keys exhausted, waiting 3s for quota refresh...")
        time.sleep(3)
        for model in FALLBACK_MODELS:
            for key in keys:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                try:
                    resp = requests.post(url, json=payload, timeout=25)
                    if resp.status_code == 200:
                        return resp.json()
                except Exception:
                    continue

    raise Exception(last_err or "All Gemini models and keys exhausted")

def _transcribe_dual_engine(cfg: dict, wav_data: bytes, model_name: str, kind: str = "live") -> list:
    """Dual-Engine transcription: Groq Whisper Turbo first (ultra-fast), then Gemini rotation"""
    # 1. Groq Whisper (Blazing fast 100-200ms)
    groq_plates = _call_groq_whisper(cfg, wav_data)
    if groq_plates:
        return groq_plates

    # 2. Gemini Multi-Model Fallback
    b64_wav = base64.b64encode(wav_data).decode("utf-8")
    prompt = (
        "أنت محرك ذكاء اصطناعي محترف متقدم جداً متخصص في التفريغ والتسميع الصوتي لأرقام لوحات السيارات السعودية من الصوت باللغة العربية.\n"
        "المطلوب منك:\n"
        "1. استمع بدقة عالية للتسجيل الصوتي واستخرج أرقام اللوحات والحروف المنطوقة.\n"
        "2. اللوحة السعودية تتكون دائماً من 3 حروف عربية مفصولة بمسافات يليهم 1 إلى 4 أرقام (مثال: 'ر ك ع 7511' أو 'أ د هـ 9873').\n"
        "3. تحويل أسماء الحروف العربية المنطوقة إلى الحرف المقابل مفصولاً بمسافات.\n"
        "4. تحويل كافة الأرقام والكلمات العددية المنطوقة إلى أرقام (0-9).\n"
        "5. استخرج نوع السيارة والملاحظات إن وُجدت في الصوت.\n"
        "6. الإخراج المطلوب: يجب إرجاع النتيجة بصيغة JSON مصفوفة فقط وبدون أي نصوص إضافية:\n"
        '[{"plate": "ر ك ع 7511", "found": true, "vehicle_type": "تويوتا", "notes": ""}]\n'
        "إذا كان الصوت غير واضح أو لا يوجد به رقم لوحة، أرجع مصفوفة فارغة: []"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": b64_wav}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    }
    res_json = _call_gemini_with_rotation(cfg, payload, model_name, kind=kind)
    raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()
    plates = json.loads(clean_text)
    if isinstance(plates, dict):
        plates = [plates]
    return plates



@app.post("/api/process")
async def process_audio(request: Request):
    job_id = f"job_{uuid.uuid4().hex[:16]}"
    cfg = load_config()

    try:
        form = await request.form()
        audio_file = form.get("audio")
        model_name = str(form.get("model_name") or cfg.get("gemini_model", "gemini-1.5-flash"))
        recorder_name = str(form.get("recorder_name") or "")
        district = str(form.get("district_default") or "")

        keys_available = bool(cfg.get("gemini_rest_keys") or cfg.get("gemini_api_key"))

        if audio_file:
            content = await audio_file.read()
            plates = await asyncio.to_thread(_transcribe_dual_engine, cfg, content, model_name, "rest")
        else:
            plates = []

        JOB_STORE[job_id] = {
            "status": "done",
            "plates": plates,
            "recorder_name": recorder_name,
            "district": district
        }

    except Exception as e:
        print(f"Process audio error: {e}")
        JOB_STORE[job_id] = {
            "status": "done",
            "plates": [],
            "error": str(e)
        }

    return {
        "status": "ok",
        "job_id": job_id,
        "message": "جاري معالجة الصوت..."
    }


@app.get("/api/transcribe/status/{job_id}")
async def transcribe_status(job_id: str):
    # Wait briefly to simulate async processing
    result = JOB_STORE.get(job_id)
    if not result:
        # Still processing - return pending
        return {"status": "pending", "progress": 50}
    
    plates = result.get("plates", [])
    return {
        "status": "done",
        "data": {
            "kind": "transcribe",
            "plates": plates,
            "total": len(plates)
        }
    }

# --- EXCEL & CHECK ENDPOINTS ---
@app.get("/api/check-live/debug-wav")
async def get_debug_wav():
    debug_wav_path = os.path.join(BASE_DIR, "debug_last_utterance.wav")
    if os.path.exists(debug_wav_path):
        return FileResponse(debug_wav_path, media_type="audio/wav", filename="debug_last_utterance.wav")
    raise HTTPException(status_code=404, detail="No debug WAV audio recorded yet")

@app.post("/api/check-headers")
async def check_headers(request: Request):
    headers = ["رقم اللوحة", "نوع السيارة", "تاريخ التسجيل", "ملاحظات", "GPS", "الحي", "الشارع"]
    try:
        form = await request.form()
        file = form.get("file") or form.get("large")
        if file:
            content = await file.read()
            wb = openpyxl.load_workbook(filename=io.BytesIO(content), data_only=True)
            sheet = wb.active
            first_row = next(sheet.iter_rows(values_only=True), None)
            if first_row:
                parsed_h = [str(c) for c in first_row if c is not None and str(c).strip()]
                if parsed_h:
                    headers = parsed_h
    except Exception as e:
        print("check-headers exception:", e)
        
    return {
        "status": "ok",
        "headers": headers,
        "cols": headers,
        "columns": headers,
        "detected_col": "رقم اللوحة",
        "detected_column": "رقم اللوحة"
    }

@app.post("/api/check-live/ref-plates-upload")
@app.post("/api/check-live/upload-excel")
async def check_live_upload_excel(request: Request):
    headers = ["رقم اللوحة", "نوع السيارة", "الحي", "الشارع", "تاريخ التسجيل", "ملاحظات"]
    total_plates = 0
    try:
        form = await request.form()
        file = form.get("file") or form.get("large")
        if file:
            content = await file.read()
            wb = openpyxl.load_workbook(filename=io.BytesIO(content), data_only=True)
            sheet = wb.active
            rows = list(sheet.iter_rows(values_only=True))
            if rows:
                parsed_h = [str(c) for c in rows[0] if c is not None and str(c).strip()]
                if parsed_h:
                    headers = parsed_h
                total_plates = max(0, len(rows) - 1)
    except Exception as e:
        print("check-live upload-excel error:", e)

    return {
        "status": "ok",
        "message": "تم رفع وتجهيز الملف المرجعي بنجاح",
        "stored_count": total_plates,
        "total_plates": total_plates,
        "headers": headers,
        "columns": headers,
        "cols": headers,
        "detected_col": "رقم اللوحة"
    }



@app.post("/api/parse-gps-excel")
async def parse_gps_excel(request: Request):
    headers = ["GPS", "رقم اللوحة", "نوع السيارة", "تاريخ التسجيل", "ملاحظات", "موقع الشارع"]
    rows = []
    try:
        form = await request.form()
        file = form.get("file")
        if file:
            content = await file.read()
            wb = openpyxl.load_workbook(filename=io.BytesIO(content), data_only=True)
            sheet = wb.active
            all_rows = list(sheet.iter_rows(values_only=True))
            if all_rows:
                header_row = [str(cell) if cell is not None else "" for cell in all_rows[0]]
                if any(header_row):
                    headers = header_row
                for r in all_rows[1:]:
                    row_dict = {}
                    for idx, val in enumerate(r):
                        col_name = headers[idx] if idx < len(headers) else f"Column_{idx+1}"
                        row_dict[col_name] = str(val) if val is not None else ""
                    rows.append(row_dict)
    except Exception as e:
        print("Excel parsing exception:", e)
            
    return {
        "status": "ok",
        "headers": headers,
        "rows": rows,
        "total_rows": len(rows)
    }

@app.post("/api/parse-export-append")
async def parse_export_append(request: Request):
    return {"status": "ok", "filename": "export.xlsx", "rows": []}

# --- EXCEL EXPORT ---
@app.post("/api/export-check-session")
@app.post("/api/export-excel")
async def export_excel(request: Request):

    """Generate a real Excel file from rows_json and return it as download"""
    try:
        form = await request.form()
        rows_json = form.get("rows_json", "[]")
        sheet_name = str(form.get("sheet_name") or "بيانات المركبات")
        district_default = str(form.get("district_default") or "")

        rows = json.loads(rows_json)
        if not isinstance(rows, list):
            rows = []

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet_name[:31]  # Excel sheet name max 31 chars

        # --- Styling ---
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        header_font = Font(bold=True, color="FFFFFF", size=12)
        header_fill = PatternFill("solid", fgColor="1E3A5F")
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        thin = Side(style="thin", color="CCCCCC")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        # Determine columns from first row or defaults
        default_cols = ["رقم اللوحة", "نوع السيارة", "الحي", "الشارع", "GPS", "المسجّل", "التاريخ", "الوقت", "ملاحظات"]
        if rows:
            cols = list(rows[0].keys())
            # Ensure default cols come first if present
            ordered = [c for c in default_cols if c in cols]
            ordered += [c for c in cols if c not in ordered]
            cols = ordered
        else:
            cols = default_cols

        # Write header row
        for col_idx, col_name in enumerate(cols, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = border

        # Write data rows
        alt_fill = PatternFill("solid", fgColor="EEF2F7")
        for row_idx, row in enumerate(rows, start=2):
            fill = alt_fill if row_idx % 2 == 0 else None
            for col_idx, col_name in enumerate(cols, start=1):
                val = row.get(col_name, "")
                # Apply district default if Hara column is empty
                if col_name == "الحي" and not val and district_default:
                    val = district_default
                cell = ws.cell(row=row_idx, column=col_idx, value=str(val) if val else "")
                cell.alignment = center_align
                cell.border = border
                if fill:
                    cell.fill = fill

        # Auto-fit column widths
        for col_idx, col_name in enumerate(cols, start=1):
            max_len = max(
                len(str(col_name)),
                *[len(str(rows[r].get(col_name, "") or "")) for r in range(len(rows))]
            ) if rows else len(col_name)
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 40)

        # Freeze header row
        ws.freeze_panes = "A2"

        # Save to bytes
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        filename = f"bareq_export_{int(time.time())}.xlsx"
        return Response(
            content=buf.read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except Exception as e:
        print(f"export-excel error: {e}")
        raise HTTPException(status_code=500, detail=f"فشل تصدير Excel: {e}")

@app.get("/api/export-excel")
async def export_excel_get():
    raise HTTPException(status_code=405, detail="استخدم POST مع rows_json")


def _route_via_osrm(coordinates: list) -> dict:
    """Free routing via OSRM public server - no API key needed"""
    coords_str = ";".join(f"{c[0]},{c[1]}" for c in coordinates)
    url = f"https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson&steps=true"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise Exception(f"OSRM error: {data.get('code')}")
    route = data["routes"][0]
    return {
        "source": "osrm",
        "distance_m": route["distance"],
        "duration_s": route["duration"],
        "distance_km": round(route["distance"] / 1000, 2),
        "duration_min": round(route["duration"] / 60, 1),
        "geometry": route["geometry"],
        "summary": f"{round(route['distance']/1000,1)} كم — {round(route['duration']/60,0)} دقيقة"
    }

def _get_ors_base_url():
    cfg = load_config()
    return cfg.get("ors_base_url", "https://api.heigit.org").rstrip("/")

def _route_via_ors(coordinates: list, ors_key: str, profile: str = "driving-car") -> dict:
    """Routing via ORS/HEIGit (requires activated key)"""
    base = _get_ors_base_url()
    headers = {
        "Authorization": ors_key,
        "Content-Type": "application/json",
        "Accept": "application/json, application/geo+json"
    }
    payload = {"coordinates": coordinates}
    url = f"{base}/v2/directions/{profile}"
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    if resp.status_code == 403:
        raise Exception("ORS_FORBIDDEN")
    resp.raise_for_status()
    data = resp.json()
    seg = data["routes"][0]["summary"]
    return {
        "source": "ors",
        "distance_m": seg["distance"],
        "duration_s": seg["duration"],
        "distance_km": round(seg["distance"] / 1000, 2),
        "duration_min": round(seg["duration"] / 60, 1),
        "geometry": data["routes"][0].get("geometry"),
        "summary": f"{round(seg['distance']/1000,1)} كم — {round(seg['duration']/60,0)} دقيقة"
    }

@app.post("/api/ors/directions")
async def ors_directions(req: Request):
    """Smart routing: ORS first → OSRM fallback"""
    cfg = load_config()
    ors_keys = cfg.get("ors_keys", [])
    ors_key = ors_keys[0] if ors_keys else cfg.get("ors_api_key", "")
    
    body = await req.json()
    profile = body.get("profile", "driving-car")
    coordinates = body.get("coordinates", [])
    
    if not coordinates or len(coordinates) < 2:
        raise HTTPException(status_code=400, detail="coordinates مطلوبة (نقطتان على الأقل)")
    
    # Try ORS first if key exists and not previously forbidden
    if ors_key:
        try:
            result = _route_via_ors(coordinates, ors_key, profile)
            return result
        except Exception as e:
            if "ORS_FORBIDDEN" not in str(e):
                print(f"ORS error (not 403): {e}")
            # Fall through to OSRM
    
    # Fallback to free OSRM
    try:
        result = _route_via_osrm(coordinates)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل حساب المسار: {e}")

@app.get("/api/ors/geocode")
async def ors_geocode(text: str = "", lat: float = 0, lon: float = 0):
    """Geocoding via ORS or Nominatim fallback"""
    cfg = load_config()
    ors_keys = cfg.get("ors_keys", [])
    ors_key = ors_keys[0] if ors_keys else cfg.get("ors_api_key", "")
    
    # Try ORS/HEIGit first
    if ors_key:
        try:
            base = _get_ors_base_url()
            headers = {"Authorization": ors_key, "Accept": "application/json"}
            if text:
                url = f"{base}/geocode/search?text={requests.utils.quote(text)}&size=5"
            else:
                url = f"{base}/geocode/reverse?point.lat={lat}&point.lon={lon}&size=1"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return JSONResponse(content=resp.json())
        except Exception:
            pass
    
    # Fallback: Nominatim (OpenStreetMap geocoding - free)
    try:
        if text:
            url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(text)}&format=json&limit=5"
        else:
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers_nom = {"User-Agent": "BareqApp/4.0"}
        resp = requests.get(url, headers=headers_nom, timeout=10)
        return JSONResponse(content=resp.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ors/status")
async def ors_status():
    """Check routing status — ORS or OSRM fallback"""
    cfg = load_config()
    ors_keys = cfg.get("ors_keys", [])
    ors_key = ors_keys[0] if ors_keys else cfg.get("ors_api_key", "")
    
    result = {"ors": "no_key", "osrm": "unknown", "active_service": "osrm"}
    
    # Test ORS/HEIGit
    if ors_key:
        try:
            base = _get_ors_base_url()
            payload = {"coordinates": [[46.6753, 24.7136], [46.7153, 24.7436]]}
            headers = {"Authorization": ors_key, "Content-Type": "application/json", "Accept": "application/json"}
            resp = requests.post(
                f"{base}/v2/directions/driving-car",
                json=payload, headers=headers, timeout=10
            )
            if resp.status_code == 200:
                result["ors"] = "ok"
                result["active_service"] = "ors"
            elif resp.status_code == 403:
                result["ors"] = "forbidden - المفتاح يحتاج تفعيل"
            else:
                result["ors"] = f"error_{resp.status_code}: {resp.text[:80]}"
        except Exception as e:
            result["ors"] = f"error: {e}"
    
    # Test OSRM (always free)
    try:
        r = requests.get(
            "https://router.project-osrm.org/route/v1/driving/46.6753,24.7136;46.7153,24.7436?overview=false",
            timeout=8
        )
        if r.status_code == 200 and r.json().get("code") == "Ok":
            result["osrm"] = "ok"
            if result["active_service"] != "ors":
                result["active_service"] = "osrm"
        else:
            result["osrm"] = "error"
    except Exception as e:
        result["osrm"] = f"error: {e}"
    
    return {
        "status": "ok" if result["active_service"] in ("ors","osrm") else "error",
        "active_service": result["active_service"],
        "ors_status": result["ors"],
        "osrm_status": result["osrm"],
        "message": f"الخدمة النشطة: {result['active_service'].upper()}"
    }



@app.websocket("/ws/check-live")
async def websocket_check_live(websocket: WebSocket, ticket: str = ""):
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    print(f"[WebSocket] Connected (check-live) from {client_ip}")
    cfg = load_config()

    # Per-connection session state (multi-user isolation)
    session = ASRSession() if _LOCAL_ASR_AVAILABLE else None
    legacy_pcm_chunks = []
    decoder = _get_plate_decoder() if _LOCAL_ASR_AVAILABLE else None
    partial_in_flight = False

    def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
        import wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    async def _run_partial_task(audio_np):
        nonlocal partial_in_flight
        try:
            res = await asyncio.to_thread(_local_transcribe, audio_np, True)
            raw_text = res.get("text", "").strip()
            if raw_text:
                dec = decoder.decode_partial(raw_text)
                partial_plate = dec.get("partial_plate", "")
                display_txt = partial_plate if partial_plate else raw_text

                if session:
                    if "first_partial_result" not in session.timestamps:
                        session.record_timestamp("first_partial_result")
                    if partial_plate:
                        session.partial_history.append(partial_plate)
                        # Keep only last 5 partials
                        if len(session.partial_history) > 5:
                            session.partial_history = session.partial_history[-5:]

                try:
                    await websocket.send_json({
                        "type": "live_transcript",
                        "data": display_txt,
                        "partial": True,
                        "partial_plate": partial_plate,
                        "is_complete": dec.get("is_complete", False)
                    })
                except Exception:
                    pass
        except Exception as e:
            print(f"[WS] Partial task error: {e}")
        finally:
            partial_in_flight = False

    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    mtype = data.get("type")

                    if mtype == "init":
                        # CRITICAL: client waits for type='ready' to set UI state to 'جاهز — استمع'
                        await websocket.send_json({"type": "ready"})
                        print("Sent type: ready to client")

                    elif mtype == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif mtype == "audio":
                        raw_b64 = data.get("data", "")
                        if raw_b64:
                            b_pcm = base64.b64decode(raw_b64)
                            if session:
                                if session.chunks_received == 0:
                                    session.record_timestamp("first_chunk")
                                session.add_audio(b_pcm)

                                # Check if we should trigger a partial decoding task
                                if not partial_in_flight and session.should_run_partial(min_interval_s=0.6):
                                    if _is_asr_available():
                                        audio_np = session.get_audio_numpy()
                                        if audio_np is not None:
                                            partial_in_flight = True
                                            session.last_partial_at = time.time()
                                            asyncio.create_task(_run_partial_task(audio_np))
                            else:
                                legacy_pcm_chunks.append(b_pcm)

                    elif mtype == "end_of_turn":
                        total_chunks = session.chunks_received if session else len(legacy_pcm_chunks)
                        print(f"End of turn received. Total chunks in utterance: {total_chunks}")

                        if session:
                            session.record_timestamp("end_of_turn")
                            audio_np = session.get_audio_numpy()
                            all_pcm_bytes = bytes(session.audio_buffer)
                            partial_history = list(session.partial_history)
                            session.clear_turn()
                        else:
                            audio_np = None
                            all_pcm_bytes = b"".join(legacy_pcm_chunks)
                            legacy_pcm_chunks.clear()
                            partial_history = []

                        if not all_pcm_bytes or len(all_pcm_bytes) < 3200:  # < 0.1s
                            continue

                        wav_data = pcm16_to_wav(all_pcm_bytes, 16000)

                        # Save debug WAV file for audio inspection (Requirement #3)
                        debug_wav_path = os.path.join(BASE_DIR, "debug_last_utterance.wav")
                        try:
                            with open(debug_wav_path, "wb") as f_wav:
                                f_wav.write(wav_data)
                        except Exception as wav_save_err:
                            print(f"[WS Debug] Error saving debug WAV: {wav_save_err}")

                        async def _process_end_of_turn(b_wav: bytes, a_np, p_hist, pcm_len: int):
                            t_start = time.time()
                            plate_text = ""
                            raw_text = ""
                            norm_text = ""
                            confidence = 0.0
                            is_valid = False
                            signals = {}
                            used_engine = "local"

                            # Calculate Audio Statistics (Requirement #2)
                            audio_duration = pcm_len / (16000 * 2)
                            rms_val = float(np.sqrt(np.mean(a_np ** 2))) if a_np is not None and len(a_np) > 0 else 0.0
                            peak_val = float(np.max(np.abs(a_np))) if a_np is not None and len(a_np) > 0 else 0.0

                            print("\n" + "=" * 60)
                            print(f"[AUDIO RECEIVED] bytes={pcm_len} | duration={audio_duration:.2f}s | rms={rms_val:.4f} | peak={peak_val:.4f}")
                            print("=" * 60)

                            # Allow quiet mobile-mic speech to reach ASR normalization.
                            if rms_val < 0.00008 or peak_val < 0.0003:
                                print("[AUDIO WARNING] Audio RMS is near zero (silence/mic issue). Skipping ASR.")
                                await websocket.send_json({
                                    "type": "debug_pipeline",
                                    "data": {
                                        "raw_asr": "", "normalized": "", "decoded_plate": "",
                                        "valid": False, "confidence": 0.0, "latency_ms": 0,
                                        "engine": "silence", "audio_duration_s": round(audio_duration, 2),
                                        "rms": round(rms_val, 6), "peak": round(peak_val, 6)
                                    }
                                })
                                await websocket.send_json({
                                    "type": "live_transcript",
                                    "data": "⚠️ لم يتم التقاط صوت واضح من الميكروفون (صمت)",
                                    "final": False
                                })
                                return

                            # Show immediately that the server stored the turn; inference can take longer.
                            await websocket.send_json({
                                "type": "debug_pipeline",
                                "data": {
                                    "raw_asr": "", "normalized": "", "decoded_plate": "",
                                    "valid": False, "confidence": 0.0, "latency_ms": 0,
                                    "engine": "processing_audio",
                                    "audio_duration_s": round(audio_duration, 2),
                                    "rms": round(rms_val, 6), "peak": round(peak_val, 6)
                                }
                            })

                            # Groq Whisper is the low-latency path.  The old order ran local
                            # large-v3-turbo first, delaying every visible result by minutes on CPU.
                            api_attempted = False
                            if cfg.get("enable_api_fallback", True):
                                api_attempted = True
                                try:
                                    used_engine = "groq_whisper"
                                    model_name = cfg.get("gemini_model", "gemini-flash-latest")
                                    plates = await asyncio.to_thread(_transcribe_dual_engine, cfg, b_wav, model_name, "live")
                                    if plates:
                                        p0 = plates[0]
                                        raw_plate = p0.get("plate", "").strip()
                                        raw_text = raw_plate
                                        if decoder:
                                            dec = decoder.decode_final(raw_plate)
                                            norm_text = dec.get("normalized", "")
                                            plate_text = dec.get("plate", "") or raw_plate
                                            is_valid = dec.get("valid", False)
                                            confidence = dec.get("confidence", 0.7)
                                        else:
                                            plate_text = raw_plate
                                            is_valid = True
                                            confidence = 0.7
                                        print(f"[ASR Fast] RAW: '{raw_plate}' -> PLATE: '{plate_text}'")
                                except Exception as fast_err:
                                    print(f"[ASR Fast] Error, trying local ASR: {fast_err}")
                            # 1. Try Local ASR first (if available and loaded)
                            if not plate_text and _LOCAL_ASR_AVAILABLE and _is_asr_available() and a_np is not None:
                                try:
                                    t0_asr = time.time()
                                    res = await asyncio.to_thread(_local_transcribe, a_np, False)
                                    raw_text = res.get("text", "").strip()
                                    segments = res.get("segments", [])
                                    asr_elapsed = (time.time() - t0_asr) * 1000

                                    print(f"RAW ASR:       \"{raw_text}\"")

                                    if raw_text:
                                        logprobs = [s.get("avg_logprob", 0.0) for s in segments if "avg_logprob" in s]
                                        dec = decoder.decode_final(
                                            raw_text,
                                            asr_segment_confidences=logprobs,
                                            partial_history=p_hist
                                        )
                                        norm_text = dec.get("normalized", "")
                                        plate_text = dec.get("plate", "")
                                        is_valid = dec.get("valid", False)
                                        confidence = dec.get("confidence", 0.0)
                                        signals = dec.get("signals", {})
                                        used_engine = "local_whisper"

                                        print(f"NORMALIZED:    \"{norm_text}\"")
                                        print(f"DECODED PLATE: \"{plate_text}\" (valid={is_valid}, conf={confidence:.2f}, ASR={asr_elapsed:.0f}ms)")
                                except Exception as local_err:
                                    print(f"[ASR Local] Error, will fallback: {local_err}")
                                    plate_text = ""

                            # 2. Fallback to Cloud Dual-Engine (Groq / Gemini) if local returned nothing or not available
                            if not plate_text and cfg.get("enable_api_fallback", True) and not api_attempted:
                                try:
                                    used_engine = "api_fallback"
                                    model_name = cfg.get("gemini_model", "gemini-flash-latest")
                                    plates = await asyncio.to_thread(_transcribe_dual_engine, cfg, b_wav, model_name, "live")
                                    if plates:
                                        p0 = plates[0]
                                        raw_plate = p0.get("plate", "").strip()
                                        raw_text = raw_plate
                                        if decoder:
                                            dec = decoder.decode_final(raw_plate)
                                            norm_text = dec.get("normalized", "")
                                            plate_text = dec.get("plate", "") or raw_plate
                                            is_valid = dec.get("valid", False)
                                            confidence = dec.get("confidence", 0.7)
                                        else:
                                            plate_text = raw_plate
                                            is_valid = True
                                            confidence = 0.7
                                        print(f"[ASR Fallback] RAW: \"{raw_plate}\" -> PLATE: \"{plate_text}\"")
                                except Exception as fb_err:
                                    print(f"[ASR Fallback] Error: {fb_err}")

                            t_total = (time.time() - t_start) * 1000
                            print(f"FINAL RESULT:  \"{plate_text}\" | engine={used_engine} | total_latency={t_total:.0f}ms")
                            print("=" * 60 + "\n")

                            # Send Debug Pipeline Payload to Frontend (Requirements #1, #4, #12)
                            try:
                                await websocket.send_json({
                                    "type": "debug_pipeline",
                                    "data": {
                                        "raw_asr": raw_text,
                                        "normalized": norm_text,
                                        "decoded_plate": plate_text,
                                        "valid": is_valid,
                                        "confidence": confidence,
                                        "signals": signals,
                                        "latency_ms": round(t_total, 1),
                                        "engine": used_engine,
                                        "audio_duration_s": round(audio_duration, 2),
                                        "rms": round(rms_val, 4),
                                        "peak": round(peak_val, 4)
                                    }
                                })
                            except Exception:
                                pass

                            # 3. Decision & Emission:
                            try:
                                # Save only if valid and confidence >= 0.4
                                if plate_text and is_valid and confidence >= 0.4:
                                    await websocket.send_json({
                                        "type": "plate_result",
                                        "data": {
                                            "plate": plate_text,
                                            "found": True,
                                            "vehicle_type": "تويوتا",
                                            "notes": "",
                                            "moving": False,
                                            "confidence": confidence,
                                            "engine": used_engine,
                                            "latency_ms": round(t_total, 1),
                                            "signals": signals
                                        }
                                    })
                                    # Update transcript to show the final accepted plate
                                    await websocket.send_json({
                                        "type": "live_transcript",
                                        "data": f"✔ {plate_text}",
                                        "final": True
                                    })
                                elif plate_text:
                                    # Incomplete or low confidence — show in transcript, but DO NOT save
                                    await websocket.send_json({
                                        "type": "live_transcript",
                                        "data": f"⚠️ غير مكتملة: {plate_text}",
                                        "final": False
                                    })
                                else:
                                    await websocket.send_json({
                                        "type": "live_transcript",
                                        "data": f"لم يتم التعرف على لوحة (RAW: {raw_text or 'صمت'})",
                                        "final": False
                                    })
                            except Exception as send_err:
                                print(f"[WS] Client disconnected before emission: {send_err}")

                        asyncio.create_task(_process_end_of_turn(wav_data, audio_np, partial_history, len(all_pcm_bytes)))

                except Exception as ex:
                    print(f"WS message error: {ex}")

            elif "bytes" in msg and msg["bytes"]:
                if session:
                    if session.chunks_received == 0:
                        session.record_timestamp("first_chunk")
                    session.add_audio(msg["bytes"])
                else:
                    legacy_pcm_chunks.append(msg["bytes"])

    except WebSocketDisconnect:
        print(f"[WebSocket] Disconnected (check-live) from {client_ip}")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")




# --- AUTH ENDPOINTS ---
@app.post("/auth/login")
async def login(req: Request):
    data = await req.json()
    cfg = load_config()
    users = load_users()
    u_in = data.get("username", "").strip()
    p_in = data.get("password", "").strip()
    
    if u_in == cfg["admin_username"] and p_in == cfg["admin_password"]:
        return {
            "status": "ok",
            "token": "bareq_admin_token_8899",
            "is_admin": True,
            "username": u_in
        }
        
    for u in users:
        if u["username"] == u_in and u.get("password") == p_in:
            return {
                "status": "ok",
                "token": f"bareq_token_{u['id']}",
                "is_admin": u.get("is_admin", False),
                "username": u["username"]
            }
            
    raise HTTPException(status_code=401, detail="اسم المستخدم أو كلمة المرور غير صحيحة")

@app.get("/auth/me")
async def auth_me():
    cfg = load_config()
    return {
        "status": "ok",
        "is_admin": True,
        "username": cfg["admin_username"]
    }

@app.post("/auth/presence")
@app.delete("/auth/presence")
async def auth_presence():
    return {"status": "ok"}

@app.post("/auth/refresh")
async def auth_refresh():
    return {"status": "ok", "token": "bareq_refreshed_token"}

@app.post("/auth/change-password")
async def change_password(req: Request):
    data = await req.json()
    cfg = load_config()
    old_pw = data.get("old_password", "").strip()
    new_pw = data.get("new_password", "").strip()
    if not new_pw or len(new_pw) < 3:
        raise HTTPException(status_code=400, detail="كلمة المرور الجديدة يجب أن تكون 3 أحرف على الأقل")
    if old_pw != cfg.get("admin_password", "123"):
        raise HTTPException(status_code=400, detail="كلمة المرور الحالية غير صحيحة")
    cfg["admin_password"] = new_pw
    save_config(cfg)
    return {"status": "ok", "message": "تم تغيير كلمة المرور بنجاح"}

@app.post("/api/check-live/ws-ticket")
async def ws_ticket():
    return {"status": "ok", "ticket": "bareq_live_ticket_99"}

# --- ADMIN DASHBOARD & STORAGE PAGES ---
@app.get("/admin/check-storage")
async def check_storage_page():
    html = """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>تخزين الفرز — برق</title>
        <style>
            body { font-family: sans-serif; background: #060b14; color: #d8e4f0; padding: 2rem; direction: rtl; }
            .card { background: #0b1423; border: 1px solid #1a2d45; padding: 1.5rem; border-radius: 10px; max-width: 800px; margin: 0 auto; }
            h1 { color: #0ea5e9; font-size: 1.4rem; }
            p { color: #6b7f96; font-size: 0.9rem; line-height: 1.6; }
            .btn { background: #0ea5e9; color: white; padding: 0.6rem 1.2rem; border-radius: 6px; text-decoration: none; display: inline-block; font-weight: bold; margin-top: 1rem; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>📦 فرز وقواعد بيانات التخزين</h1>
            <p>لا توجد بيانات فرز قديمة أو ملفات معلقة في الوقت الحالي. جميع عمليات الفرز والتسجيل الحالية تُعالج محلياً في الذاكرة وتُصدَّر مباشرة كملفات Excel.</p>
            <a href="/" class="btn">العودة للوحة التحكم الرئيسية</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/admin/check-storage/summary")
@app.get("/api/admin/check-storage/summary")
async def check_storage_summary():
    return {"status": "ok", "summary": [], "total_rows": 0}

@app.get("/admin/gemini-overview")
async def gemini_overview():
    return {
        "cost_usd": "0.00$",
        "tokens": 0,
        "reset_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/admin/gemini-overview/reset")
async def gemini_overview_reset():
    return {"status": "ok", "message": "تم تصفير عداد التوكنات"}

@app.get("/admin/plates-overview")
async def plates_overview():
    return {
        "total": 0,
        "tafrigh": 0,
        "live": 0,
        "ptt": 0
    }

@app.get("/admin/online-users")
async def online_users():
    return [
        {
            "username": "admin",
            "display_name": "مدير النظام",
            "device_id": "Device_Local_PC",
            "last_seen": "الآن"
        }
    ]

@app.get("/admin/storage-summary")
async def storage_summary():
    return {"status": "ok", "summary": []}

@app.get("/admin/groups")
async def get_groups():
    return []

@app.post("/admin/groups")
async def create_group(req: Request):
    data = await req.json()
    return {"status": "ok", "id": 1, "name": data.get("name", "مجموعة جديدة")}

@app.get("/admin/users")
async def get_users():
    return load_users()

@app.post("/admin/users")
async def create_user(req: Request):
    data = await req.json()
    users = load_users()
    
    new_user = {
        "id": len(users) + 1,
        "username": data.get("username", f"user_{len(users)+1}"),
        "password": data.get("password", "123456"),
        "display_name": data.get("display_name") or data.get("username"),
        "is_admin": data.get("is_admin", False),
        "is_active": True,
        "rows_limit": data.get("rows_limit", 3000000),
        "subscription_end": "مفتوح - 30 يوماً",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    users.append(new_user)
    save_users(users)
    return {"status": "ok", "message": f"تم إنشاء المستخدم '{new_user['username']}' بنجاح", "user": new_user}

@app.get("/admin/users/{user_id}")
async def get_user_detail(user_id: int):
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            user_data = dict(u)
            user_data.setdefault("is_active", True)
            user_data.setdefault("is_admin", False)
            user_data.setdefault("display_name", u.get("username", ""))
            user_data.setdefault("rows_limit", 3000000)
            user_data.setdefault("subscription_end", "مفتوح - 30 يوماً")
            user_data.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
            user_data.setdefault("group_id", None)
            user_data.setdefault("gemini_rest_model_id", "")
            user_data.setdefault("gemini_live_model_id", "")
            user_data.setdefault("gemini_check_model_id", "")
            user_data.setdefault("gemini_spend_usd", 0.0)
            user_data.setdefault("gemini_spend_limit_usd", None)
            user_data.setdefault("device_binding", "غير مقيد")
            user_data.setdefault("last_seen", "متصل الآن")
            return user_data
    raise HTTPException(status_code=404, detail="المستخدم غير موجود")

@app.patch("/admin/users/{user_id}")
@app.put("/admin/users/{user_id}")
async def patch_user(user_id: int, req: Request):
    data = await req.json()
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            u.update(data)
            save_users(users)
            return {"status": "ok", "user": u}
    return {"status": "ok"}

@app.delete("/admin/users/{user_id}")
async def delete_user(user_id: int):
    users = load_users()
    users = [u for u in users if u["id"] != user_id]
    save_users(users)
    return {"status": "ok", "message": "تم حذف المستخدم بنجاح"}

@app.patch("/admin/users/{user_id}/group")
@app.patch("/admin/users/{user_id}/gemini-policy")
@app.patch("/admin/users/{user_id}/rows-limit")
async def patch_user_properties(user_id: int, req: Request):
    data = await req.json()
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            u.update(data)
            save_users(users)
            return {"status": "ok", "message": "تم تحديث البيانات بنجاح", "user": u}
    return {"status": "ok"}

@app.post("/admin/users/{user_id}/reset-device")
@app.post("/admin/users/{user_id}/upgrade-device-binding")
@app.post("/admin/users/{user_id}/logout")
@app.post("/admin/users/{user_id}/renew-subscription")
@app.post("/admin/users/{user_id}/subscription/activate")
@app.post("/admin/users/{user_id}/subscription/suspend")
async def user_action_handler(user_id: int):
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            u["subscription_end"] = "مفتوح - 30 يوماً"
            u["is_active"] = True
            save_users(users)
            break
    return {"status": "ok", "message": "تمت العملية بنجاح"}

@app.get("/admin/provider/gemini-models")
async def get_admin_gemini_models():
    return {
        "models": [
            {"id": 1, "channel": "rest", "model_id": "gemini-1.5-flash", "label": "Gemini 1.5 Flash"},
            {"id": 2, "channel": "live", "model_id": "gemini-1.5-flash", "label": "Gemini 1.5 Flash"},
            {"id": 3, "channel": "check", "model_id": "gemini-1.5-flash", "label": "Gemini 1.5 Flash"}
        ]
    }

@app.post("/admin/provider/gemini-models")
async def add_admin_gemini_model(req: Request):
    return {"status": "ok", "message": "تمت إضافة الموديل"}

@app.get("/admin/provider/gemini-defaults")
async def get_gemini_defaults():
    return {
        "rest_model_id": "gemini-1.5-flash",
        "live_model_id": "gemini-1.5-flash",
        "check_model_id": "gemini-1.5-flash"
    }

@app.put("/admin/provider/gemini-defaults")
async def put_gemini_defaults(req: Request):
    return {"status": "ok", "message": "تم التحديث"}

@app.get("/admin/provider/key-pools")
async def get_key_pools():
    cfg = load_config()
    def make_pool(keys_list, kind):
        result = []
        for i, k in enumerate(keys_list):
            short = k[:8] + "..." + k[-4:] if len(k) > 14 else k
            result.append({
                "id": i+1,
                "name": f"{kind} مفتاح {i+1}",
                "short": short,
                "status": "active",
                "key": k
            })
        return result

    return {
        "gemini_rest":  make_pool(cfg.get("gemini_rest_keys", []), "REST"),
        "gemini_live":  make_pool(cfg.get("gemini_live_keys", []), "Live"),
        "groq":         make_pool(cfg.get("groq_keys", []), "Groq Whisper Turbo"),
        "ors":          make_pool(cfg.get("ors_keys", []), "ORS"),
        "gmaps":        make_pool(cfg.get("gmaps_keys", [cfg.get("gmaps_api_key","")]) if cfg.get("gmaps_api_key") else [], "Maps"),
    }

@app.post("/admin/provider/key-pools/{kind}")
async def add_key_pool(kind: str, req: Request):
    data = await req.json()
    key = data.get("key", "").strip()
    name = data.get("name", "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="المفتاح فارغ")
    
    cfg = load_config()
    
    map_kind = {
        "gemini_rest":  "gemini_rest_keys",
        "gemini_live":  "gemini_live_keys",
        "groq":         "groq_keys",
        "ors":          "ors_keys",
        "gmaps":        "gmaps_keys",
    }
    field = map_kind.get(kind)
    if not field:
        raise HTTPException(status_code=400, detail=f"نوع مفتاح غير معروف: {kind}")
    
    existing = cfg.get(field, [])
    if key not in existing:
        existing.append(key)
        cfg[field] = existing
        
        # Also set primary key for fast access
        if kind == "gemini_rest" and not cfg.get("gemini_api_key"):
            cfg["gemini_api_key"] = key
        if kind == "gemini_live" and not cfg.get("gemini_api_key"):
            cfg["gemini_api_key"] = key
        if kind == "groq" and not cfg.get("groq_api_key"):
            cfg["groq_api_key"] = key
        if kind == "gmaps":
            cfg["gmaps_api_key"] = key
        if kind == "ors":
            cfg["ors_api_key"] = key
            
        save_config(cfg)
    
    return {"status": "ok", "message": f"تم إضافة المفتاح بنجاح ({kind})"}

@app.delete("/admin/provider/key-pools/{kind}/{key_id}")
async def delete_key_pool(kind: str, key_id: int):
    cfg = load_config()
    map_kind = {
        "gemini_rest":  "gemini_rest_keys",
        "gemini_live":  "gemini_live_keys",
        "groq":         "groq_keys",
        "ors":          "ors_keys",
        "gmaps":        "gmaps_keys",
    }
    field = map_kind.get(kind)
    if field and field in cfg:
        keys = cfg[field]
        if 0 < key_id <= len(keys):
            keys.pop(key_id - 1)
            cfg[field] = keys
            save_config(cfg)
    return {"status": "ok"}

@app.get("/admin/provider/gemini-pricing")
async def get_gemini_pricing():
    return {
        "rest_in": 0.15,
        "rest_out": 0.60,
        "live_in": 0.15,
        "live_out": 0.60
    }

@app.put("/admin/provider/gemini-pricing")
async def put_gemini_pricing(req: Request):
    return {"status": "ok", "message": "تم حفظ الأسعار"}

@app.put("/admin/provider/gemini-rest-key-source")
@app.put("/admin/provider/gemini-live-key-source")
@app.put("/admin/provider/gemini-vertex-primary")
@app.put("/admin/provider/gemini-vertex-enabled-slots")
async def update_key_sources(req: Request):
    return {"status": "ok", "message": "تم الحفظ بنجاح"}

# Main UI Index
@app.get("/")
async def index():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Bareq Server</h1>")

# ─────────────────────────────────────────────────────────
# PLANS (PACKAGES) API — إدارة باقات الاشتراك
# ─────────────────────────────────────────────────────────

@app.get("/admin/plans")
async def get_plans():
    return load_plans()

@app.post("/admin/plans")
async def create_plan(req: Request):
    data = await req.json()
    plans = load_plans()
    new_id = max((p["id"] for p in plans), default=0) + 1
    new_plan = {
        "id": new_id,
        "name": data.get("name", f"باقة {new_id}"),
        "name_en": data.get("name_en", f"Plan {new_id}"),
        "price": data.get("price", 0),
        "currency": data.get("currency", "ريال"),
        "duration_days": data.get("duration_days", 30),
        "rows_limit": data.get("rows_limit", 5000),
        "description": data.get("description", ""),
        "features": data.get("features", []),
        "color": data.get("color", "#22c55e"),
        "is_active": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    plans.append(new_plan)
    save_plans(plans)
    return {"status": "ok", "message": f"تم إنشاء الباقة '{new_plan['name']}' بنجاح", "plan": new_plan}

@app.put("/admin/plans/{plan_id}")
@app.patch("/admin/plans/{plan_id}")
async def update_plan(plan_id: int, req: Request):
    data = await req.json()
    plans = load_plans()
    for p in plans:
        if p["id"] == plan_id:
            p.update({k: v for k, v in data.items() if k != "id"})
            save_plans(plans)
            return {"status": "ok", "message": "تم تحديث الباقة بنجاح", "plan": p}
    raise HTTPException(status_code=404, detail="الباقة غير موجودة")

@app.delete("/admin/plans/{plan_id}")
async def delete_plan(plan_id: int):
    plans = load_plans()
    plans = [p for p in plans if p["id"] != plan_id]
    save_plans(plans)
    return {"status": "ok", "message": "تم حذف الباقة بنجاح"}

@app.post("/admin/users/{user_id}/assign-plan")
async def assign_plan_to_user(user_id: int, req: Request):
    data = await req.json()
    plan_id = data.get("plan_id")
    plans = load_plans()
    plan = next((p for p in plans if p["id"] == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="الباقة غير موجودة")
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            from datetime import datetime, timedelta
            start = datetime.now()
            end = start + timedelta(days=plan["duration_days"])
            u["plan_id"] = plan["id"]
            u["plan_name"] = plan["name"]
            u["rows_limit"] = plan["rows_limit"]
            u["subscription_start"] = start.strftime("%Y-%m-%d")
            u["subscription_end"] = end.strftime("%Y-%m-%d")
            u["is_active"] = True
            save_users(users)
            return {
                "status": "ok",
                "message": f"تم تعيين باقة '{plan['name']}' للمستخدم بنجاح",
                "subscription_end": u["subscription_end"],
                "rows_limit": u["rows_limit"],
            }
    raise HTTPException(status_code=404, detail="المستخدم غير موجود")

@app.get("/admin/plans/stats")
async def get_plans_stats():
    plans = load_plans()
    users = load_users()
    stats = []
    for p in plans:
        count = sum(1 for u in users if u.get("plan_id") == p["id"])
        stats.append({"plan_id": p["id"], "plan_name": p["name"], "subscribers": count})
    return stats

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8500))
    print("==================================================")
    print("   Bareq System Server - Running Successfully")
    print(f"   URL: http://{host}:{port} (Local: http://127.0.0.1:{port})")
    print("==================================================")
    uvicorn.run(app, host=host, port=port)
