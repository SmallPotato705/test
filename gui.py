import os
import io
import json
import base64
import socket
import webbrowser
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import shutil
import hashlib
import time
import dash
from dash import Dash, html, dcc, Output, Input, State, ctx
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL

from pathlib import Path
from zoneinfo import ZoneInfo
import uuid
from dash import no_update
import requests
from pathlib import Path
import sys

from utils.app_index_string_css import app_index_string

from threading import Lock
RUNNER_LOCK = Lock()

IP = "http://127.0.0.1:5000"
urlGetModel = IP + "/get_models"
urlGetPrediction =  IP + "/get_prediction"
urlGetOneTest = IP + "/apc_and_ems_and_predict_Test"

INFER_CFG_NAME = "config.json"

def get_app_root() -> str:
    # 打包(EXE) 時回 EXE 所在資料夾；開發時回程式檔所在資料夾
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR     = get_app_root()
DATA_DIR    = os.path.join(APP_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DEPLOY_DIR  = os.path.join(DATA_DIR, "deploy_tasks")

# 建資料夾
for d in (DATA_DIR, UPLOADS_DIR, DEPLOY_DIR):
    os.makedirs(d, exist_ok=True)

UPLOADS_DIR_PATH = Path(UPLOADS_DIR).resolve()
DEPLOY_DIR_PATH  = Path(DEPLOY_DIR).resolve()

print("[uploads]", UPLOADS_DIR_PATH)
print("[deploy ]", DEPLOY_DIR_PATH)
# ---------------------------
# Helpers (安全相關強化)
# ---------------------------
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "5"))
REQ_BASE_TIMEOUT_SEC  = float(os.getenv("REQ_BASE_TIMEOUT_SEC", "30"))
REQ_PER_FILE_SEC      = float(os.getenv("REQ_PER_FILE_SEC", "20"))
REQ_MAX_RETRY         = int(os.getenv("REQ_MAX_RETRY", "1"))

def resource_path(relpath: str) -> str:
    # 打包(onefile)時 PyInstaller 會把資源解到 _MEIPASS
    base = getattr(sys, "_MEIPASS", APP_DIR) if getattr(sys, "frozen", False) else APP_DIR
    return os.path.join(base, relpath)


def set_task_enabled(slug: str, enabled: bool):
    cfg = read_infer_config(slug) or {}
    cfg["enabled"] = bool(enabled)
    save_infer_config(slug, cfg)

def get_task_enabled(slug: str) -> bool:
    return bool((read_infer_config(slug) or {}).get("enabled", False))

def _infer_cfg_path(slug: str) -> Path:
    p = (DEPLOY_DIR_PATH / (slug or "") / INFER_CFG_NAME).resolve()
    if not _is_safe_subpath(DEPLOY_DIR_PATH, p):
        raise ValueError("不合法的部屬任務設定路徑")
    return p

def read_infer_config(slug: str) -> dict:
    try:
        p = _infer_cfg_path(slug)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def save_infer_config(slug: str, cfg: dict) -> bool:
    try:
        p = _infer_cfg_path(slug)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg or {}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False
    
    
def _chunks(seq, size):
    seq = list(seq or [])
    for i in range(0, len(seq), max(1, size)):
        yield seq[i:i+size]

def _post_with_retry(url, json_payload, timeout, max_retry=1):
    last_err = None
    for attempt in range(max(1, max_retry) + 1):
        try:
            r = requests.post(url, json=json_payload,
                              headers={"Accept": "application/json"},
                              timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            
            if attempt < max_retry:
                time.sleep(0.6 * (2 ** attempt))
    raise last_err
    
def _is_safe_subpath(base: Path, candidate: Path) -> bool:
    base = base.resolve()
    cand = candidate.resolve()
    try:
        return os.path.commonpath([base]) == os.path.commonpath([base, cand])
    except Exception:
        return False

def _safe_clear_dir(dir_path: str, expected_parent: Path) -> bool:
    try:
        p = Path(dir_path).resolve()
        if not _is_safe_subpath(expected_parent, p):
            return False
        if p.parent.parent != expected_parent:
            return False
        if p.name not in ("normal", "abnormal"):
            return False
        slug_dir = p.parent
        if not slug_dir.is_dir() or slug_dir == expected_parent:
            return False
        shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)
        return True
    except Exception:
        return False

def _migration_marker_path(base_path: Path) -> Path:
    return base_path / ".migrated_from"

def _legacy_migration_enabled() -> bool:
    return os.getenv("DISABLE_LEGACY_MIGRATION", "0") not in ("1", "true", "True")

def _short_uid(n: int = 6) -> str:
    return uuid.uuid4().hex[:n]

def _uniqueish_task_slug(seed: str) -> str:
    s = (seed or "").strip()
    if not s or s.lower() in {"task", "untitled", "default"}:
        return f"task-{_short_uid()}"
    return s

def _sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    name = "".join(('-' if ch in bad else ch) for ch in name)
    name = name.strip().strip(".")
    return name or f"file-{_short_uid()}"

def _looks_like_legacy_id(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    try:
        if len(s) == 36 and "-" in s:
            uuid.UUID(s)
            return True
    except Exception:
        pass
    if len(s) == 32 and all(c in "0123456789abcdefABCDEF" for c in s):
        return True
    return False


def _to_float_array(x):

    import numpy as _np, json as _json
    if x is None:
        return _np.array([], dtype=float)
    if isinstance(x, (list, tuple, _np.ndarray)):
        try:
            return _np.asarray(x, dtype=float).reshape(-1)
        except Exception:
            return _np.array([], dtype=float)
    if isinstance(x, (int, float)):
        return _np.array([float(x)], dtype=float)
    if isinstance(x, str):
        s = x.strip()

        try:
            obj = _json.loads(s)
            return _to_float_array(obj)
        except Exception:
            pass

        try:
            parts = [p for p in s.split(",") if p.strip() != ""]
            return _np.asarray([float(p) for p in parts], dtype=float).reshape(-1)
        except Exception:
            return _np.array([], dtype=float)

    return _np.array([], dtype=float)

def make_infer_task_dir(slug: str) -> Path:
    p = (DEPLOY_DIR_PATH / slug).resolve()
    if not _is_safe_subpath(DEPLOY_DIR_PATH, p):
        raise ValueError("不合法的部屬任務路徑")
    p.mkdir(parents=True, exist_ok=False)
    return p

def ensure_infer_task_dirs(task: dict) -> Path:
    slug = (task.get("slug") or slugify_name(task.get("name") or "")).strip()
    if not slug:
        raise ValueError("部屬任務資料夾名（slug）無效")
    base_path = (DEPLOY_DIR_PATH / slug).resolve()
    if not _is_safe_subpath(DEPLOY_DIR_PATH, base_path):
        raise ValueError("不合法的部屬任務路徑")
    base_path.mkdir(parents=True, exist_ok=True)
    return str(base_path)


def figure_from_detection_item(item: dict) -> go.Figure:
    """
    用偵測結果畫圖。優先畫 reconstruction & anomaly_scores。
    若兩者皆不可用，退而畫原始 CSV（item['file']）。
    """
    try:
        # 1) 聰明抓欄位（支援多種命名）
        rec_keys = ["reconstruction", "recon", "reconstructed", "prediction", "pred", "yhat", "forecast"]
        sco_keys = ["anomaly_scores", "anomaly_score", "scores", "score", "anomaly", "outlier_scores"]

        rec_raw = None
        for k in rec_keys:
            if k in item:
                rec_raw = item.get(k)
                break

        sco_raw = None
        for k in sco_keys:
            if k in item:
                sco_raw = item.get(k)
                break

        rec = _to_float_array(rec_raw)
        sco = _to_float_array(sco_raw)

        # 2) 兩個都空 → 讀原始 CSV 畫波形
        if rec.size == 0 and sco.size == 0:
            fpath = item.get("file")
            if fpath and os.path.exists(fpath):
                return figure_from_csv_path(fpath)
            # 沒有檔案可讀：回報最少資訊的空圖
            fig = empty_wave_fig()
            fig.update_layout(title="沒有可畫的資料（reconstruction/anomaly_scores 缺失，且無 file）")
            return fig

        # 3) 正常畫圖
        fig = go.Figure()
        if rec.size > 0:
            fig.add_trace(go.Scattergl(x=np.arange(rec.size), y=rec, mode="lines",
                                       name="Reconstruction", line=dict(width=1)))
        if sco.size > 0:
            fig.add_trace(go.Scattergl(x=np.arange(sco.size), y=sco, mode="lines",
                                       name="Anomaly Score", line=dict(width=1)))

        fig.update_layout(
            # title=None  # 不設標題，檔名放到區塊標題顯示
            margin=dict(l=40, r=10, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
            xaxis=dict(title="樣點"),
            yaxis=dict(title="幅值"),
            autosize=True, dragmode="zoom",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        
        return fig

    except Exception as e:
        # 不要吃掉錯誤：在圖上提示，方便排查
        fig = empty_wave_fig()
        fig.update_layout(title=f"繪圖失敗：{e}")
        return fig
    


def scan_deploy_tasks() -> list:
    tasks = []
    try:
        for entry in sorted(os.listdir(DEPLOY_DIR)):
            base = os.path.join(DEPLOY_DIR, entry)
            if not os.path.isdir(base):
                continue
            sid = hashlib.md5(entry.encode("utf-8")).hexdigest()[:12]
            mtime = datetime.fromtimestamp(os.path.getmtime(base), tz=ZoneInfo("Asia/Taipei")).strftime("%Y/%m/%d %H:%M:%S")
            tasks.append({
                "id": sid,
                "name": entry,
                "slug": entry,
                "desc": "",
                "time": mtime,
                "normal": [],
                "abnormal": []
            })
    except Exception:
        pass
    return tasks

def make_task_dir(slug: str) -> Path:
    if not slug:
        raise ValueError("空白的 slug")
    p = (UPLOADS_DIR_PATH / slug).resolve()
    if not _is_safe_subpath(UPLOADS_DIR_PATH, p):
        raise ValueError("不合法的任務路徑")
    p.mkdir(parents=True, exist_ok=False)
    return p

def slugify_name(name: str) -> str:
    s = (name or "").strip() or "task"
    safe = []
    for ch in s:
        if ch.isalnum() or ch in "-_ ":
            safe.append(ch)
        else:
            safe.append("-")
    s = "".join(safe).replace(" ", "-")
    while "--" in s:
        s = s.replace("--", "-")
    s = s.strip("-_")
    s = _uniqueish_task_slug(s)
    s = s[:64]
    return s

def ensure_task_dirs(task: dict, allow_migrate: bool = True) -> tuple[str, str]:
    slug = (task.get("slug") or slugify_name(task.get("name") or "")).strip()
    if not slug:
        raise ValueError("任務資料夾名（slug）無效")

    base_path = (UPLOADS_DIR_PATH / slug).resolve()
    if not _is_safe_subpath(UPLOADS_DIR_PATH, base_path):
        raise ValueError("不合法的任務路徑")

    # ---- 保守 legacy migration（A+B 已套用）----
    old_id = str(task.get("id", "")).strip()
    migrate_marker = _migration_marker_path(base_path)
    do_migrate = (
        allow_migrate and
        _legacy_migration_enabled() and
        _looks_like_legacy_id(old_id) and
        bool(old_id) and
        old_id != slug and
        not base_path.exists()
    )
    if do_migrate:
        old_base = (UPLOADS_DIR_PATH / old_id).resolve()
        is_direct_child = (old_base.parent == UPLOADS_DIR_PATH)
        if (
            is_direct_child and
            old_base != UPLOADS_DIR_PATH and
            old_base.exists() and
            old_base.is_dir() and
            _is_safe_subpath(UPLOADS_DIR_PATH, old_base) and
            not migrate_marker.exists()
        ):
            try:
                shutil.move(str(old_base), str(base_path))
                try:
                    migrate_marker.write_text(old_id, encoding="utf-8")
                except Exception:
                    pass
            except Exception:
                pass

    normal = base_path / "normal"
    abnormal = base_path / "abnormal"
    normal.mkdir(parents=True, exist_ok=True)
    abnormal.mkdir(parents=True, exist_ok=True)
    return str(normal), str(abnormal)

def save_b64_to_file(content_b64: str, fpath: str) -> bool:
    try:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        payload = content_b64.split(",")[-1]
        with open(fpath, "wb") as fw:
            fw.write(base64.b64decode(payload))
        return True
    except Exception:
        return False

def scan_task_files(task: dict) -> tuple:
    # 熱路徑掃描：禁止遷移（A）
    normal_dir, abnormal_dir = ensure_task_dirs(task, allow_migrate=False)
    def list_dir(d: str) -> list:
        out = []
        try:
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    out.append({"name": fn, "path": p})
        except Exception:
            pass
        return out
    return list_dir(normal_dir), list_dir(abnormal_dir)

def scan_uploads_for_tasks() -> list:
    tasks = []
    try:
        for entry in sorted(os.listdir(UPLOADS_DIR)):
            base = os.path.join(UPLOADS_DIR, entry)
            if not os.path.isdir(base):
                continue
            normal_dir = os.path.join(base, "normal")
            abnormal_dir = os.path.join(base, "abnormal")
            def list_dir(d: str) -> list:
                out = []
                if os.path.isdir(d):
                    for fn in sorted(os.listdir(d)):
                        p = os.path.join(d, fn)
                        if os.path.isfile(p):
                            out.append({"name": fn, "path": p})
                return out
            sid = hashlib.md5(entry.encode("utf-8")).hexdigest()[:12]
            mtime = datetime.fromtimestamp(os.path.getmtime(base), tz=ZoneInfo("Asia/Taipei")).strftime("%Y/%m/%d %H:%M:%S")
            tasks.append({
                "id": sid,
                "name": entry,
                "slug": entry,
                "desc": "",
                "time": mtime,
                "normal": list_dir(normal_dir),
                "abnormal": list_dir(abnormal_dir)
            })
    except Exception:
        pass
    return tasks

def _read_logo_src(rel_path: str = "logo/database.png") -> str | None:
    try:
        p = os.path.normpath(resource_path(rel_path))
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

LOGO_DB_SRC = _read_logo_src()

def empty_wave_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=40, r=10, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        autosize=True, dragmode="zoom",
        annotations=[dict(text="尚未載入資料", x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(color="#64748b", size=16))]
    )
    return fig

def figure_from_csv_path(path: str) -> go.Figure:
    try:
        df = pd.read_csv(path)
        if df.empty:
            return empty_wave_fig()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        series = df[num_cols[0]] if num_cols else pd.to_numeric(df.iloc[:, 0], errors="coerce")
        x = np.arange(len(series))
        fig = go.Figure(go.Scattergl(x=x, y=series, mode="lines", line=dict(width=1)))
        fig.update_layout(
            margin=dict(l=40, r=10, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
            xaxis=dict(title="樣點"), yaxis=dict(title="幅值"),
            autosize=True, dragmode="zoom"
        )
        return fig
    except Exception:
        return empty_wave_fig()
    
def _is_task_running(run: dict, task_id: str) -> bool:
    if not run or not run.get("enabled"):
        return False
    # 單執行舊格式
    if run.get("task"):
        return str(run["task"].get("id")) == str(task_id)
    # 多執行新格式
    for t in (run.get("tasks") or []):
        if str(t.get("id")) == str(task_id):
            return True
    return False

def figure_from_csv_content(content_b64: str) -> go.Figure:
    try:
        if not content_b64:
            return empty_wave_fig()
        content_string = content_b64.split(",")[-1]
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except Exception:
            df = pd.read_csv(io.BytesIO(decoded))
        if df.empty:
            return empty_wave_fig()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        series = df[num_cols[0]] if num_cols else pd.to_numeric(df.iloc[:, 0], errors="coerce")
        x = np.arange(len(series))
        fig = go.Figure(go.Scattergl(x=x, y=series, mode="lines", line=dict(width=1)))
        fig.update_layout(
            margin=dict(l=40, r=10, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
            xaxis=dict(title="樣點"), yaxis=dict(title="幅值"),
            autosize=True, dragmode="zoom"
        )
        return fig
    except Exception:
        return empty_wave_fig()

def spec_fig(seed: int = 0, title: str = "") -> go.Figure:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, 96)
    band = np.tanh((base - 0.15) * 8)
    mat = rng.normal(0, 0.12, (96, 96))
    mat += band[:, None] * 1.2
    fig = go.Figure(data=go.Heatmap(z=mat, colorscale="Turbo", showscale=False))
    fig.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False), height=220,
        title=dict(text=title, x=0.02, y=0.95, font=dict(size=12, color="#0b1220")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def status_dot(color: str = "#22c55e") -> html.Span:
    return html.Span(style={"display": "inline-block", "width": 8, "height": 8,
                            "borderRadius": 8, "background": color, "marginRight": 8})

def file_row(label: str, ok: bool = True) -> html.Div:
    color = "#22c55e" if ok else "#ef4444"
    return html.Div(
        [
            status_dot(color),
            html.Span(
                label,
                className="file-label",
                style={
                    "whiteSpace": "normal",
                    "wordBreak": "break-all",
                    "display": "block",
                    "flex": "1",
                    "minWidth": 0
                },
            ),
        ],
        className="file-row",
        style={
            "display": "flex",
            "alignItems": "flex-start",
            "gap": "8px",
            "maxWidth": "100%",
        },
    )

def fetch_model_options():
    try:
        r = requests.request("GET", urlGetModel, timeout=5)
        r.raise_for_status()
        data = r.json()

        items = data.get("models", data) if isinstance(data, dict) else data
        opts = []
        for m in items:
            if isinstance(m, dict):
                label = m.get("label") or m.get("name") or str(m.get("id") or "")
                value = m.get("value") or m.get("name") or m.get("id") or label
            else:
                label = str(m)
                value = label
            if label:
                opts.append({"label": label, "value": str(value)})

        # 去重保序
        seen, dedup = set(), []
        for o in opts:
            if o["value"] in seen:
                continue
            seen.add(o["value"])
            dedup.append(o)
        return dedup
    except Exception:
        return []

# ==============================
# Dash app
# ==============================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "異常檢測 – 樣板"

# ---------- Layout ----------
app.layout = html.Div([
    # 左側導覽選單
    html.Div([
        html.Div([html.Div("L3C", className="nav-logo-1"), html.Div("異常檢測", className="nav-sub")], className="nav-brand"),
        dcc.Store(id="nav-store", data={"top": "project", "sub": "data"}),
        html.Div([
            html.Div("專案", id="nav-top-project", n_clicks=0, className="nav-top-item"),
            html.Div("部屬", id="nav-top-infer", n_clicks=0, className="nav-top-item")
        ], className="nav-top"),
        
        html.Div(id="subnav-project", className="nav-list-project", children=[
            html.Div("資料", id="nav-sub-data", n_clicks=0, className="nav-item"),
            html.Div("分析", id="nav-sub-analyze", n_clicks=0, className="nav-item"),
        ]),
        
        html.Div(id="subnav-infer", className="nav-list-infer", children=[
            html.Div("設定", id="nav-sub-setting", n_clicks=0, className="nav-item"),
        ])

    ], className="sidenav"),

    # 右側內容區
    html.Div([
        # 頂部導航列
        html.Div([
            html.Div(id="brand-title",
                children=[html.Span("L3C"), html.Span(" 異常檢測", style={"opacity": .7})],
                className="brand"),

            html.Button("◀ 返回上一頁", id="btn-back", n_clicks=0, className="btn back", style={"display": "none"}),
            html.Div("", className="page-title"),
        ], className="topbar"),

        dcc.Location(id="url"),

        # 全域錯誤提示
        html.Div(id="msg-toast", className="toast error", style={"display": "none"},
                 children=[html.Span(id="msg-text", className="toast-text")]),
        dcc.Interval(id="msg-timer", interval=2500, n_intervals=0, disabled=True),

        # ===== 專案 > 分析頁 =====
        html.Div([
            html.Div([
                html.Div("任務清單", className="section-title"),
                dcc.Dropdown(id="anal-task-dd", options=[], value=None, clearable=False, placeholder="選擇任務"),

                html.Div("模型", className="section-title", style={"marginTop": 16}),
                html.Div([
                    dcc.Dropdown(id="model-dd", options=[], value=None, placeholder="Select...", clearable=False,
                                 style={"flex": 1}),
                    html.Button("🔍 搜尋模型", id="btn-refresh-models", n_clicks=0, className="btn small outline",
                                style={"whiteSpace": "nowrap"}),
                ], style={"display": "flex", "gap": "8px", "marginBottom": "12px"}),

                html.Div([
                    html.Div([
                        html.Div("正常資料", className="sub-title"),
                        dcc.Checklist(
                            id="anal-normal-ck",
                            options=[], value=[],
                            className="list-box",
                            inputStyle={"marginRight": "8px"},
                            labelStyle={"display": "block", "marginBottom": "6px", "wordBreak": "break-all"}
                        )
                    ], className="list-col"),
                
                    html.Div([
                        html.Div("異常資料", className="sub-title"),
                        dcc.Checklist(
                            id="anal-abnormal-ck",
                            options=[], value=[],
                            className="list-box",
                            inputStyle={"marginRight": "8px"},
                            labelStyle={"display": "block", "marginBottom": "6px", "wordBreak": "break-all"}
                        )
                    ], className="list-col"),
                ], className="two-lists"),
                
                
                
                html.Div([
                    html.Button("開始分析", id="btn-detect", className="btn full")
                ], style={"marginTop": 16})
            ], className="side-left"),

            html.Div([
                html.Div([
                    html.Span("目前閥值："),
                    html.Strong(id="thres-value", children="0.50")
                ], style={"marginTop": 6, "color": "#334155"}),
                
                html.Div([
                    dcc.Slider(
                        id="thres-slider",
                        min=0.0, max=1.0, step=0.01, value=0.80,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={"padding": "10px 4px"}),
                
            ], className="side-middle", style={"display": "flex", "flexDirection": "column", "justifyContent": "flex-start"}),

            html.Div([
                dcc.Loading(
                    id="loading-detect",
                    type="default",              # 可改 'circle' / 'dot' / 'cube'
                    color="#4f46e5",             # 想改顏色就改這裡
                    children=html.Div([
                        html.Div([
                            html.Span(id="analysis-title", children="分析數據圖"),
                            html.Span([
                                dcc.Dropdown(
                                    id="result-dd",
                                    options=[],
                                    value=None,
                                    placeholder="選擇要顯示的檔案…",
                                    style={"minWidth": "260px"}
                                ),
                                html.Button("重置", id="btn-reset-zoom", n_clicks=0, className="btn small outline")
                            ], className="plot-ctrl")
                        ], className="section-title flex"),
            
                        html.Div(
                            dcc.Graph(
                                id="wave",
                                figure=empty_wave_fig(),
                                config={"displayModeBar": False, "responsive": True},
                                style={"width": "100%", "height": "100%"}
                            ),
                            className="plot-holder"
                        )
                    ])
                )
            ], className="edit-right")
        
        ], className="main", id="page-project-analyze"),
        
        # ===== 部屬 > 設定 （任務清單） =====
        html.Div([
            html.Div([
                html.Div("部屬設定", className="section-title"),
                html.Div([
                    html.Div([dcc.Input(id="infer-task-name", placeholder="任務名稱", className="input")], className="form-col"),
                    html.Div([dcc.Input(id="infer-task-desc", placeholder="任務描述", className="input")], className="form-col"),
                    html.Button("新增任務", id="infer-btn-add-task", className="btn outline teal")
                ], className="form-row"),
                
                html.Div("任務清單", className="section-title", style={"marginTop": 12}),
                html.Div(id="infer-task-list", className="task-list"),
            ], className="data-card")
        ], className="main", id="page-analyze-setting", style={"display": "none"}),
        
        # ===== 部屬 > 設定 > 編輯頁 =====
        html.Div([
            html.Div([
                html.Div([
                    # 置中標題
                    html.Div("APC參數設定", className="section-title",
                             style={"textAlign": "center", "fontWeight": "600",
                                    "fontSize": "1.1rem", "marginBottom": "16px"}),
            
                    html.Div([
                        html.Div([
                            html.Label("apc_type", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-apc-type", type="text", placeholder="輸入 apc_type",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                
                        html.Div([
                            html.Label("function", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-function", type="text", placeholder="輸入 function",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                
                        html.Div([
                            html.Label("tooltype", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-tooltype", type="text", placeholder="輸入 tooltype",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                
                        html.Div([
                            html.Label("chamber", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-chamber", type="text", placeholder="輸入 chamber",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                
                        html.Div([
                            html.Label("recipe", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-recipe", type="text", placeholder="輸入 recipe",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "16px"}),
                    ]),
                    
                    html.Hr(style={"margin": "16px 0"}),

                ], className="side-middle", style={
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "flex-start",
                    "padding": "12px",
                    "flex": "0 0 60px",     # ← 固定欄寬
                    "minWidth": "280px",     # ← 下限，避免太窄
                }),
                    
                html.Div([
                    html.Div("龍捲風設定", className="section-title",
                             style={"textAlign": "center", "fontWeight": "600",
                                    "fontSize": "1.1rem", "marginBottom": "16px"}),
            
                    html.Div([
                        html.Div([
                            html.Label("URL EMS CALL", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-url-ems-call-type", type="text", placeholder="輸入 URL EMS CALL",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                
                        html.Div([
                            html.Label("CORP ID", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-corp-id", type="text", placeholder="輸入 CORP ID",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                
                        html.Div([
                            html.Label("API KEY", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-api-key", type="text", placeholder="輸入 API KEY",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                
                        html.Div([
                            html.Label("ERROR POLICY", style={"fontWeight": "500", "color": "#334155"}),
                            dcc.Input(id="infer-error-policy", type="text", placeholder="輸入 ERROR POLICY",
                                      className="input", style={"width": "100%", "marginTop": "4px"})
                        ], style={"marginBottom": "12px"}),
                    ]),
                    
                    html.Hr(style={"margin": "16px 0"}),

                ], className="side-middle", style={
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "flex-start",
                    "padding": "12px",
                    "flex": "0 0 60px",     # ← 固定欄寬
                    "minWidth": "280px",     # ← 下限，避免太窄
                }),
                    
                html.Div([
                    html.Div("其他設定", className="section-title",
                             style={"textAlign": "center", "fontWeight": "600",
                                    "fontSize": "1.1rem", "marginBottom": "16px"}),
            
                    html.Div([
                        html.Div([
                            html.Span("閥值調控", style={"fontWeight": "500", "color": "#334155"}),
                            html.Span(id="infer-thres-value", style={"marginLeft": "auto", "fontWeight": "600"})
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"}),
                    
                        dcc.Slider(
                            id="infer-thres-slider",
                            min=0.0, max=1.0, step=0.01, value=0.80,
                            tooltip={"placement": "bottom", "always_visible": False}
                        ),
                        
                        html.Div([
                            html.Span("觀測窗口", style={"fontWeight": "500", "color": "#334155"}),
                            html.Span(id="infer-time-range", style={"marginLeft": "auto", "fontWeight": "600"})
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"}),
                    
                        dcc.Slider(
                            id="infer-window-slider",
                            min=1, max=10, step=1, value=5,
                            tooltip={"placement": "bottom", "always_visible": False}
                        ),
                        
                        html.Div([
                            html.Div([
                                html.Span("偵測時間（分鐘）", style={"fontWeight": "500", "color": "#334155"}),
                                html.Span(id="infer-detect-time-value", style={"marginLeft": "auto", "fontWeight": "600"})
                            ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"}),
                        
                            dcc.Slider(
                                id="infer-detect-time",
                                min=5, max=60, step=5, value=30,
                                tooltip={"placement": "bottom", "always_visible": False},
                                marks={i: f"{i}" for i in range(5, 65, 5)}
                            ),
                        ], style={"marginBottom": "14px"}),
  
                    ], style={"marginBottom": "8px"}),
                    
                    html.Hr(style={"margin": "16px 0"}),
                    
                    html.Div([
                        html.Button("測試", id="btn-run_apc", className="btn apc", style={"width": "100%", "marginTop": 16})
                    ], style={"width": "100%", "marginTop": 16}),
                    
                ], className="side-middle", style={
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "flex-start",
                    "padding": "12px",
                    "flex": "0 0 60px",     # ← 固定欄寬
                    "minWidth": "280px",     # ← 下限，避免太窄
                }),

                html.Div([
                    dcc.Loading(
                        id="loading-infer",
                        type="default",
                        color="#4f46e5",
                        children=html.Div([
                            html.Div([
                                html.Span(id="infer-title", children="結果圖"),
                                html.Span([
                                    dcc.Dropdown(
                                        id="infer-result-dd",
                                        options=[],
                                        value=None,
                                        placeholder="選擇要顯示的檔案…",
                                        style={"minWidth": "260px"}
                                    ),
                                    html.Button("重置", id="btn-infer-reset-zoom", n_clicks=0, className="btn small outline")
                                ], className="plot-ctrl")
                            ], className="section-title flex"),

                            html.Div(
                                dcc.Graph(
                                    id="infer-wave",
                                    figure=empty_wave_fig(),
                                    config={"displayModeBar": False, "responsive": True},
                                    style={"width": "100%", "height": "100%"}
                                ),
                                className="plot-holder",
                                style={"minWidth": 0, "height": "calc(100vh - 260px)"}  # 可依你的頁頭高度微調
                            )
                        ])
                    )
                ], className="edit-right", style={
                    "flex": "1 1 auto",   # ← 讓它吃掉剩餘空間
                    "minWidth": 0         # ← 沒這個會被內容撐爆，無法縮
                })

            ], className="editor-row", style={
                "display": "flex",
                "gap": "16px",
                "alignItems": "stretch",
                "width": "100%",
                "minWidth": 0            # ← 關鍵：允許子項縮
            })
        ], className="main", id="page-setting-editor", style={"display": "none"}),
        
        # ===== 專案 > 資料頁（任務清單） =====
        html.Div([
            html.Div([
                html.Div("新增任務", className="section-title"),
                html.Div([
                    html.Div([dcc.Input(id="task-name", placeholder="任務名稱", className="input")], className="form-col"),
                    html.Div([dcc.Input(id="task-desc", placeholder="任務描述", className="input")], className="form-col"),
                    html.Button("新增任務", id="btn-add-task", className="btn outline teal")
                ], className="form-row"),
                html.Div("任務清單", className="section-title", style={"marginTop": 12}),
                html.Div(id="task-list", className="task-list")
            ], className="data-card")
        ], className="main", id="page-project-dataset", style={"display": "none"}),

        # ===== 專案 > 資料 > 編輯頁 =====
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("正常資料列表"),
                        html.Div([
                            dcc.Upload(id="u-normal-file", children=html.Span("📄", className="ticon"), multiple=True),
                            dcc.Upload(id="u-normal-folder", children=html.Span("🗂", className="ticon"), multiple=True),
                            html.Button("🗑", id="btn-normal-clear", n_clicks=0, className="ticon btn-icon")
                        ], className="toolbar")
                    ], className="section-title flex"),
                    html.Div(id="normal-list", className="file-list")
                ], className="edit-mid"),

                html.Div([
                    html.Div([
                        html.Span("異常資料列表"),
                        html.Div([
                            dcc.Upload(id="u-abnormal-file", children=html.Span("📄", className="ticon"), multiple=True),
                            dcc.Upload(id="u-abnormal-folder", children=html.Span("🗂", className="ticon"), multiple=True),
                            html.Button("🗑", id="btn-abnormal-clear", n_clicks=0, className="ticon btn-icon")
                        ], className="toolbar")
                    ], className="section-title flex"),
                    html.Div(id="abnormal-list", className="file-list")
                ], className="edit-mid"),

                html.Div([
                    html.Div([html.Span("數據圖"),
                              html.Span([html.Button("重置", id="btn-reset-zoom-editor", n_clicks=0,
                                                     className="btn small outline")], className="plot-ctrl")],
                             className="section-title flex"),
                    
                    html.Div(dcc.Graph(id="editor-wave", figure=empty_wave_fig(),
                                       config={"displayModeBar": False, "responsive": True},
                                       style={"width": "100%", "height": "100%"}), className="plot-holder")
                    
                ], className="edit-right")
            ], className="editor-row")
        ], className="main", id="page-project-editor", style={"display": "none", "flexDirection": "column"}),

        # Store 元件
        dcc.Store(id="tasks-store", data=[]),
        dcc.Store(id="data-mode", data="list"),
        dcc.Store(id="editor-store", data={}),
        dcc.Store(id="normal-files", data=[]),
        dcc.Store(id="abnormal-files", data=[]),
        dcc.Store(id="selected-file", data=None),
        dcc.Store(id="save-signal", data=None),
        dcc.Store(id="notice-signal", data=None),
        dcc.Store(id="files-changed-signal", data=None),        
        dcc.Store(id="anal-selected", data={"normal": [], "abnormal": []}),
        dcc.Store(id="detect-results", data=None),
        dcc.Store(id="infer-file", data=None),
        dcc.Store(id="infer-results", data=None),
        dcc.Store(id="infer-tasks-store", data=[]),
        dcc.Store(id="infer-mode", data="list"),
        dcc.Store(id="infer-editor-store", data={}),
        dcc.Store(id="infer-run", data={"enabled": False, "interval_ms": 60000, "tasks": []}, storage_type="local"),
        dcc.Store(id="infer-run-signal", data=None),   # 每次 tick 會放一個 uuid 與任務資訊
        dcc.Interval(id="infer-runner", interval=1000, n_intervals=0, disabled=True),
        dcc.Store(id="infer-run-mode", data="multi"),

    ], className="content"),
], className="shell")

# ---------- Callbacks ----------

# 分析頁 任務下拉
@app.callback(
    Output("anal-task-dd", "options"),
    Output("anal-task-dd", "value"),
    Input("tasks-store", "data"),
    State("anal-task-dd", "value")
)
def _anal_task_dropdown(tasks, current):
    tasks = tasks or []
    opts = [{"label": (t.get("name") or t.get("slug") or "(未命名)"),
             "value": (t.get("slug") or "").strip()} for t in tasks if t.get("slug")]
    if current and any(o["value"] == current for o in opts):
        val = current
    else:
        val = opts[0]["value"] if opts else None
    return opts, val

@app.callback(
    Output("model-dd", "options"),
    Output("model-dd", "value"),
    Output("notice-signal", "data", allow_duplicate=True),
    Input("btn-refresh-models", "n_clicks"),
    State("model-dd", "value"),
    prevent_initial_call="initial_duplicate"
)
def refresh_model_list(n_clicks, current_value):
    opts = fetch_model_options()
    if not opts:
        return dash.no_update, dash.no_update, {"type": "error", "text": "取得模型清單失敗或為空"}
    new_value = current_value if (current_value and any(o["value"] == current_value for o in opts)) else opts[0]["value"]
    return opts, new_value, None

@app.callback(
    Output("anal-normal-ck", "options"), Output("anal-normal-ck", "value"),
    Output("anal-abnormal-ck", "options"), Output("anal-abnormal-ck", "value"),
    Input("anal-task-dd", "value"),
    Input("files-changed-signal", "data"),
    State("tasks-store", "data"),
    prevent_initial_call=False
)
def _anal_checklists(slug, _sig, tasks):
    if not slug:
        return [], [], [], []
    task = next((t for t in (tasks or []) if (t.get("slug") or "").strip() == slug), None) or {"slug": slug}
    normals, abns = scan_task_files(task)

    # Checklist 的 options 要有唯一 value，這裡用「完整路徑」，label 用檔名
    n_opts = [{"label": f"{f['name']}", "value": f["path"]} for f in normals]
    a_opts = [{"label": f"{f['name']}", "value": f["path"]} for f in abns]

    n_vals = []
    a_vals = []
    
    return n_opts, n_vals, a_opts, a_vals

@app.callback(
    Output("infer-run", "data"),
    Input("url", "href"),
    prevent_initial_call=False
)
def hydrate_infer_run(_):
    """
    載入頁面時根據 deploy_tasks + 各任務 config.json 重建執行狀態：
    - 為每個任務帶入 period_ms 與 next_due_ts（now + period）
    - 全域 enabled = 是否有任務被標記為啟用
    - 全域 interval_ms 固定 1000（但我們實際上不再使用它來控制 Interval）
    """
    tasks = scan_deploy_tasks() or []
    active = []
    now_ms = int(time.time() * 1000)

    for t in tasks:
        slug = (t.get("slug") or "").strip()
        if not slug:
            continue
        if not get_task_enabled(slug):
            continue

        cfg = read_infer_config(slug) or {}
        minutes = int(cfg.get("detect_time") or 5)
        minutes = max(1, minutes)
        period_ms = minutes * 60 * 1000

        active.append({
            "id": t.get("id"),
            "slug": slug,
            "name": t.get("name"),
            "period_ms": period_ms,
            "next_due_ts": now_ms + period_ms  # 首次載入後的下一次到期點
        })

    return {"enabled": bool(active), "interval_ms": 1000, "tasks": active}

@app.callback(
    Output("infer-thres-value", "children"),
    Output("infer-time-range", "children"),
    Output("infer-detect-time-value", "children"),
    Input("infer-thres-slider", "value"),
    Input("infer-window-slider", "value"),
    Input("infer-detect-time", "value"),
    prevent_initial_call=False   # 保證第一次載入就更新顯示
)
def update_infer_values(thres, window, detect):
    # 保護 None，並格式化輸出
    thres_val = "—" if thres is None else f"{float(thres):.2f}"
    window_val = "—" if window is None else f"{int(window)}"
    # detect 是 0~1 之間的浮點數，保留 2 位小數
    detect_val = "—" if detect is None else f"{int(detect)}"
    # 如果要顯示百分比，可以改成 f"{float(detect)*100:.1f}%"

    return thres_val, window_val, detect_val


def infer_user_subfunction(task: dict, tick: dict):
    """
    你要執行的工作寫在這裡
    task: {"id":..., "slug":..., "name":...}
    tick: {"id": uuid, "ts": "YYYY/MM/DD HH:MM:SS", "count": n}
    """
    pass  # 留空給你實作

@app.callback(
    Output("infer-apc-type", "value"),
    Output("infer-function", "value"),
    Output("infer-tooltype", "value"),
    Output("infer-chamber", "value"),
    Output("infer-recipe", "value"),
    Output("infer-url-ems-call-type", "value"),
    Output("infer-corp-id", "value"),
    Output("infer-api-key", "value"),
    Output("infer-error-policy", "value"),
    Output("infer-thres-slider", "value"),
    Output("infer-window-slider", "value"),
    Output("infer-detect-time", "value"),
    Input("infer-editor-store", "data"),
    State("infer-tasks-store", "data"),
    prevent_initial_call=True
)
def load_infer_editor_fields(editor, tasks):
    if not editor or tasks is None:
        raise PreventUpdate

    task_id = str(editor.get("task_id") or "")
    if not task_id:
        raise PreventUpdate

    task = next((t for t in (tasks or []) if str(t.get("id")) == task_id), None)
    if not task:
        raise PreventUpdate

    slug = (task.get("slug") or "").strip()
    cfg = read_infer_config(slug) or {}

    return (
        cfg.get("apc_type", ""),
        cfg.get("function", ""),
        cfg.get("tooltype", ""),
        cfg.get("chamber", ""),
        cfg.get("recipe", ""),
        cfg.get("url_ems_call", ""),
        cfg.get("corp_id", ""),
        cfg.get("api_key", ""),
        cfg.get("error_policy", ""),
        float(cfg.get("thres", 0.80)) if cfg.get("thres") is not None else 0.80,
        int(cfg.get("window", 5)) if cfg.get("window") is not None else 5,
        int(cfg.get("detect_time", 30)) if cfg.get("detect_time") is not None else 30,
    )
@app.callback(
    Output("infer-run", "data", allow_duplicate=True),   # ← 新增：回寫 next_due_ts
    Output("infer-run-signal", "data"),
    Input("infer-runner", "n_intervals"),
    State("infer-run", "data"),
    State("infer-apc-type", "value"),
    State("infer-function", "value"),
    State("infer-tooltype", "value"),
    State("infer-chamber", "value"),
    State("infer-recipe", "value"),
    State("infer-url-ems-call-type", "value"),
    State("infer-corp-id", "value"),
    State("infer-api-key", "value"),
    State("infer-error-policy", "value"),
    State("infer-thres-slider", "value"),
    State("infer-window-slider", "value"),
    State("infer-detect-time", "value"),
    prevent_initial_call=True
)
def on_runner_tick(n, run, apc_type, function, tooltype, chamber, recipe,
                   url_ems_call, corp_id, api_key, error_policy,
                   thres, window, detect_time):

    if not run or not run.get("enabled"):
        raise PreventUpdate

    base_payload = {
        "apc_type": apc_type or "",
        "function": function or "",
        "tooltype": tooltype or "",
        "chamber": chamber or "",
        "recipe": recipe or "",
        "url_ems_call": url_ems_call or "",
        "corp_id": corp_id or "",
        "api_key": api_key or "",
        "error_policy": error_policy or "",
        "thres": float(thres) if thres is not None else None,
        "window": int(window) if window is not None else None,
        "detect_time": int(detect_time) if detect_time is not None else None
    }

    tasks = (run.get("tasks") or []).copy()
    if not tasks:
        raise PreventUpdate

    now_ms = int(time.time() * 1000)
    ok, fail, details = 0, 0, []
    
    ran_any = False  # ← 新增旗標
    
    # 逐一檢查是否到期
    for i, t in enumerate(tasks):
        slug = (t.get("slug") or "").strip()
        if not slug:
            fail += 1
            details.append("missing-slug")
            continue

        period_ms = int(t.get("period_ms") or 300000)
        next_due  = int(t.get("next_due_ts") or 0)

        # 未到期 → 跳過
        if now_ms < next_due:
            continue

        # 讀該任務設定，覆蓋 UI 後備值
        cfg = read_infer_config(slug) or {}

        def pick(key):
            v = cfg.get(key)
            return base_payload[key] if v in (None, "") else v

        payload = {
            "apc_type": pick("apc_type"),
            "function": pick("function"),
            "tooltype": pick("tooltype"),
            "chamber": pick("chamber"),
            "recipe": pick("recipe"),
            "url_ems_call": pick("url_ems_call"),
            "corp_id": pick("corp_id"),
            "api_key": pick("api_key"),
            "error_policy": pick("error_policy"),
            "thres": float(pick("thres")) if pick("thres") not in (None, "") else None,
            "window": int(pick("window")) if pick("window") not in (None, "") else None,
            "detect_time": int(pick("detect_time")) if pick("detect_time") not in (None, "") else None
        }
        
        ran_any = True   # ← 有到期才算
        
        try:
            timeout = REQ_BASE_TIMEOUT_SEC
            _ = _post_with_retry(
                urlGetOneTest,
                json_payload={**payload, "slug": slug},
                timeout=timeout,
                max_retry=REQ_MAX_RETRY
            )
            ok += 1
            # 成功或失敗都要把下一次到期往後推，避免密集重試造成同時觸發
            # 若拖得太久造成 now >> next_due，補到最近未來一格
        except requests.exceptions.Timeout:
            fail += 1
            details.append(f"{slug}: timeout")
        except Exception as e:
            fail += 1
            details.append(f"{slug}: {e}")
        finally:
            # 無論成功失敗都往後對齊到最近的「未來」tick
            next_due_new = next_due
            while next_due_new <= now_ms:
                next_due_new += period_ms
            tasks[i] = {**t, "period_ms": period_ms, "next_due_ts": next_due_new}


    if not ran_any:
        # 沒有任何任務到期 → 不要動畫面，也不要彈 toast
        return no_update, no_update
    
    # 回寫更新後的 next_due_ts；enabled 狀態保持不變
    new_run = {**run, "tasks": tasks}

    return new_run, {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y/%m/%d %H:%M:%S"),
        "count": int(n or 0),
        "ok": ok,
        "fail": fail,
        "details": details
    }

@app.callback(
    Output("notice-signal", "data", allow_duplicate=True),
    Input("btn-run_apc", "n_clicks"),
    State("infer-apc-type", "value"),
    State("infer-function", "value"),
    State("infer-tooltype", "value"),
    State("infer-chamber", "value"),
    State("infer-recipe", "value"),
    State("infer-url-ems-call-type", "value"),
    State("infer-corp-id", "value"),
    State("infer-api-key", "value"),
    State("infer-error-policy", "value"),
    State("infer-thres-slider", "value"),
    State("infer-window-slider", "value"),
    State("infer-detect-time", "value"),
    State("infer-editor-store", "data"),
    State("infer-tasks-store", "data"),
    prevent_initial_call=True
)
def run_apc_test(n, apc_type, function, tooltype, chamber, recipe,
                 url_ems_call, corp_id, api_key, error_policy,
                 thres, window, detect_time,
                 editor, infer_tasks):
    if not n:
        raise PreventUpdate

    # 基本 payload（不含 slug；slug 僅在 request 時合併）
    payload = {
        "apc_type": apc_type or "",
        "function": function or "",
        "tooltype": tooltype or "",
        "chamber": chamber or "",
        "recipe": recipe or "",
        "url_ems_call": url_ems_call or "",
        "corp_id": corp_id or "",
        "api_key": api_key or "",
        "error_policy": error_policy or "",
        "thres": float(thres) if thres is not None else None,
        "window": int(window) if window is not None else None,
        "detect_time": int(detect_time) if detect_time is not None else None
    }

    # 取得目前編輯的部署任務 slug，並把參數存到該任務的 config.json
    slug = None
    try:
        task_id = str((editor or {}).get("task_id") or "")
        if task_id and infer_tasks:
            t = next((x for x in infer_tasks if str(x.get("id")) == task_id), None)
            if t:
                slug = (t.get("slug") or "").strip() or None
                if slug:
                    save_infer_config(slug, payload)  # 存檔時「不含」slug
    except Exception:
        pass

    # 呼叫測試 API：只在 request 時合併 slug
    try:
        timeout = REQ_BASE_TIMEOUT_SEC
        req_body = {**payload, "slug": slug} if slug else payload
        resp = _post_with_retry(urlGetOneTest, json_payload=req_body,
                                timeout=timeout, max_retry=REQ_MAX_RETRY)
        return {"type": "success", "text": f"API 呼叫成功：{resp}"}
    except requests.exceptions.Timeout:
        return {"type": "error", "text": "API 呼叫逾時"}
    except Exception as e:
        return {"type": "error", "text": f"呼叫失敗：{e}"}

@app.callback(
    Output("step-modal", "style"), Output("step-select", "value"),
    Input("btn-add-step", "n_clicks"), Input("btn-cancel-step", "n_clicks"), Input("btn-confirm-step", "n_clicks"),
    State("step-select", "value"), prevent_initial_call=True
)
def handle_modal(add_clicks, cancel_clicks, confirm_clicks, selected):
    trig = ctx.triggered_id
    hide, show = {"display": "none"}, {"display": "flex"}
    if trig == "btn-add-step":
        return show, None
    if trig == "btn-cancel-step":
        return hide, None
    if trig == "btn-confirm-step":
        return hide, None
    return hide, None

@app.callback(
    Output("steps-list", "children"), Input("steps-store", "data"))
def render_steps(data):
    items = []
    for i, label in enumerate(data or [], start=1):
        items.append(html.Div([
            html.Div([
                html.Span(f"步驟 {i}", className="step-num"),
                html.Button("×", id={"type": "del-step", "index": i-1}, n_clicks=0, className="icon-btn trash")
            ], className="step-head"),
            html.Div(label, className="step-title")
        ], className="step-card"))
    return items

@app.callback(
    Output("page-project-analyze", "style"), 
    Output("page-project-dataset", "style"), 
    Output("page-project-editor", "style"),
    Output("page-analyze-setting", "style"),
    Output("page-setting-editor", "style"),
    Input("nav-store", "data"), 
    Input("data-mode", "data"),
    Input("infer-mode", "data")
)
def toggle_pages(state, mode, infer_mode):
    state = state or {"top": None, "sub": None}

    # 只點到頂層（例如點了「部屬」但尚未點「設定」）→ 不改變右側頁面
    if state.get("sub") is None:
        raise PreventUpdate

    show = {"display": "flex"}
    hide = {"display": "none"}

    # 專案 > 資料（清單/編輯）
    if state["top"] == "project" and state["sub"] == "data":
        return hide, (show if mode == "list" else hide), (show if mode == "edit" else hide), hide, hide

    # 部屬 > 設定（清單/編輯）
    if state["top"] == "infer" and state["sub"] == "setting":
        return hide, hide, hide, (show if infer_mode == "list" else hide), (show if infer_mode == "edit" else hide)

    # 預設仍是分析頁
    return show, hide, hide, hide, hide



@app.callback(
    Output("infer-tasks-store", "data"),
    Output("notice-signal", "data", allow_duplicate=True),
    Input("url", "href"),
    Input("infer-btn-add-task", "n_clicks"),
    Input({"type": "infer-task-del", "task_id": ALL}, "n_clicks"),
    Input("nav-sub-setting", "n_clicks"),
    Input("nav-top-infer", "n_clicks"),
    State("infer-task-name", "value"),
    State("infer-task-desc", "value"),
    State("infer-tasks-store", "data"),
    State("infer-run", "data"),   # 👈 新增：目前執行狀態
    prevent_initial_call="initial_duplicate"
)
def infer_tasks_store_manager(href, add_clicks, del_clicks, go_setting_clicks, _,
                              name, desc, data, run):
    trig = ctx.triggered_id
    data = (data or []).copy()

    # 第一次載入頁面 → 不自動帶資料（維持空），避免與專案清單衝突
    if trig == "url":
        if data:
            raise PreventUpdate
        return [], None

    # 只要點到「部屬 > 設定」，就掃描 deploy_tasks 以同步清單
    if trig == "nav-sub-setting" or trig == "nav-top-infer":
        return scan_deploy_tasks(), None

    # 新增部屬任務（完全使用 deploy_tasks）
    if trig == "infer-btn-add-task":
        name_clean = (name or "").strip()
        if not name_clean:
            return dash.no_update, {"type": "error", "text": "請輸入任務名稱"}
        if not all(ch.isalnum() or ch in "-_ " for ch in name_clean):
            return dash.no_update, {"type": "error", "text": "任務名稱只能包含英數字、空白、底線或連字號"}

        new_slug = slugify_name(name_clean)
        if not new_slug:
            return dash.no_update, {"type": "error", "text": "此名稱無法轉為合法資料夾，請改用英數或可轉寫字元"}

        # 只看 deploy_tasks 內是否衝突
        existing_slugs = {(t.get("slug") or slugify_name(t.get("name") or "")) for t in data}
        folder_path = (DEPLOY_DIR_PATH / new_slug).resolve()
        if new_slug in existing_slugs or folder_path.exists():
            return dash.no_update, {"type": "error", "text": f"已存在同名部屬任務或資料夾：{name_clean}"}

        item = {
            "id": str(uuid.uuid4()),
            "name": name_clean,
            "slug": new_slug,
            "desc": (desc or ""),
            "time": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y/%m/%d %H:%M:%S"),
            # 部屬任務現在不需要 normal/abnormal，但保留欄位以維持通用結構
            "normal": [],
            "abnormal": []
        }

        try:
            make_infer_task_dir(item["slug"])
        except FileExistsError:
            return dash.no_update, {"type": "error", "text": f"資料夾已存在：{item['slug']}"}
        except Exception as e:
            return dash.no_update, {"type": "error", "text": f"建立資料夾失敗：{e}"}

        data.append(item)
        return data, None

    # 刪除部屬任務：只刪 data/deploy_tasks/<slug>
    if isinstance(trig, dict) and trig.get("type") == "infer-task-del":
        task_id = trig.get("task_id")
        if not task_id:
            return dash.no_update, {"type": "error", "text": "刪除失敗：缺少任務 ID"}

        idx = next((i for i, t in enumerate(data) if str(t.get("id")) == str(task_id)), -1)
        if idx == -1:
            return dash.no_update, {"type": "error", "text": "刪除失敗：找不到任務"}

        # 只有按鈕真的被點過才進來
        if not isinstance(del_clicks, (list, tuple)) or idx < 0 or idx >= len(del_clicks) or (del_clicks[idx] or 0) <= 0:
            raise PreventUpdate

        task = data[idx]

        # 👇 執行中禁止刪除（雙保險）
        if _is_task_running(run, task.get("id")):
            return dash.no_update, {"type": "error", "text": "此任務正在執行，請先停止再刪除"}

        slug = (task.get("slug") or slugify_name(task.get("name") or "")).strip()
        if not slug:
            return dash.no_update, {"type": "error", "text": "刪除失敗：部屬任務資料夾名無效"}

        task_dir = (DEPLOY_DIR_PATH / slug).resolve()
        if not _is_safe_subpath(DEPLOY_DIR_PATH, task_dir):
            return dash.no_update, {"type": "error", "text": "刪除失敗：不合法的部屬任務路徑"}

        try:
            if task_dir.is_dir():
                shutil.rmtree(task_dir)
        except Exception as e:
            return dash.no_update, {"type": "error", "text": f"刪除資料夾失敗：{e}"}

        del data[idx]
        return data, None

    raise PreventUpdate
    

@app.callback(
    Output("infer-task-list", "children"),
    Input("infer-tasks-store", "data"),
    Input("infer-run", "data")
)
def infer_tasks_render(data, run):
    rows = []
    for i, t in enumerate(data or []):
        if not t.get("id"):
            t["id"] = str(uuid.uuid4())

        is_running = _is_task_running(run, t["id"])
        run_label = "⏸" if is_running else "▶"

        rows.append(
            html.Div(
                [
                    html.Img(src=LOGO_DB_SRC or "", className="file-icon-img"),
                    html.Div([
                        html.Div(t.get("name", "(未命名)"), className="task-title"),
                        html.Div(t.get("desc", ""), className="task-desc"),
                        html.Div(t.get("time", ""), className="task-meta")
                    ], className="task-info"),
                    html.Div([
                        html.Button(
                            "✎",
                            id={"type": "infer-task-edit", "task_id": t["id"]},
                            n_clicks=0,
                            className="icon-btn",
                            disabled=is_running,
                            title="執行中，請先停止後再編輯" if is_running else "編輯",
                            key=f"edit-{t['id']}",
                        ),
                        html.Button(
                            run_label,
                            id={"type": "infer-task-run", "task_id": t["id"]},
                            n_clicks=0,
                            className="icon-btn",
                            key=f"run-{t['id']}",
                        ),
                        html.Button(
                            "🗑",
                            id={"type": "infer-task-del", "task_id": t["id"]},
                            n_clicks=0,
                            className="icon-btn",
                            key=f"del-{t['id']}",
                        ),
                    ], className="task-actions", key=f"actions-{t['id']}"),
                ],
                className="task-row",
                key=f"row-{t['id']}",   # ← 這行最重要：穩定的 row key
            )
        )
    return rows



@app.callback(
    Output("infer-run", "data", allow_duplicate=True),
    Output("notice-signal", "data", allow_duplicate=True),
    Output("infer-tasks-store", "data", allow_duplicate=True),  # 保持之前建議：順手回寫該列時間
    Input({"type": "infer-task-run", "task_id": ALL}, "n_clicks"),
    State("infer-tasks-store", "data"),
    State("infer-run", "data"),
    State("infer-run-mode", "data"),
    prevent_initial_call=True
)
def toggle_infer_runner(run_clicks, tasks, current_run, run_mode):
    if not any((n or 0) > 0 for n in (run_clicks or [])):
        raise PreventUpdate

    trig = ctx.triggered_id
    if not (isinstance(trig, dict) and trig.get("type") == "infer-task-run"):
        raise PreventUpdate

    # 找到被點擊的任務
    task_id = str(trig.get("task_id"))
    tasks = (tasks or []).copy()
    task = next((t for t in tasks if str(t.get("id")) == task_id), None)
    if not task:
        raise PreventUpdate

    slug = (task.get("slug") or "").strip()
    cfg = read_infer_config(slug) or {}
    minutes = int(cfg.get("detect_time"))
    per_task_interval_ms = minutes * 60 * 1000
    now_ms = int(time.time() * 1000)

    def now_ts():
        return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y/%m/%d %H:%M:%S")

    cur = (current_run or {}).copy()
    run_mode = (run_mode or "single").lower()

    # 相容舊格式
    active = cur.get("tasks") or []
    if cur.get("task"):
        active = [cur["task"]]
        cur.pop("task", None)

    exists = any(str(t.get("id")) == task_id for t in active)

    if exists:
        # 從執行清單移除
        active = [t for t in active if str(t.get("id")) != task_id]
        cur["tasks"] = active
        cur["enabled"] = bool(active)
        # 不動 interval_ms
        set_task_enabled(slug, False)

        # 更新該列時間（顯示最近一次動作）
        for i, t in enumerate(tasks):
            if str(t.get("id")) == task_id:
                tasks[i] = {**t, "time": now_ts()}
                break

        return cur, {"type": "success", "text": f"已停止：{task.get('name','')}"}, tasks
    else:
        # 加入執行清單
        active.append({
            "id": task.get("id"),
            "slug": slug,
            "name": task.get("name"),
            "period_ms": per_task_interval_ms,
            "next_due_ts": now_ms + per_task_interval_ms
        })
        cur["tasks"] = active
        cur["enabled"] = True

        set_task_enabled(slug, True)

        # 更新該列時間（開始執行時間）
        for i, t in enumerate(tasks):
            if str(t.get("id")) == task_id:
                tasks[i] = {**t, "time": now_ts()}
                break

        return cur, {"type": "success", "text": f"已加入執行：{task.get('name','')}（每 {minutes} 分）"}, tasks


@app.callback(
    Output("infer-run", "data", allow_duplicate=True),
    Input("infer-tasks-store", "data"),
    State("infer-run", "data"),
    prevent_initial_call=True
)
def sync_run_with_tasks(current_tasks, run):
    # 目前存在的任務 id
    alive_ids = {str(t.get("id")) for t in (current_tasks or [])}

    # 取出現行 run 狀態（相容舊單一格式）
    run = (run or {"enabled": False, "interval_ms": 60000, "tasks": []}).copy()
    active = run.get("tasks") or []
    if run.get("task"):          # 舊格式：單一任務
        active = [run["task"]]

    # 僅保留仍存在的任務
    active = [t for t in active if str(t.get("id")) in alive_ids]

    # 清掉舊格式欄位並回寫
    run.pop("task", None)
    run["tasks"] = active
    # enabled 僅在 active 非空而且原本為 True 時保持 True
    run["enabled"] = bool(active) and bool(run.get("enabled"))
    # interval_ms 維持原值（或預設）
    run["interval_ms"] = int(run.get("interval_ms") or 60000)
    return run

@app.callback(
    Output("infer-mode", "data"),
    Output("infer-editor-store", "data"),
    Output("notice-signal", "data", allow_duplicate=True),
    Input({"type": "infer-task-edit", "task_id": ALL}, "n_clicks"),
    Input("btn-back", "n_clicks"),
    State("infer-mode", "data"),
    State("infer-tasks-store", "data"),
    State("infer-run", "data"),
    prevent_initial_call=True
)
def go_infer_editor(edit_clicks, back_clicks, mode, tasks, run):
    trig = ctx.triggered_id

    if trig == "btn-back":
        return "list", {}, None

    if isinstance(trig, dict) and trig.get("type") == "infer-task-edit":
        task_id = trig.get("task_id")
        if not task_id:
            raise PreventUpdate

        # 找到對應的任務與它的 index
        tasks = tasks or []
        task = next((t for t in tasks if str(t.get("id")) == str(task_id)), None)
        if not task:
            raise PreventUpdate

        try:
            idx = [str(t.get("id")) for t in tasks].index(str(task_id))
        except ValueError:
            raise PreventUpdate

        # 檢查是不是被點擊
        if isinstance(edit_clicks, (list, tuple)):
            if idx < 0 or idx >= len(edit_clicks) or not edit_clicks[idx]:
                raise PreventUpdate

        # 如果任務正在執行 → 阻擋
        if _is_task_running(run, task.get("id")):
            return no_update, no_update, {"type": "error", "text": "此任務目前正在執行，請先停止再進入編輯"}

        # 否則正常進入編輯頁
        return "edit", {"task_id": task_id}, None

    raise PreventUpdate

@app.callback(
    Output("infer-runner", "disabled"),
    Output("infer-runner", "interval"),
    Input("infer-run", "data"),
    prevent_initial_call=False
)
def sync_runner_component(run):
    run = run or {}
    enabled = bool(run.get("enabled"))
    return (not enabled), 20000


@app.callback(
    Output("notice-signal", "data", allow_duplicate=True),
    Input("infer-run-signal", "data"),
    prevent_initial_call=True
)
def show_runner_tick(sig):
    if not sig:
        raise PreventUpdate
    ok = int(sig.get("ok", 0))
    fail = int(sig.get("fail", 0))
    ts = sig.get("ts", "")
    details = sig.get("details") or []
    note = (", ".join(details)) if details else "OK"
    typ = "success" if fail == 0 else "error"
    return {"type": typ, "text": f"例行執行（{ts}）：成功 {ok}／失敗 {fail}；{note}"}

@app.callback(Output("data-mode", "data", allow_duplicate=True),
              Input("nav-store", "data"), prevent_initial_call=True)
def reset_mode_on_nav(state):
    if state and state.get("top") == "project" and state.get("sub") == "data":
        return "list"
    raise PreventUpdate

@app.callback(
    Output("data-mode", "data"), 
    Output("editor-store", "data"),
    Input({"type": "task-edit", "task_id": ALL}, "n_clicks"),
    Input("btn-back", "n_clicks"),
    State("tasks-store", "data"),
    prevent_initial_call=True
)
def go_editor(edit_clicks, back_clicks, tasks):
    trig = ctx.triggered_id
    if trig == "btn-back":
        return "list", {}

    if isinstance(trig, dict) and trig.get("type") == "task-edit":
        task_id = str(trig.get("task_id"))
        tasks = tasks or []
        # 找出目前資料中的 index（仍可沿用你後續用 index 的地方）
        try:
            idx = [str(t.get("id")) for t in tasks].index(task_id)
        except ValueError:
            raise PreventUpdate

        # 確認這顆按鈕真的被點過
        if not any((edit_clicks or [])):
            raise PreventUpdate

        return "edit", {"index": idx}

    raise PreventUpdate

@app.callback(Output("selected-file", "data"),
              Input({"type": "nfile", "index": ALL}, "n_clicks"),
              Input({"type": "afile", "index": ALL}, "n_clicks"),
              State("normal-files", "data"), State("abnormal-files", "data"),
              prevent_initial_call=True)
def pick_file(nclicks_n, nclicks_a, normals, abns):
    trig = ctx.triggered_id
    if isinstance(trig, dict):
        idx = int(trig.get("index", -1))
        if trig.get("type") == "nfile" and normals and 0 <= idx < len(normals):
            item = normals[idx]
            return {"list": "normal", "name": item.get("name"), "path": item.get("path"), "content": item.get("content")}
        if trig.get("type") == "afile" and abns and 0 <= idx < len(abns):
            item = abns[idx]
            return {"list": "abnormal", "name": item.get("name"), "path": item.get("path"), "content": item.get("content")}
    raise PreventUpdate

@app.callback(Output("normal-list", "children"),
              Input("normal-files", "data"), Input("selected-file", "data"))
def render_normal_list(files, selected):
    files = files or []
    sel_name = selected.get("name") if (selected and selected.get("list") == "normal") else None
    out = []
    for i, f in enumerate(files):
        name = f["name"] if isinstance(f, dict) else str(f)
        cls = "file-row file file-btn" + (" active" if name == sel_name else "")
        out.append(html.Button(name, id={"type": "nfile", "index": i}, n_clicks=0, className=cls))
    return out

@app.callback(Output("abnormal-list", "children"),
              Input("abnormal-files", "data"), Input("selected-file", "data"))
def render_abnormal_list(files, selected):
    files = files or []
    sel_name = selected.get("name") if (selected and selected.get("list") == "abnormal") else None
    out = []
    for i, f in enumerate(files):
        name = f["name"] if isinstance(f, dict) else str(f)
        cls = "file-row file file-btn" + (" active" if name == sel_name else "")
        out.append(html.Button(name, id={"type": "afile", "index": i}, n_clicks=0, className=cls))
    return out

@app.callback(
    Output("editor-wave", "figure"),
    Input("selected-file", "data"),
    Input("btn-reset-zoom-editor", "n_clicks")
)

def update_editor_wave(selected, _):
    if selected:
        if selected.get("path") and os.path.exists(selected["path"]):
            return figure_from_csv_path(selected["path"])
        if selected.get("content"):
            return figure_from_csv_content(selected["content"])
    return empty_wave_fig()

@app.callback(Output("normal-files", "data", allow_duplicate=True),
              Output("files-changed-signal", "data", allow_duplicate=True),
              Input("u-normal-file", "contents"),
              Input("u-normal-folder", "contents"),
              Input("btn-normal-clear", "n_clicks"),
              State("u-normal-file", "filename"),
              State("u-normal-folder", "filename"),
              State("editor-store", "data"), State("tasks-store", "data"),
              prevent_initial_call=True)
def on_normal_upload_new(c1, c2, clear_clicks, fn1, fn2, editor, tasks):
    trig = ctx.triggered_id
    if not editor or tasks is None or editor.get("index") is None:
        raise PreventUpdate
    i = int(editor.get("index"))
    if i < 0 or i >= len(tasks or []):
        raise PreventUpdate
    task = tasks[i]
    normal_dir, _ = ensure_task_dirs(task, allow_migrate=False)

    if trig == "btn-normal-clear":
        ok = _safe_clear_dir(normal_dir, UPLOADS_DIR_PATH)
        if not ok:
            raise PreventUpdate
        return [], str(uuid.uuid4())

    names, contents = [], []
    if trig == "u-normal-file" and fn1 and c1:
        names = fn1 if isinstance(fn1, list) else [fn1]
        contents = c1 if isinstance(c1, list) else [c1]
    elif trig == "u-normal-folder" and fn2 and c2:
        names = fn2 if isinstance(fn2, list) else [fn2]
        contents = c2 if isinstance(c2, list) else [c2]
    else:
        raise PreventUpdate

    for n, cont in zip(names, contents):
        if not n or not cont:
            continue
        base = _sanitize_filename(os.path.basename(str(n).replace("\\", "/")))
        path = os.path.join(normal_dir, base)
        save_b64_to_file(cont, path)

    scan_n, _ = scan_task_files(task)
    return scan_n, str(uuid.uuid4())

@app.callback(Output("abnormal-files", "data", allow_duplicate=True),
              Output("files-changed-signal", "data", allow_duplicate=True),
              Input("u-abnormal-file", "contents"),
              Input("u-abnormal-folder", "contents"),
              Input("btn-abnormal-clear", "n_clicks"),
              State("u-abnormal-file", "filename"),
              State("u-abnormal-folder", "filename"),
              State("editor-store", "data"), State("tasks-store", "data"),
              prevent_initial_call=True)
def on_abnormal_upload_new(c1, c2, clear_clicks, fn1, fn2, editor, tasks):
    trig = ctx.triggered_id
    if not editor or tasks is None or editor.get("index") is None:
        raise PreventUpdate
    i = int(editor.get("index"))
    if i < 0 or i >= len(tasks or []):
        raise PreventUpdate
    task = tasks[i]
    _, abnormal_dir = ensure_task_dirs(task, allow_migrate=False)

    if trig == "btn-abnormal-clear":
        ok = _safe_clear_dir(abnormal_dir, UPLOADS_DIR_PATH)
        if not ok:
            raise PreventUpdate
        return [], str(uuid.uuid4())

    names, contents = [], []
    if trig == "u-abnormal-file" and fn1 and c1:
        names = fn1 if isinstance(fn1, list) else [fn1]
        contents = c1 if isinstance(c1, list) else [c1]
    elif trig == "u-abnormal-folder" and fn2 and c2:
        names = fn2 if isinstance(fn2, list) else [fn2]
        contents = c2 if isinstance(c2, list) else [c2]
    else:
        raise PreventUpdate

    for n, cont in zip(names, contents):
        if not n or not cont:
            continue
        base = _sanitize_filename(os.path.basename(str(n).replace("\\", "/")))
        path = os.path.join(abnormal_dir, base)
        save_b64_to_file(cont, path)

    _, scan_a = scan_task_files(task)
    return scan_a, str(uuid.uuid4())

@app.callback(
    Output("tasks-store", "data"),
    Output("notice-signal", "data"),
    Input("url", "href"),
    Input("btn-add-task", "n_clicks"),
    Input({"type": "task-del", "task_id": ALL}, "n_clicks"),
    State("task-name", "value"), State("task-desc", "value"),
    State("tasks-store", "data"),
    prevent_initial_call=False
)
def tasks_store_manager(href, add_clicks, del_clicks, name, desc, data):
    trig = ctx.triggered_id

    if trig == "url":
        if data:
            raise PreventUpdate
        data = scan_uploads_for_tasks() or []
        for t in data:
            t.setdefault("slug", (t.get("name") or "").strip())
            if not t.get("id"):
                t["id"] = hashlib.md5(t["slug"].encode("utf-8")).hexdigest()[:12]
            else:
                t["id"] = str(t["id"])
        return data, None

    data = (data or []).copy()

    if trig == "btn-add-task":
        name_clean = (name or "").strip()
        if not name_clean:
            return dash.no_update, {"type": "error", "text": "請輸入任務名稱"}
        if not all(ch.isalnum() or ch in "-_ " for ch in name_clean):
            return dash.no_update, {"type": "error", "text": "任務名稱只能包含英數字、空白、底線或連字號"}

        new_slug = slugify_name(name_clean)
        if not new_slug:
            return dash.no_update, {"type": "error", "text": "此名稱無法轉為合法資料夾，請改用英數或可轉寫字元"}

        existing_slugs = { (t.get("slug") or slugify_name(t.get("name") or "")) for t in data }
        folder_path = (UPLOADS_DIR_PATH / new_slug).resolve()
        if new_slug in existing_slugs or folder_path.exists():
            return dash.no_update, {"type": "error", "text": f"已存在同名任務或資料夾：{name_clean}"}

        item = {
            "id": str(uuid.uuid4()),
            "name": name_clean,
            "slug": new_slug,
            "desc": (desc or ""),
            "time": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y/%m/%d %H:%M:%S"),
            "normal": [],
            "abnormal": []
        }

        try:
            make_task_dir(item["slug"])
        except FileExistsError:
            return dash.no_update, {"type": "error", "text": f"資料夾已存在：{item['slug']}"}
        except Exception as e:
            return dash.no_update, {"type": "error", "text": f"建立資料夾失敗：{e}"}

        data.append(item)
        return data, None

    # -------- 刪除任務（以 task_id）--------
    if isinstance(trig, dict) and trig.get("type") == "task-del":
        task_id = trig.get("task_id")
        if not task_id:
            return dash.no_update, {"type": "error", "text": "刪除失敗：缺少任務 ID"}

        # 以 ID 尋找索引
        idx = next((i for i, t in enumerate(data) if str(t.get("id")) == str(task_id)), -1)
        if idx == -1:
            return dash.no_update, {"type": "error", "text": "刪除失敗：找不到任務"}

        # 🔐 重要：只有按鈕真的被點擊過（n_clicks > 0）才允許刪除
        if not isinstance(del_clicks, (list, tuple)) or idx < 0 or idx >= len(del_clicks) or (del_clicks[idx] or 0) <= 0:
            raise PreventUpdate

        task = data[idx]
        slug = (task.get("slug") or slugify_name(task.get("name") or "")).strip()
        if not slug:
            return dash.no_update, {"type": "error", "text": "刪除失敗：任務資料夾名無效"}

        task_dir = (UPLOADS_DIR_PATH / slug).resolve()
        if not _is_safe_subpath(UPLOADS_DIR_PATH, task_dir):
            return dash.no_update, {"type": "error", "text": "刪除失敗：不合法的資料夾路徑"}

        try:
            if task_dir.is_dir():
                shutil.rmtree(task_dir)
        except Exception as e:
            return dash.no_update, {"type": "error", "text": f"刪除資料夾失敗：{e}"}

        del data[idx]
        return data, None

@app.callback(Output("task-list", "children"), Input("tasks-store", "data"))
def tasks_render(data):
    rows = []
    for t in (data or []):
        task_id = str(t.get("id"))
        rows.append(
            html.Div(
                [
                    html.Img(src=LOGO_DB_SRC or "", className="file-icon-img"),
                    html.Div([
                        html.Div(t.get("name", "(未命名)"), className="task-title"),
                        html.Div(t.get("desc", ""), className="task-desc"),
                        html.Div(t.get("time", ""), className="task-meta")
                    ], className="task-info"),
                    html.Div([
                        # 第 2 點會把這顆 edit 改成用 task_id（見下）
                        html.Button("✎", id={"type": "task-edit", "task_id": task_id},
                                    n_clicks=0, className="icon-btn"),
                        html.Button("🗑", id={"type": "task-del", "task_id": task_id},
                                    n_clicks=0, className="icon-btn"),
                    ], className="task-actions")
                ],
                className="task-row",
                key=task_id,                # 👈 固定這一列的身份
            )
        )
    return rows

@app.callback(
    Output("msg-toast", "style"), Output("msg-text", "children"), Output("msg-timer", "disabled"),
    Input("notice-signal", "data"), Input("msg-timer", "n_intervals"),
    prevent_initial_call=True
)
def _toggle_msg(sig, n_int):
    show = {"display": "flex"}
    hide = {"display": "none"}
    trig = ctx.triggered_id
    if trig == "notice-signal" and sig:
        return show, sig.get("text", ""), False
    return hide, dash.no_update, True

@app.callback(
    Output("normal-files", "data", allow_duplicate=True),
    Output("abnormal-files", "data", allow_duplicate=True),
    Input("editor-store", "data"), State("tasks-store", "data"),
    prevent_initial_call=True
)
def load_files_for_task(editor, tasks):
    if not editor or tasks is None:
        raise PreventUpdate
    idx = editor.get("index")
    if idx is None:
        raise PreventUpdate
    i = int(idx)
    if i < 0 or i >= len(tasks or []):
        raise PreventUpdate
    t = tasks[i]
    normal_files, abnormal_files = scan_task_files(t)
    return normal_files, abnormal_files

@app.callback(
    Output("tasks-store", "data", allow_duplicate=True),
    Input("files-changed-signal", "data"),
    State("normal-files", "data"), State("abnormal-files", "data"),
    State("editor-store", "data"), State("tasks-store", "data"),
    prevent_initial_call=True
)
def persist_files_to_task(_changed_signal, normals, abns, editor, tasks):
    if not _changed_signal:
        raise PreventUpdate
    if not editor or tasks is None:
        raise PreventUpdate
    idx = editor.get("index")
    if idx is None:
        raise PreventUpdate
    i = int(idx)
    if i < 0 or i >= len(tasks or []):
        raise PreventUpdate
    tasks = (tasks or []).copy()
    t = tasks[i].copy()
    def sanitize(lst):
        out = []
        for it in (lst or []):
            if isinstance(it, dict):
                out.append({"name": it.get("name"), "path": it.get("path")})
        return out
    t["normal"] = sanitize(normals)
    t["abnormal"] = sanitize(abns)
    tasks[i] = t
    return tasks

@app.callback(
    Output("nav-store", "data"),
    Input("nav-top-project", "n_clicks"), Input("nav-top-infer", "n_clicks"),
    Input("nav-sub-data", "n_clicks"), Input("nav-sub-analyze", "n_clicks"),
    Input("nav-sub-setting", "n_clicks"),
    State("nav-store", "data"), prevent_initial_call=True
)
def nav_update(c1, c2, c3, c4, c5, state):
    state = state or {"top": None, "sub": None}
    trig = ctx.triggered_id
    if trig == "nav-top-project":
        # 直接帶預設子頁：資料
        return {"top": "project", "sub": "data"}
    elif trig == "nav-top-infer":
        # 直接帶預設子頁：設定
        return {"top": "infer", "sub": "setting"}
    elif trig == "nav-sub-data":
        return {"top": "project", "sub": "data"}
    elif trig == "nav-sub-analyze":
        return {"top": "project", "sub": "analyze"}
    elif trig == "nav-sub-setting":
        return {"top": "infer", "sub": "setting"}
    return state

@app.callback(
    Output("infer-mode", "data", allow_duplicate=True),
    Input("nav-store", "data"),
    prevent_initial_call=True
)
def reset_infer_mode_on_nav(state):
    if state and state.get("top") == "infer" and state.get("sub") == "setting":
        return "list"
    raise PreventUpdate
    
@app.callback(Output("btn-back", "style"),
              Input("nav-store", "data"), 
              Input("data-mode", "data"),
              Input("infer-mode", "data"))
def show_back(state, mode, infer_mode):
    show, hide = {"display": "inline-flex"}, {"display": "none"}
    if state and state.get("top") == "project" and state.get("sub") == "data" and mode == "edit":
        return show
    if state and state.get("top") == "infer" and state.get("sub") == "setting" and infer_mode == "edit":
        return show
    return hide

@app.callback(Output("nav-top-project", "className"), Output("nav-top-infer", "className"),
              Output("subnav-project", "style"), Output("subnav-infer", "style"),
              Output("nav-sub-data", "className"), Output("nav-sub-analyze", "className"),
              Output("nav-sub-setting", "className"),
              Input("nav-store", "data"))
def paint_nav(state):
    state = state or {"top": None, "sub": None}
    top = state.get("top")
    sub = state.get("sub")

    def active(base_cls: str, cond: bool) -> str:
        return f"{base_cls} active" if cond else base_cls

    # 主選單高亮
    proj_cls  = active("nav-top-item", top == "project")
    infer_cls = active("nav-top-item", top == "infer")

    # 子選單顯示/隱藏
    show_col = {"display": "flex", "flexDirection": "column", "gap": "6px"}
    hide     = {"display": "none"}
    sub_style_project = show_col if top == "project" else hide
    sub_style_infer   = show_col if top == "infer"   else hide

    # 專案子選單高亮
    proj_data_cls    = active("nav-item", top == "project" and sub == "data")
    proj_analyze_cls = active("nav-item", top == "project" and sub == "analyze")

    # 部屬子選單高亮（此例只有「設定」）
    infer_setting_cls = active("nav-item", top == "infer" and sub == "setting")

    return (
        proj_cls, infer_cls,
        sub_style_project, sub_style_infer,
        proj_data_cls, proj_analyze_cls, infer_setting_cls
    )

@app.callback(
    Output("brand-title", "children"),
    Input("data-mode", "data"),
    State("editor-store", "data"),
    State("tasks-store", "data"),
    Input("infer-mode", "data"),
    State("infer-editor-store", "data"),
    State("infer-tasks-store", "data"),
)
def update_brand(mode, editor, tasks, infer_mode, infer_editor, infer_tasks):
    # 專案編輯（保持原樣）
    if mode == "edit" and editor and tasks is not None:
        try:
            idx = int(editor.get("index", -1))
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(tasks or []):
            name = (tasks[idx].get("name") or "(未命名)").strip() or "(未命名)"
            return [html.Span(name)]

    # 部屬編輯（改為 task_id）
    if infer_mode == "edit" and infer_editor and infer_tasks is not None:
        task_id = str(infer_editor.get("task_id") or "")
        if task_id:
            t = next((x for x in (infer_tasks or []) if str(x.get("id")) == task_id), None)
            if t:
                name = (t.get("name") or "(未命名)").strip() or "(未命名)"
                return [html.Span(name)]

    return [html.Span("L3C"), html.Span(" 異常檢測", style={"opacity": .7})]

@app.callback(
    Output("normal-files", "data", allow_duplicate=True),
    Output("abnormal-files", "data", allow_duplicate=True),
    Output("selected-file", "data", allow_duplicate=True),
    Output("editor-wave", "figure", allow_duplicate=True),
    Input("btn-back", "n_clicks"),
    prevent_initial_call=True
)
def clear_editor_states(n):
    if not n:
        raise PreventUpdate
    return [], [], None, empty_wave_fig()


@app.callback(
    Output("anal-selected", "data"),
    Input("anal-normal-ck", "value"),
    Input("anal-abnormal-ck", "value"),
    prevent_initial_call=False
)
def _save_anal_selected(n_vals, a_vals):
    return {"normal": (n_vals or []), "abnormal": (a_vals or [])}


@app.callback(
    Output("notice-signal", "data", allow_duplicate=True),
    Output("detect-results", "data"),
    Output("result-dd", "options"),
    Output("result-dd", "value"),
    Input("btn-detect", "n_clicks"),
    State("model-dd", "value"),
    State("anal-selected", "data"),
    prevent_initial_call=True
)
def run_detection(n, model_name, selected):
    if not n:
        raise PreventUpdate

    sel = selected or {"normal": [], "abnormal": []}
    n_files = list(sel.get("normal") or [])
    a_files = list(sel.get("abnormal") or [])
    total_ct = len(n_files) + len(a_files)
    if total_ct == 0:
        return {"type": "error", "text": "請先在左側選擇正常/異常資料"}, dash.no_update, dash.no_update, dash.no_update

    # ---- 分批切塊（各自照 MAX_FILES_PER_REQUEST 切）----
    n_batches = list(_chunks(n_files, MAX_FILES_PER_REQUEST))
    a_batches = list(_chunks(a_files, MAX_FILES_PER_REQUEST))

    # 為了讓每批都有東西：依序配對 normal/abnormal 的批次，較長者剩餘批次單獨送
    max_batches = max(len(n_batches), len(a_batches))
    combined_results = []
    ok_batches = 0
    fail_batches = 0

    for i in range(max_batches):
        n_part = n_batches[i] if i < len(n_batches) else []
        a_part = a_batches[i] if i < len(a_batches) else []

        payload = {
            "model_name": model_name,
            "normal": n_part,
            "abnormal": a_part
        }
        # 動態 timeout：基礎 + 每檔加總
        batch_ct = len(n_part) + len(a_part)
        timeout_sec = REQ_BASE_TIMEOUT_SEC + REQ_PER_FILE_SEC * batch_ct

        try:
            data = _post_with_retry(
                urlGetPrediction,
                json_payload=payload,
                timeout=timeout_sec,
                max_retry=REQ_MAX_RETRY
            )
            # ---- 正規化後端回覆（延用你既有規則）----
            results = []
            if isinstance(data, list):
                results = [it for it in data if isinstance(it, dict)]
            elif isinstance(data, dict):
                if isinstance(data.get("results"), list):
                    results = [it for it in data["results"] if isinstance(it, dict)]
                else:
                    tmp = []
                    for k in ("normal", "abnormal"):
                        arr = data.get(k)
                        if isinstance(arr, list):
                            for it in arr:
                                if isinstance(it, dict):
                                    if "kind" not in it:
                                        it = {**it, "kind": k}
                                    tmp.append(it)
                    results = tmp

            if results:
                combined_results.extend(results)
                ok_batches += 1
            else:
                fail_batches += 1

        except requests.exceptions.Timeout:
            fail_batches += 1
        except Exception:
            fail_batches += 1

    if not combined_results:
        return {"type": "error", "text": f"推論失敗或無結果（共 {max_batches} 批，成功 {ok_batches} 批，失敗 {fail_batches} 批）"}, dash.no_update, dash.no_update, dash.no_update

    # Dropdown options / value（沿用你的格式）
    opts = []
    for i, it in enumerate(combined_results):
        from pathlib import Path as _Path
        fname = _Path(str(it.get("file", ""))).name if isinstance(it.get("file", ""), (str, _Path)) else ""
        kind  = it.get("kind", "?")
        label = f"{kind} · {fname or f'item#{i}'}"
        opts.append({"label": label, "value": i})
    val = opts[0]["value"] if opts else None

    msg = {"type": "success",
           "text": f"檢測完成，共 {len(combined_results)} 筆；批次：成功 {ok_batches}、失敗 {fail_batches}；每批上限 {MAX_FILES_PER_REQUEST}。"}
    return msg, combined_results, opts, val


@app.callback(
    Output("analysis-title", "children"),
    Input("detect-results", "data"),
    Input("result-dd", "value"),
    prevent_initial_call=True
)
def update_analysis_title(results, selected_index):
    if not results or not isinstance(results, list):
        return "分析數據圖"
    try:
        i = int(selected_index) if selected_index is not None else 0
    except (TypeError, ValueError):
        i = 0
    if i < 0 or i >= len(results):
        i = 0
    item = results[i]
    fname = Path(str(item.get("file", ""))).name or "(未命名)"
    kind  = (item.get("kind") or "").strip()
    return f"{fname} ({kind})" if kind else fname


@app.callback(
    Output("wave", "figure"),
    Input("detect-results", "data"),      
    Input("result-dd", "value"),          
    Input("btn-reset-zoom", "n_clicks"),  
    Input("thres-slider", "value"),
    prevent_initial_call=True
)
def draw_detected_figure(results, selected_index, reset_clicks, thres_value):
    if not results or not isinstance(results, list):
        fig = empty_wave_fig()
        return fig

    try:
        i = int(selected_index) if selected_index is not None else 0
    except (TypeError, ValueError):
        i = 0
    if i < 0 or i >= len(results):
        i = 0

    fig = figure_from_detection_item(results[i])

    fig.update_layout(uirevision=int(reset_clicks or 0))

    try:
        # Plotly 提供的 API（需要 plotly>=5）：add_hline
        fig.add_hline(y=float(thres_value), line_color="black", line_width=1, opacity=0.9)
        # 也可在圖標題右上顯示當前閥值（可選）
        # fig.update_layout(title=f"目前閥值：{float(thres_value):.2f}")
    except Exception:
        pass

    return fig

app.clientside_callback(
    """
    function(n_clicks, results, notice) {
        // 有結果或有訊息（成功/錯誤）→ 解鎖並還原文字
        if (results || (notice && notice.type)) {
            return [false, "開始分析"];
        }
        // 有點擊但尚未有結果/訊息 → 顯示分析中並鎖定
        if (n_clicks && n_clicks > 0) {
            return [true, "⏳ 分析中…"];
        }
        return [false, "開始分析"];
    }
    """,
    Output("btn-detect", "disabled"),
    Output("btn-detect", "children"),
    Input("btn-detect", "n_clicks"),
    Input("detect-results", "data"),   # 成功時會更新
    Input("notice-signal", "data"),    # 失敗/錯誤時會更新
    prevent_initial_call=True
)

@app.callback(
    Output("thres-value", "children"),
    Input("thres-slider", "value")
)
def _show_thres_value(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "-"
    
# ---------- CSS & folder-picker ----------
app.index_string = app_index_string

# ---------- Entrypoint ----------
if __name__ == "__main__":
    def find_free_port(start: int = 8050, limit: int = 20) -> int:
        for p in range(start, start + limit):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return start

    # try:
    #     with open(os.path.join(UPLOADS_DIR, "test.txt"), "w", encoding="utf-8") as f:
    #         f.write("hello")
    #     print("[selftest] 已寫入：", os.path.join(UPLOADS_DIR, "test.txt"))
    # except Exception as e:
    #     print("[selftest] 寫入失敗：", e)

    # cfg_path = os.path.join(DATA_DIR, "config.json")
    # if os.path.exists(cfg_path):
    #     with open(cfg_path, "r", encoding="utf-8") as f:
    #         cfg_text = f.read()
    #     print("[selftest] 讀到 config.json，長度=", len(cfg_text))
    # else:
    #     print("[selftest] 找不到 config.json（可忽略）")

    # 你原本的開啟瀏覽器 & run
    port = find_free_port()
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=True, host="127.0.0.1", port=port, use_reloader=False)
    
    
