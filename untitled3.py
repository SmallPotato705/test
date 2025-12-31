import os
import json
from datetime import datetime
from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from langflow.custom.custom_component.component_with_cache import ComponentWithCache
from langflow.io import MessageTextInput, SliderInput, Output, IntInput, BoolInput
from langflow.logging import logger


# ====== 本機 state 目錄（跟你 AnalyzeFeatureReport 一致：用 cwd/state）======
BASE_DIR = os.getcwd()
STATE_DIR = os.path.join(BASE_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)


def _error_response(tool_name: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "message": f"{tool_name} failed: {message}"}


def _is_blank(v: Any) -> bool:
    return not (isinstance(v, str) and v.strip())


def _state_path(state_uri: str) -> str:
    # 允許傳檔名或絕對路徑
    if os.path.isabs(state_uri):
        return state_uri
    return os.path.join(STATE_DIR, state_uri)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 state 檔案：{path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("state JSON 不是 dict（object）")
    return obj


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class OllamaFeatureInsightWriterComponent(ComponentWithCache):
    display_name = "Ollama Feature Insight Writer (Local JSON)"
    description = (
        "不呼叫任何 API；直接讀取本機 state JSON 檔，"
        "針對每個 feature 用 Ollama 產生 insight，"
        "最後寫回同一份 JSON 檔。"
    )
    icon = "Ollama"
    name = "ollama_feature_insight_writer"

    inputs = [
        MessageTextInput(
            name="state_uri",
            display_name="state_uri（檔名或絕對路徑）",
            info="例如: feature_xxxxxx.json（建議只放檔名；會從 cwd/state/ 讀）",
            value="",
            tool_mode=True,
        ),
        MessageTextInput(
            name="ollama_base_url",
            display_name="Ollama Base URL",
            info="例如: http://host.docker.internal:11434",
            value="http://host.docker.internal:11434",
        ),
        MessageTextInput(
            name="model_name",
            display_name="Model Name",
            info="例如: llama3.1, qwen2.5, mistral...（需已 pull）",
            value="llama3.1",  # ✅ 給預設值，避免空字串爆炸
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec={"min": 0, "max": 1, "step": 0.01},
            advanced=True,
        ),
        MessageTextInput(
            name="system",
            display_name="System",
            info="系統訊息（可留空）",
            value="",
            advanced=True,
        ),
        IntInput(
            name="insight_min_chars",
            display_name="Insight Min Chars",
            value=30,
            advanced=True,
        ),
        IntInput(
            name="insight_max_chars",
            display_name="Insight Max Chars",
            value=100,
            advanced=True,
        ),
        IntInput(
            name="timeout_sec",
            display_name="Timeout (sec)",
            value=60,
            advanced=True,
        ),
        BoolInput(
            name="overwrite_existing",
            display_name="Overwrite existing insight",
            info="False: 只補空的 insight；True: 全部重寫",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="state_uri", name="out_state_uri", method="run"),
    ]

    # ---------- helpers ----------
    def _build_llm(self) -> ChatOllama:
        model = str(self.model_name).strip()
        if not model:
            raise ValueError("model_name 必填（例如 llama3.1 / qwen2.5 / mistral）")

        sys_msg = (str(self.system).strip() if self.system else "")
        # ChatOllama 的 system 參數給 None 比給 "" 安全
        system_param = sys_msg if sys_msg else None

        return ChatOllama(
            base_url=str(self.ollama_base_url).rstrip("/"),
            model=model,
            temperature=float(self.temperature),
            system=system_param,
            request_timeout=float(self.timeout_sec),
        )

    def _prompt_for_feature(self, f: dict) -> str:
        name = f.get("name", "UNKNOWN")

        metrics = f.get("metrics") or {}
        charts = f.get("charts") or {}
        stats = (charts.get("stats") or {})

        ts = charts.get("ts") or {}
        raw = ts.get("raw") or []
        n = len(raw) if isinstance(raw, list) else int(ts.get("n") or 0)

        acf = charts.get("acf") or {}
        acf_lags = 0
        if isinstance(acf.get("lags"), list) and len(acf["lags"]) > 0:
            acf_lags = len(acf["lags"]) - 1

        # 自動展開 metrics
        metrics_lines = [f"- {k}: {v}" for k, v in (metrics.items() if isinstance(metrics, dict) else [])]

        return (
            "你是時序異常偵測前的特徵評估助理。"
            "請以繁體中文撰寫一段專業 insight，"
            f"長度約 {int(self.insight_min_chars)}–{int(self.insight_max_chars)} 字。\n"
            "請直接下結論，不要條列、不需重複數值或逐項描述指標。\n"
            "請綜合統計結果判斷此欄位是否適合進行異常偵測，"
            "並說明較合適的偵測策略類型（例如統計型、規則型或模型），"
            "同時指出使用上的主要風險或限制條件。"
            "最後請簡要說明在實際應用中，異常結果應如何解讀，"
            "例如是單點異常、連續偏移，或需搭配其他訊號判斷。\n\n"
            f"Feature: {name}\n"
            f"資料點數 N: {n}\n"
            f"ACF lags: {acf_lags}\n\n"
            "Metrics:\n" + ("\n".join(metrics_lines) if metrics_lines else "- (none)") + "\n\n"
            "Stats:\n"
            f"- min: {stats.get('min')}\n"
            f"- max: {stats.get('max')}\n"
            f"- mean: {stats.get('mean')}\n"
            f"- std: {stats.get('std')}\n"
            f"- q1: {stats.get('q1')}\n"
            f"- q3: {stats.get('q3')}\n"
        )

    # ---------- main ----------
    def run(self, state_uri: str = "", **kwargs: Any) -> Dict[str, Any]:
        tool_name = "ollama_feature_insight_writer"

        try:
            # 1) 決定 state_uri
            state_uri = (state_uri or str(self.state_uri)).strip()
            if not state_uri:
                return _error_response(tool_name, "state_uri 必填（例如 feature_xxxxxx.json）")

            path = _state_path(state_uri)

            # 2) 讀取 state JSON
            state = _load_json(path)

            features: List[dict] = state.get("features") or []
            if not isinstance(features, list):
                return _error_response(tool_name, "state['features'] 不是 list，請確認 state JSON 結構")

            # 3) 建立 LLM（不呼叫 API，只有本機 Ollama）
            llm = self._build_llm()

            updated = 0
            skipped = 0
            errors = 0

            overwrite = bool(self.overwrite_existing)

            for f in features:
                if not isinstance(f, dict):
                    continue

                if (not overwrite) and (not _is_blank(f.get("insight"))):
                    skipped += 1
                    continue

                try:
                    prompt = self._prompt_for_feature(f)
                    resp = llm.invoke([HumanMessage(content=prompt)])
                    text = (getattr(resp, "content", "") or "").strip()
                    text = " ".join(text.split())  # 單行化

                    # 若模型回傳空字串，保底給一句（避免寫回空）
                    if not text:
                        text = "此特徵可用於異常偵測，但需先確認資料品質與取樣一致性，並搭配合適的偵測策略解讀。"

                    f["insight"] = text
                    updated += 1
                except Exception as e:
                    errors += 1
                    logger.exception("insight generation failed for feature=%s", f.get("name"))
                    # 不中斷整批，記錄錯誤即可
                    f.setdefault("_insight_error", str(e))

            # 4) 寫回 JSON（同一檔）
            state["_insight_writer"] = {
                "updated_features": int(updated),
                "skipped_features": int(skipped),
                "error_features": int(errors),
                "overwrite_existing": overwrite,
                "state_uri": state_uri,
                "saved_at": datetime.now().isoformat(),
            }
            _save_json(path, state)

            # 5) 回傳
            return {
                "status": "success",
                "message": "已從本機讀取 state，產生 insight，並寫回同一份 JSON（未呼叫任何 State API）。",
                "out_state_uri": state_uri,
                "saved_path": path,
                "updated_features": updated,
                "skipped_features": skipped,
                "error_features": errors,
            }

        except Exception as e:
            logger.exception("%s failed", tool_name)
            return _error_response(tool_name, str(e))







# =============================================================================
# 
# =============================================================================
import os
import json
import uuid
from typing import Any, Dict, List
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd

from langflow.custom import Component
from langflow.io import StrInput, Output
from langflow.schema import Data

from langchain_core.tools import tool


# ========================
# 目錄設定（Langflow 內用 cwd 較穩）
# ========================
BASE_DIR = os.getcwd()
STATE_DIR = os.path.join(BASE_DIR, "state")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ========================
# HTML 模板：請把你原本的 HTML_TEMPLATE 整段貼在這裡
# ========================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__REPORT_TITLE__</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: #f1f5f9;
            color: #1e293b;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border-radius: 1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        @keyframes loadWidth {
            from { width: 0; }
            to { width: var(--target-width); }
        }
        .progress-bar-fill {
            animation: loadWidth 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }

        select {
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
        }

        input[type=range] { -webkit-appearance: none; background: transparent; }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%;
            background: #6366f1; cursor: pointer; margin-top: -6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3); transition: transform 0.1s;
        }
        input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.1); }
        input[type=range]::-webkit-slider-runnable-track {
            width: 100%; height: 4px; cursor: pointer; background: #e2e8f0; border-radius: 2px;
        }
        input[type=range]:disabled::-webkit-slider-thumb { background: #cbd5e1; cursor: not-allowed; }

        .cursor-help { cursor: help; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
    </style>
</head>
<body class="min-h-screen pb-12">

    <!-- Header -->
    <nav class="bg-slate-900 text-white shadow-lg sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center gap-3">
                    <div class="bg-indigo-500/20 p-2 rounded-lg border border-indigo-500/30">
                        <i data-lucide="activity" class="text-indigo-400 w-5 h-5"></i>
                    </div>
                    <div>
                        <h1 class="text-lg font-bold tracking-tight text-slate-100">時序異常偵測</h1>
                        <p class="text-[10px] text-slate-400 font-medium tracking-wide uppercase">Feature Analysis Dashboard</p>
                    </div>
                </div>

                <div class="hidden md:flex items-center gap-8">
                    <div class="text-right">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Features</p>
                        <p class="text-sm font-mono font-bold text-indigo-400" id="total-features-count">--</p>
                    </div>
                    <div class="h-8 w-px bg-slate-800"></div>
                    <div class="text-right">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Points</p>
                        <p class="text-sm font-mono font-bold text-emerald-400" id="points-disp">N=--</p>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        <!-- 1. AI Diagnosis -->
        <div class="glass-card p-0 overflow-hidden border-indigo-100 ring-1 ring-indigo-50 shadow-md mb-8">
            <div class="flex flex-col sm:flex-row">
                <div class="bg-gradient-to-br from-indigo-600 to-violet-700 p-5 sm:w-56 flex flex-row sm:flex-col items-center sm:items-start justify-center sm:justify-center gap-3 text-white shrink-0">
                    <div class="bg-white/20 p-2.5 rounded-xl backdrop-blur-sm shadow-inner">
                        <i data-lucide="sparkles" class="w-6 h-6"></i>
                    </div>
                    <div>
                        <span class="text-sm font-bold uppercase tracking-wider block">AI Diagnosis</span>
                        <span class="text-[10px] text-indigo-200 block sm:mt-1">智能分析建議</span>
                    </div>
                </div>
                <div class="p-6 bg-white/80 flex-1 flex items-center">
                    <div class="w-full">
                        <p class="text-sm leading-7 text-slate-700 font-medium whitespace-pre-line" id="llm-insight">
                            Loading analysis...
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- ✅ items-stretch：左右同高更好看 -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">

            <!-- ✅ 左欄改成 flex 讓下方卡片可撐滿 -->
            <aside class="lg:col-span-3 flex flex-col gap-5">

                <div class="glass-card p-5">
                    <label class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Target Feature</label>
                    <div class="relative group">
                        <select id="featureSelector" class="w-full bg-slate-50 border border-slate-200 text-slate-700 font-medium text-sm rounded-lg py-3 px-4 pr-10 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all cursor-pointer hover:bg-white hover:shadow-sm">
                        </select>
                        <div class="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none text-slate-400 group-hover:text-indigo-500 transition-colors">
                            <i data-lucide="chevron-down" class="w-4 h-4"></i>
                        </div>
                    </div>
                </div>

                <!-- ✅ flex-1：拉長填滿左欄 -->
                <div class="glass-card p-5 space-y-6 flex-1">
                    <h3 class="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-2 mb-4">Statistical Health</h3>

                    <div>
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-xs font-medium text-slate-500 flex items-center gap-1.5 cursor-help group" title="平穩性 (Stationarity)：衡量數據的統計特性（均值、變異數）是否隨時間穩定。">
                                Stationarity
                                <i data-lucide="info" class="w-3 h-3 text-slate-300 group-hover:text-indigo-400 transition-colors"></i>
                            </span>
                            <span class="text-sm font-bold font-mono" id="val-stat-text">--</span>
                        </div>
                        <div class="relative w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div id="bar-stat" class="absolute top-0 left-0 h-full rounded-full progress-bar-fill transition-colors duration-300" style="--target-width: 0%"></div>
                        </div>
                    </div>

                    <div>
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-xs font-medium text-slate-500 flex items-center gap-1.5 cursor-help group" title="自相關性 (ACF Lag-1)：衡量當前數值與前一時刻的相關程度。">
                                Autocorrelation
                                <i data-lucide="info" class="w-3 h-3 text-slate-300 group-hover:text-indigo-400 transition-colors"></i>
                            </span>
                            <span class="text-sm font-bold font-mono" id="val-acf-text">--</span>
                        </div>
                         <div class="flex gap-1 h-1.5">
                            <div id="acf-seg-1" class="flex-1 bg-slate-100 rounded-l-full transition-colors duration-300"></div>
                            <div id="acf-seg-2" class="flex-1 bg-slate-100 transition-colors duration-300"></div>
                            <div id="acf-seg-3" class="flex-1 bg-slate-100 rounded-r-full transition-colors duration-300"></div>
                         </div>
                         <div class="text-[10px] text-right mt-1 font-medium" id="tag-acf">--</div>
                    </div>

                    <div class="grid grid-cols-2 gap-3 pt-2">
                        <div class="bg-slate-50 p-3 rounded-lg border border-slate-100">
                            <span class="text-[10px] text-slate-400 font-bold uppercase block mb-1 flex items-center gap-1 cursor-help" title="缺值率 (Missing Rate)：時序模型對連續性要求高。">
                                Missing <i data-lucide="help-circle" class="w-3 h-3 text-slate-300"></i>
                            </span>
                            <span class="text-sm font-bold text-slate-700 font-mono" id="val-missing">--</span>
                        </div>
                        <div class="bg-slate-50 p-3 rounded-lg border border-slate-100">
                            <span class="text-[10px] text-slate-400 font-bold uppercase block mb-1 flex items-center gap-1 cursor-help" title="峰度 (Kurtosis)：衡量分佈的肥尾程度。">
                                Kurtosis <i data-lucide="help-circle" class="w-3 h-3 text-slate-300"></i>
                            </span>
                            <span class="text-sm font-bold text-slate-700 font-mono" id="val-kurt">--</span>
                        </div>
                    </div>

                    <div>
                        <h4 class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Basic Statistics</h4>
                        <div class="grid grid-cols-2 gap-y-2 gap-x-4 text-xs">
                            <div class="flex justify-between border-b border-slate-50 pb-1">
                                <span class="text-slate-500">Min</span>
                                <span class="font-mono font-medium text-slate-700" id="stat-min">--</span>
                            </div>
                            <div class="flex justify-between border-b border-slate-50 pb-1">
                                <span class="text-slate-500">Max</span>
                                <span class="font-mono font-medium text-slate-700" id="stat-max">--</span>
                            </div>
                            <div class="flex justify-between border-b border-slate-50 pb-1">
                                <span class="text-slate-500">Mean</span>
                                <span class="font-mono font-medium text-slate-700" id="stat-mean">--</span>
                            </div>
                            <div class="flex justify-between border-b border-slate-50 pb-1">
                                <span class="text-slate-500">Std</span>
                                <span class="font-mono font-medium text-slate-700" id="stat-std">--</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-500">Q1</span>
                                <span class="font-mono font-medium text-slate-700" id="stat-q1">--</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-slate-500">Q3</span>
                                <span class="font-mono font-medium text-slate-700" id="stat-q3">--</span>
                            </div>
                        </div>
                    </div>
                </div>

            </aside>

            <section class="lg:col-span-9 space-y-6">

                <div class="glass-card p-6 flex flex-col">
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-4">
                        <div>
                            <h2 class="text-base font-bold text-slate-800 flex items-center gap-2">
                                <span class="bg-blue-100 text-blue-600 p-1.5 rounded-md"><i data-lucide="line-chart" class="w-4 h-4"></i></span>
                                Time Series
                            </h2>
                            <p class="text-xs text-slate-500 mt-1 pl-9">raw signal.</p>
                        </div>
                        <div class="flex gap-4 text-xs bg-slate-50 px-3 py-1.5 rounded-full border border-slate-100">
                            <div class="flex items-center gap-1.5">
                                <span class="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-sm"></span>
                                <span class="text-slate-600 font-medium">Raw</span>
                            </div>
                        </div>
                    </div>

                    <div class="relative w-full h-[360px]">
                        <canvas id="tsChart"></canvas>
                    </div>

                    <div class="mt-4 pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center gap-6">
                        <div class="flex-1 w-full">
                            <div class="flex justify-between items-center mb-1.5">
                                <label class="text-[10px] font-bold text-slate-400 uppercase flex items-center gap-1"><i data-lucide="search" class="w-3 h-3"></i> Zoom</label>
                                <span class="text-xs text-indigo-600 font-mono font-bold bg-indigo-50 px-1.5 py-0.5 rounded" id="zoomValue">All</span>
                            </div>
                            <input type="range" id="zoomSlider" class="w-full" min="20">
                        </div>
                        <div class="flex-1 w-full">
                            <div class="flex justify-between items-center mb-1.5">
                                <label class="text-[10px] font-bold text-slate-400 uppercase flex items-center gap-1"><i data-lucide="move-horizontal" class="w-3 h-3"></i> Pan</label>
                                <span class="text-xs text-slate-500 font-mono font-bold" id="panValue">0</span>
                            </div>
                            <input type="range" id="panSlider" class="w-full" value="0" disabled>
                        </div>
                    </div>
                </div>

                <!-- ✅ ACF 全寬（在 TS 下方） -->
                <div class="glass-card p-6 min-h-[350px] flex flex-col">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-sm font-bold text-slate-800 flex items-center gap-2">
                            <span class="bg-emerald-100 text-emerald-600 p-1.5 rounded-md"><i data-lucide="repeat" class="w-3.5 h-3.5"></i></span>
                            Autocorrelation (ACF)
                        </h3>
                        <span class="text-[10px] font-mono bg-slate-100 px-2 py-1 rounded text-slate-500" id="acf-lags-badge">Lags: --</span>
                    </div>
                    <div class="relative flex-1 w-full min-h-[240px]">
                        <canvas id="acfChart"></canvas>
                    </div>
                </div>

            </section>
        </div>
    </main>

    <script>
        // ==========================================
        // BACKEND INJECTED DATA
        // ==========================================
        const REPORT_DATA = __REPORT_DATA__;

        let charts = { ts: null, acf: null };
        let currentFeature = null;
        let currentTSData = null;

        document.addEventListener('DOMContentLoaded', () => {
            if (typeof lucide !== 'undefined') lucide.createIcons();

            const tf = document.getElementById('total-features-count');
            if (tf) tf.innerText = REPORT_DATA.meta.total_features;

            const selector = document.getElementById('featureSelector');
            if (!selector) return;

            selector.innerHTML = (REPORT_DATA.features || []).map(f =>
                `<option value="${f.name}">${f.name}</option>`
            ).join('');

            selector.addEventListener('change', (e) => {
                const feature = (REPORT_DATA.features || []).find(f => f.name === e.target.value);
                if (feature) renderFeature(feature);
            });

            if ((REPORT_DATA.features || []).length > 0) {
                renderFeature(REPORT_DATA.features[0]);
            }
        });

        function renderFeature(feature) {
            currentFeature = feature;

            updateMetrics(feature.metrics, feature.charts.stats);

            const insightEl = document.getElementById('llm-insight');
            if (insightEl) {
                insightEl.style.opacity = 0;
                setTimeout(() => {
                    insightEl.textContent = feature.insight || "無分析建議";
                    insightEl.style.transition = 'opacity 0.5s ease';
                    insightEl.style.opacity = 1;
                }, 200);
            }

            currentTSData = feature.charts.ts;

            // points display
            const nPts = currentTSData?.raw?.length ?? (feature.charts.ts?.n ?? '--');
            const pd = document.getElementById('points-disp');
            if (pd) pd.innerText = `N=${nPts}`;

            // ✅ badge: "suggested_window: / Lags: 800"
            const suggestedWindow = feature.charts.ts?.suggested_window ?? '--';
            const lags = feature.charts.acf?.lags?.length ? (feature.charts.acf.lags.length - 1) : 0;
            const badge = document.getElementById('acf-lags-badge');
            if (badge) badge.innerText = `suggested_window: ${suggestedWindow} / Lags: ${lags}`;

            renderTSChartSkeleton(currentTSData);
            initZoomControls();

            renderACFChart(feature.charts.acf);
        }

        function updateMetrics(metrics, stats) {
            const statVal = metrics.stationarity;
            let statColor = statVal > 0.7 ? 'bg-emerald-500' : (statVal < 0.3 ? 'bg-rose-500' : 'bg-amber-500');
            let statTextColor = statVal > 0.7 ? 'text-emerald-600' : (statVal < 0.3 ? 'text-rose-600' : 'text-amber-600');

            const valStatText = document.getElementById('val-stat-text');
            if (valStatText) {
                valStatText.innerHTML = `<span class="${statTextColor}">${(statVal*100).toFixed(0)}</span><span class="text-slate-400 text-xs">/100</span>`;
            }

            const statBar = document.getElementById('bar-stat');
            if (statBar) {
                statBar.className = `absolute top-0 left-0 h-full rounded-full progress-bar-fill ${statColor}`;
                statBar.style.width = '0%';
                setTimeout(() => { statBar.style.setProperty('--target-width', `${statVal * 100}%`); }, 50);
            }

            const acfVal = Math.abs(metrics.acf_lag1);
            const rawAcf = metrics.acf_lag1;

            const valAcfText = document.getElementById('val-acf-text');
            if (valAcfText) valAcfText.innerText = rawAcf.toFixed(2);

            const seg1 = document.getElementById('acf-seg-1');
            const seg2 = document.getElementById('acf-seg-2');
            const seg3 = document.getElementById('acf-seg-3');
            const acfTag = document.getElementById('tag-acf');

            if (seg1 && seg2 && seg3 && acfTag) {
                [seg1, seg2, seg3].forEach(el => el.className = "flex-1 bg-slate-100 transition-colors duration-300 " + (el.id.includes('1') ? 'rounded-l-full' : (el.id.includes('3') ? 'rounded-r-full' : '')));
                if(acfVal < 0.3) {
                    seg1.className = "flex-1 bg-slate-400 rounded-l-full transition-colors duration-300";
                    acfTag.innerText = "Low (Noise)"; acfTag.className = "text-[10px] text-right mt-1 font-medium text-slate-500";
                } else if (acfVal < 0.7) {
                    seg1.className = "flex-1 bg-indigo-300 rounded-l-full transition-colors duration-300";
                    seg2.className = "flex-1 bg-indigo-400 transition-colors duration-300";
                    acfTag.innerText = "Moderate"; acfTag.className = "text-[10px] text-right mt-1 font-medium text-indigo-500";
                } else {
                    seg1.className = "flex-1 bg-indigo-400 rounded-l-full transition-colors duration-300";
                    seg2.className = "flex-1 bg-indigo-500 transition-colors duration-300";
                    seg3.className = "flex-1 bg-indigo-600 rounded-r-full transition-colors duration-300";
                    acfTag.innerText = "High (Memory)"; acfTag.className = "text-[10px] text-right mt-1 font-bold text-indigo-700";
                }
            }

            const vm = document.getElementById('val-missing');
            if (vm) vm.innerText = metrics.missing_rate.toFixed(1) + '%';

            const vk = document.getElementById('val-kurt');
            if (vk) vk.innerText = metrics.kurtosis.toFixed(1);

            const setText = (id, v) => {
                const el = document.getElementById(id);
                if (el) el.innerText = v;
            };

            setText('stat-min',  stats.min.toFixed(2));
            setText('stat-max',  stats.max.toFixed(2));
            setText('stat-mean', stats.mean.toFixed(2));
            setText('stat-std',  stats.std.toFixed(2));
            setText('stat-q1',   stats.q1.toFixed(2));
            setText('stat-q3',   stats.q3.toFixed(2));
        }

        // --------------------------
        // TS chart
        // --------------------------
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 600 },
            font: { family: "'Inter', sans-serif" }
        };

        function renderTSChartSkeleton(ts) {
            const canvas = document.getElementById('tsChart');
            if (!canvas || !ts) return;
            const ctx = canvas.getContext('2d');

            if (charts.ts) charts.ts.destroy();

            const labels = ts.labels || [];
            const raw = ts.raw || [];

            charts.ts = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Raw Value',
                            data: raw,
                            borderColor: '#3b82f6',
                            borderWidth: 2,
                            backgroundColor: 'rgba(59, 130, 246, 0.05)',
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            tension: 0.2,
                            order: 1
                        },
                    ]
                },
                options: {
                    ...commonOptions,
                    plugins: {
                        legend: { display: false },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { grid: { display: false }, ticks: { maxTicksLimit: 8, color: '#94a3b8' } },
                        y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' } }
                    }
                }
            });
        }

        // --------------------------
        // Zoom/Pan controls
        // --------------------------
        function initZoomControls() {
            const zSlider = document.getElementById('zoomSlider');
            const pSlider = document.getElementById('panSlider');
            const zoomVal = document.getElementById('zoomValue');
            const panVal = document.getElementById('panValue');

            if(!zSlider || !pSlider || !zoomVal || !panVal) return;
            if(!currentTSData || !currentTSData.labels) return;

            const totalPoints = currentTSData.labels.length;
            if (totalPoints <= 0) return;

            zSlider.max = totalPoints;
            zSlider.value = totalPoints;
            pSlider.max = 0;
            pSlider.value = 0;
            pSlider.disabled = true;

            zoomVal.innerText = 'All';
            panVal.innerText = '0';

            function updateChart() {
                const visible = parseInt(zSlider.value);
                const maxPan = Math.max(0, totalPoints - visible);

                pSlider.max = maxPan;
                if (parseInt(pSlider.value) > maxPan) pSlider.value = maxPan;
                pSlider.disabled = maxPan <= 0;

                const start = parseInt(pSlider.value);
                const end = start + visible - 1;

                const pct = Math.round((visible/totalPoints)*100);
                zoomVal.innerText = visible === totalPoints ? 'All' : `${pct}%`;
                panVal.innerText = String(start);

                if (charts.ts) {
                    charts.ts.options.scales.x.min = currentTSData.labels[start];
                    charts.ts.options.scales.x.max = currentTSData.labels[Math.min(end, totalPoints-1)];
                    charts.ts.update('none');
                }
            }

            zSlider.oninput = updateChart;
            pSlider.oninput = updateChart;
        }

        // --------------------------
        // ACF
        // --------------------------
        function renderACFChart(data) {
            const canvas = document.getElementById('acfChart');
            if (!canvas || !data || !data.lags || !data.values) return;

            const ctx = canvas.getContext('2d');
            if (charts.acf) charts.acf.destroy();

            charts.acf = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.lags,
                    datasets: [{
                        label: 'Autocorrelation',
                        data: data.values,
                        backgroundColor: data.values.map(v => v > 0.5 ? '#10b981' : (v < 0 ? '#f43f5e' : '#cbd5e1')),
                        borderRadius: 3
                    }]
                },
                options: {
                    ...commonOptions,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#94a3b8', maxTicksLimit: 10 } },
                        y: { min: -1, max: 1, grid: { color: '#f1f5f9' }, ticks: { stepSize: 0.5, color: '#94a3b8' } }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""


def _error_response(tool_name: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "message": f"{tool_name} failed: {message}"}


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s


def _resolve_any_path(p: str) -> str:
    """
    將輸入（可能是檔名/相對路徑/絕對路徑/file:// URL）轉成絕對路徑。
    - 允許 file:///app/state/xxx.json
    - 允許 "feature_xxx.json"（只給檔名）
    - 允許 "state/feature_xxx.json"（相對）
    """
    if not p:
        return ""

    s = _strip_quotes(str(p))

    # file:// URL
    if s.startswith("file://"):
        u = urlparse(s)
        path = unquote(u.path)

        # Windows 可能會是 /C:/Users/...
        if os.name == "nt" and path.startswith("/") and len(path) > 3 and path[2] == ":":
            path = path[1:]
        s = path

    # 先嘗試直接當作路徑（絕對或相對）
    if os.path.isabs(s):
        return os.path.abspath(s)

    # 相對路徑：用 cwd 補齊
    cand = os.path.abspath(os.path.join(os.getcwd(), s))
    if os.path.exists(cand):
        return cand

    # 如果只是檔名 or 找不到：回傳「預設 state dir」的候選
    # (不保證存在，load_state 會再判斷)
    return os.path.abspath(os.path.join(STATE_DIR, os.path.basename(s)))


def _load_state(state_uri_or_url: str) -> Dict[str, Any]:
    if not state_uri_or_url:
        raise FileNotFoundError("state_uri/state_url 為空。")

    path = _resolve_any_path(state_uri_or_url)

    # 若解析後不存在，再 fallback 一次：STATE_DIR/basename
    if not os.path.exists(path):
        fallback = os.path.abspath(os.path.join(STATE_DIR, os.path.basename(state_uri_or_url)))
        if os.path.exists(fallback):
            path = fallback

    if not os.path.exists(path):
        raise FileNotFoundError(f"state 檔案不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("state JSON 不是 dict（object）")

    # 把實際讀到的 path 放回去，方便 debug
    obj.setdefault("_resolved_state_path", path)
    return obj


def _save_report_html(html: str) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    filename = f"feature_report_{uuid.uuid4().hex}.html"
    full_path = os.path.abspath(os.path.join(REPORT_DIR, filename))
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(html)
    return full_path


def _render_html(report_title: str, report_data: Dict[str, Any]) -> str:
    report_json = json.dumps(report_data, ensure_ascii=False)
    return (
        HTML_TEMPLATE
        .replace("__REPORT_TITLE__", report_title)
        .replace("__REPORT_DATA__", report_json)
    )


def _build_report_data_from_state_and_csv(state: Dict[str, Any], data_uri: str) -> Dict[str, Any]:
    """
    state JSON 不存 raw，所以這裡在產生 HTML 時才從 CSV 把 raw/labels 注入進 REPORT_DATA。
    注入後不回寫 state，避免 state 膨脹。
    """
    meta = state.get("meta", {}) or {}
    features = state.get("features", []) or []

    df = pd.read_csv(data_uri)

    out_features: List[Dict[str, Any]] = []
    for f in features:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if not name or name not in df.columns:
            continue

        s0 = df[name]
        x = s0.dropna().astype(float).to_numpy()
        labels = [f"{i}" for i in range(x.size)]

        f2 = dict(f)
        charts = dict(f2.get("charts", {}) or {})
        ts = dict(charts.get("ts") or {})

        ts["labels"] = labels
        ts["raw"] = x.astype(float).tolist()

        charts["ts"] = ts
        f2["charts"] = charts
        out_features.append(f2)

    meta = dict(meta)
    meta["total_features"] = int(meta.get("total_features", len(out_features)))

    return {"meta": meta, "features": out_features}


class GenerateFeatureHtmlReport(Component):
    display_name = "Generate Feature HTML Report"
    description = "讀取 state_uri/state_url + CSV，產生互動式HTML報告"
    icon = "file-text"

    tool_mode = True
    is_tool = True

    inputs = [
        StrInput(
            name="data_uri",
            display_name="CSV Path (data_uri)",
            info="CSV 檔案路徑（建議用容器內路徑或 file://）",
            required=True,
            tool_mode=True,
        ),
        # ✅ Agent 可能傳 state_uri
        StrInput(
            name="state_uri",
            display_name="State URI (state_uri)",
            info="AnalyzeFeatureReport 產生的 state json 檔名/路徑/file://",
            required=True,
            tool_mode=True,  # ✅ 讓 agent tool 可以餵
        ),
    ]

    outputs = [
        Output(name="response", display_name="Response", method="run_result"),
    ]

    def run_result(self) -> Data:
        data_uri = (getattr(self, "data_uri", "") or "").strip()
        state_uri = (getattr(self, "state_uri", "") or "").strip()
        state_url = (getattr(self, "state_url", "") or "").strip()
        return Data(data=self._generate(data_uri=data_uri, state_uri=state_uri, state_url=state_url))

    def build_tools(self):
        @tool
        def generate_feature_html_report(data_uri: str, state_uri: str) -> dict:
            """Generate interactive HTML report from (data_uri, state_uri)."""
            return self._generate(data_uri=data_uri, state_uri=state_uri, state_url="")

        @tool
        def generate_feature_html_report_by_url(data_uri: str, state_url: str) -> dict:
            """Generate interactive HTML report from (data_uri, state_url). state_url can be file://... or path."""
            return self._generate(data_uri=data_uri, state_uri="", state_url=state_url)

        return [generate_feature_html_report, generate_feature_html_report_by_url]

    def _generate(self, data_uri: str, state_uri: str = "", state_url: str = "") -> Dict[str, Any]:
        tool_name = "generate_feature_html_report"
        try:
            data_path = _resolve_any_path(data_uri)
            if not data_path:
                return _error_response(tool_name, "需要提供 'data_uri'。")
            if not os.path.exists(data_path):
                return _error_response(tool_name, f"找不到 CSV：{data_path}")

            # ✅ state 來源：state_uri 優先，其次 state_url
            su = (state_uri or "").strip()
            sz = (state_url or "").strip()
            state_ref = su if su else sz
            if not state_ref:
                return _error_response(tool_name, "需要提供 'state_uri' 或 'state_url'。")

            state = _load_state(state_ref)

            # 可選：檢查 state 的 data_uri 是否一致（若 state 裡有記）
            state_data_uri = state.get("data_uri")
            if state_data_uri:
                s1 = _resolve_any_path(str(state_data_uri))
                if os.path.exists(s1) and os.path.abspath(s1) != os.path.abspath(data_path):
                    return _error_response(
                        tool_name,
                        f"data_uri 與 state 記錄不一致。state: {state_data_uri}, input: {data_path}",
                    )

            report_data = _build_report_data_from_state_and_csv(state=state, data_uri=data_path)
            report_title = f"時序異常偵測 - 特徵分析報告 - {os.path.basename(data_path)}"

            html = _render_html(report_title=report_title, report_data=report_data)
            report_path = _save_report_html(html)

            return {
                "status": "success",
                "message": "HTML 報告已生成（TS + ACF）。",
                "data_uri": data_path,
                "state_ref": state_ref,
                "resolved_state_path": state.get("_resolved_state_path"),
                "report_path": report_path,
                "n_features": int(report_data["meta"].get("total_features", len(report_data["features"]))),
                "meta": report_data["meta"],
                "report_dir": os.path.abspath(REPORT_DIR),
            }

        except Exception as e:
            return _error_response(tool_name, str(e))


# =============================================================================
# 
# =============================================================================
import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from langflow.custom import Component
from langflow.io import StrInput, Output
from langflow.schema import Data

from langchain_core.tools import tool


BASE_DIR = os.getcwd()
STATE_DIR = os.path.join(BASE_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)


def _error_response(tool_name: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "message": f"{tool_name} failed: {message}"}


def _save_state(state: Dict[str, Any], filename: Optional[str] = None) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    if not filename:
        short_id = uuid.uuid4().hex[:6]
        filename = f"feature_{short_id}.json"
    path = os.path.join(STATE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return filename


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _compute_basic_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "q1": 0.0, "q3": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)) if values.size > 1 else 0.0,
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


def _autocorr_at_lag(x: np.ndarray, lag: int) -> float:
    if lag <= 0:
        return 1.0
    if x.size <= lag + 2:
        return 0.0
    a = x[lag:]
    b = x[:-lag]
    if a.size < 3 or b.size < 3:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _acf_values(x: np.ndarray, lags: int) -> List[float]:
    return [_autocorr_at_lag(x, k) for k in range(0, lags + 1)]


def _stationarity_score_heuristic(x: np.ndarray) -> float:
    n = x.size
    if n < 20:
        return 0.5
    t = np.arange(n, dtype=float)
    y = x.astype(float)
    try:
        slope, _ = np.polyfit(t, y, 1)
    except Exception:
        slope = 0.0
    std = float(np.std(y)) + 1e-9
    trend_strength = abs(slope) * n / (std * 10.0)

    mid = n // 2
    m1, m2 = float(np.mean(y[:mid])), float(np.mean(y[mid:]))
    s1 = float(np.std(y[:mid])) + 1e-9
    s2 = float(np.std(y[mid:])) + 1e-9

    mean_shift = abs(m1 - m2) / (std * 2.0)
    var_shift = abs(s1 - s2) / (std * 2.0)

    score = 1.0 - (0.55 * trend_strength + 0.30 * mean_shift + 0.15 * var_shift)
    return float(np.clip(score, 0.0, 1.0))


def choose_acf_lags(
    x: np.ndarray,
    hard_cap: int = 200,
    min_lags: int = 10,
    max_lags: int = 200,
    threshold: float = 0.4,
    consecutive: int = 5,
) -> int:
    n = int(len(x))
    if n < 20:
        return int(min_lags)
    n_guard = n - 5
    cap = min(hard_cap, max_lags, n_guard)
    cap = max(min_lags, int(cap))

    acf = _acf_values(x, cap)
    run = 0
    for lag in range(1, cap + 1):
        if abs(acf[lag]) < threshold:
            run += 1
        else:
            run = 0
        if run >= consecutive:
            return int(max(min_lags, lag - consecutive + 1))
    return int(cap)


def _suggest_window_from_decay(decay_lag: int, n: int, win_min: int = 10, win_cap: int = 2000) -> int:
    if n <= 0:
        return win_min
    w = int(max(2, decay_lag * 2))
    return int(np.clip(w, win_min, min(win_cap, int(n))))


class AnalyzeFeatureReport(Component):
    display_name = "Analyze Feature Report"
    description = "讀取 CSV 並分析所有數值欄位，結果寫入一個 JSON 檔"
    icon = "activity"

    # （可留可不留）讓系統知道它支援 tool mode
    tool_mode = True
    is_tool = True

    inputs = [
        # ✅ 關鍵：至少一個 input 要 tool_mode=True，前端才會給你 header 的 Tool Mode toggle :contentReference[oaicite:1]{index=1}
        StrInput(
            name="data_uri",
            display_name="CSV Path (data_uri)",
            info="CSV 檔案路徑（Docker/Linux 請用容器內路徑）",
            required=True,
            tool_mode=True,   # ✅ 就是這行
        ),
    ]

    # ✅ 關鍵：只保留一個正常輸出，避免 UI 走「輸出下拉選單」
    outputs = [
        Output(name="response", display_name="Response", method="run_result"),
    ]

    def run_result(self) -> Data:
        data_uri = (getattr(self, "data_uri", "") or "").strip()
        return Data(data=self._analyze(data_uri))

    # ✅ 明確提供工具集合（Tool Mode ON 時 Langflow 會把這顆當 Toolset 丟給 Agent）
    def build_tools(self):
        @tool(
            name="analyze_feature_report",
            description="Analyze numeric columns in a CSV and write a compact state JSON. Input: data_uri (CSV path). Output: status/state_uri/meta.",
        )
        def _tool(data_uri: str) -> dict:
            return self._analyze(data_uri)

        return [_tool]

    def _analyze(self, data_uri: str) -> Dict[str, Any]:
        tool_name = "analyze_feature_report"
        data_uri = (data_uri or "").strip()

        try:
            if not data_uri:
                return _error_response(tool_name, "需要提供 'data_uri'。")
            if not os.path.exists(data_uri):
                return _error_response(
                    tool_name,
                    f"找不到檔案：{data_uri}。若你填 C:\\Users\\... 且 Langflow 在 Docker/Linux，請改填容器內路徑。",
                )

            df = pd.read_csv(data_uri)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return _error_response(tool_name, "此 CSV 沒有數值欄位。")

            acf_max_plot_lags = 800
            features: List[Dict[str, Any]] = []

            for col in numeric_cols:
                s0 = df[col]
                missing_rate = float(s0.isna().mean() * 100.0) if len(s0) else 0.0
                x = s0.dropna().astype(float).to_numpy()
                if x.size == 0:
                    continue

                n = int(x.size)
                decay_lag = choose_acf_lags(x, min_lags=10, max_lags=200, threshold=0.3, consecutive=5)
                suggested_window = _suggest_window_from_decay(decay_lag=decay_lag, n=n)

                acf_plot_lags = int(min(max(10, acf_max_plot_lags), max(10, n - 5))) if n > 15 else 10
                acf_plot_lags = max(10, acf_plot_lags)

                stats = _compute_basic_stats(x)
                kurt = float(pd.Series(x).kurtosis()) if n >= 4 else 0.0
                acf_vals = _acf_values(x, acf_plot_lags)
                acf_lag1 = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
                stationarity = _stationarity_score_heuristic(x)

                features.append(
                    {
                        "name": col,
                        "metrics": {
                            "stationarity": _safe_float(stationarity, 0.5),
                            "acf_lag1": _safe_float(acf_lag1, 0.0),
                            "missing_rate": _safe_float(missing_rate, 0.0),
                            "kurtosis": _safe_float(kurt, 0.0),
                        },
                        "charts": {
                            "ts": {
                                "n": int(n),
                                "decay_lag": int(decay_lag),
                                "suggested_window": int(suggested_window),
                                "acf_plot_lags": int(acf_plot_lags),
                            },
                            "acf": {
                                "lags": list(range(0, acf_plot_lags + 1)),
                                "values": [float(v) for v in acf_vals],
                            },
                            "stats": {
                                "min": float(stats["min"]),
                                "max": float(stats["max"]),
                                "mean": float(stats["mean"]),
                                "std": float(stats["std"]),
                                "q1": float(stats["q1"]),
                                "q3": float(stats["q3"]),
                            },
                        },
                        "insight": None,
                    }
                )

            meta = {
                "total_features": len(features),
                "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "params": {
                    "acf_threshold": 0.3,
                    "acf_consecutive": 5,
                    "acf_min_lags": 10,
                    "acf_max_plot_lags": 800,
                },
            }

            state = {
                "data_uri": data_uri,
                "created_at": datetime.now().isoformat(),
                "meta": meta,
                "features": features,
            }

            state_uri = _save_state(state)

            return {
                "status": "success",
                "message": "特徵分析完成。",
                "state_uri": state_uri,
                "n_features": len(features),
                "params": meta["params"],
                "state_dir": STATE_DIR,
            }

        except Exception as e:
            return _error_response(tool_name, str(e))


