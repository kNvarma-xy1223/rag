from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from api.routes import router
import uvicorn

app = FastAPI(title="Multilingual RAG System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API ROUTES
# =========================
app.include_router(router)


# =========================
# FRONTEND UI ROUTE
# =========================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    return HTMLResponse(content=HTML_UI)


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


# =========================
# HTML UI
# =========================

HTML_UI = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG System - Multilingual</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&family=Barlow+Condensed:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#f0f2f5;--surface:#fff;--surface-2:#f8f9fb;
    --border:#dde1e8;--border-light:#edf0f4;
    --text:#0e1726;--text-2:#4b5668;--text-3:#8892a0;
    --openai:#10a37f;--openai-bg:#e8f7f3;--openai-border:#a8ddd1;
    --cohere:#d97706;--cohere-bg:#fef3e2;--cohere-border:#f9c97a;
    --accent:#1d4ed8;--accent-hover:#1e40af;--accent-bg:#eff6ff;
    --danger:#dc2626;--danger-bg:#fef2f2;
    --success:#16a34a;--success-bg:#f0fdf4;
    --ragas:#7c3aed;--ragas-bg:#f5f3ff;--ragas-border:#c4b5fd;
    --mono:'JetBrains Mono',monospace;
    --sans:'Barlow',sans-serif;
    --condensed:'Barlow Condensed',sans-serif;
    --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
    --shadow-md:0 4px 12px rgba(0,0,0,.08),0 2px 4px rgba(0,0,0,.04);
    --radius:8px;--radius-sm:5px;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:var(--sans);background:var(--bg);color:var(--text);font-size:14px;line-height:1.6;min-height:100vh}

  /* ── Header ── */
  header{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;display:flex;align-items:center;gap:20px;height:56px;position:sticky;top:0;z-index:100;box-shadow:var(--shadow)}
  .brand-icon{width:32px;height:32px;background:var(--accent);border-radius:7px;display:flex;align-items:center;justify-content:center;font-family:var(--condensed);font-weight:700;font-size:15px;color:#fff;letter-spacing:.5px}
  .brand-title{font-family:var(--condensed);font-weight:600;font-size:18px;letter-spacing:.3px}
  .brand-sub{font-size:11px;color:var(--text-3);font-family:var(--mono);margin-top:-2px}
  .header-spacer{flex:1}
  .status-dot{width:7px;height:7px;border-radius:50%;background:#9ca3af;display:inline-block;margin-right:6px;transition:background .3s}
  .status-dot.ok{background:var(--success)}.status-dot.error{background:var(--danger)}
  .status-label{font-family:var(--mono);font-size:11px;color:var(--text-3)}

  /* ── Tabs ── */
  .tab-nav{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;display:flex}
  .tab-btn{font-family:var(--condensed);font-weight:500;font-size:13px;letter-spacing:.5px;text-transform:uppercase;padding:14px 18px;border:none;background:none;color:var(--text-3);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s,border-color .15s;white-space:nowrap}
  .tab-btn:hover{color:var(--text)}.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

  /* ── Layout ── */
  main{max-width:1280px;margin:0 auto;padding:24px}
  .tab-panel{display:none}.tab-panel.active{display:block}
  .two-col{display:grid;grid-template-columns:380px 1fr;gap:20px;align-items:start}

  /* ── Card ── */
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
  .card-header{padding:14px 18px;border-bottom:1px solid var(--border-light);display:flex;align-items:center;justify-content:space-between;gap:10px}
  .card-title{font-family:var(--condensed);font-weight:600;font-size:13px;letter-spacing:.8px;text-transform:uppercase;color:var(--text-2)}
  .card-body{padding:18px}

  /* ── Form ── */
  .field{margin-bottom:16px}.field:last-child{margin-bottom:0}
  label{display:block;font-family:var(--condensed);font-weight:600;font-size:11px;letter-spacing:.8px;text-transform:uppercase;color:var(--text-2);margin-bottom:6px}
  input[type="text"],input[type="number"],select,textarea{width:100%;padding:9px 12px;border:1px solid var(--border);border-radius:var(--radius-sm);font-family:var(--sans);font-size:13.5px;color:var(--text);background:var(--surface);transition:border-color .15s,box-shadow .15s;outline:none;-webkit-appearance:none}
  input:focus,select:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(29,78,216,.08)}
  textarea{resize:vertical;min-height:72px}
  select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238892a0' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;padding-right:32px;cursor:pointer}
  .inline-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .inline-row.three{grid-template-columns:1fr 1fr 1fr}

  /* ── Buttons ── */
  .btn{display:inline-flex;align-items:center;gap:6px;padding:9px 16px;border-radius:var(--radius-sm);border:1px solid transparent;font-family:var(--condensed);font-weight:600;font-size:13px;letter-spacing:.5px;text-transform:uppercase;cursor:pointer;transition:all .15s;white-space:nowrap}
  .btn-primary{background:var(--accent);color:#fff;border-color:var(--accent)}
  .btn-primary:hover:not(:disabled){background:var(--accent-hover);border-color:var(--accent-hover)}
  .btn-secondary{background:var(--surface);color:var(--text-2);border-color:var(--border)}
  .btn-secondary:hover:not(:disabled){background:var(--surface-2)}
  .btn-ragas{background:var(--ragas);color:#fff;border-color:var(--ragas)}
  .btn-ragas:hover:not(:disabled){background:#6d28d9;border-color:#6d28d9}
  .btn-danger{background:var(--danger-bg);color:var(--danger);border-color:#fca5a5}
  .btn-danger:hover:not(:disabled){background:#fee2e2}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn.loading::after{content:'';width:12px;height:12px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:spin .6s linear infinite;display:inline-block}
  .btn-full{width:100%;justify-content:center}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* ── Drop zone ── */
  .drop-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:32px 20px;text-align:center;cursor:pointer;transition:all .2s;background:var(--surface-2);position:relative}
  .drop-zone:hover,.drop-zone.drag-over{border-color:var(--accent);background:var(--accent-bg)}
  .drop-zone input[type="file"]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .drop-icon{width:40px;height:40px;border:2px solid var(--border);border-radius:8px;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;background:var(--surface);color:var(--text-3);font-family:var(--mono);font-size:11px;font-weight:600}
  .drop-text{font-size:13px;color:var(--text-2);margin-bottom:4px}
  .drop-hint{font-family:var(--mono);font-size:11px;color:var(--text-3)}
  .file-list-item{display:flex;align-items:center;gap:8px;padding:7px 12px;background:var(--accent-bg);border:1px solid var(--accent);border-radius:var(--radius-sm);margin-bottom:6px;font-size:12.5px;color:var(--accent);font-family:var(--mono)}
  .file-list-item .file-size{font-size:11px;color:var(--text-3);margin-left:4px}
  .file-list-item .file-remove{margin-left:auto;cursor:pointer;opacity:.6;font-size:14px;line-height:1;background:none;border:none;color:var(--accent);flex-shrink:0}

  /* ── Progress ── */
  .progress-wrap{height:8px;background:var(--border);border-radius:4px;overflow:hidden;margin-bottom:8px}
  .progress-bar{height:100%;background:linear-gradient(90deg,var(--accent),#3b82f6);border-radius:4px;transition:width .4s ease;width:0%}
  .progress-msg{font-family:var(--mono);font-size:11.5px;color:var(--text-2);margin-bottom:4px}

  /* ── Streaming cursor ── */
  .stream-cursor::after{content:'|';animation:blink .7s step-end infinite;color:var(--accent);font-size:.9em}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

  /* ── Model badges ── */
  .model-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border-radius:99px;font-family:var(--mono);font-size:11px;font-weight:500}
  .model-badge.openai{background:var(--openai-bg);color:var(--openai);border:1px solid var(--openai-border)}
  .model-badge.cohere{background:var(--cohere-bg);color:var(--cohere);border:1px solid var(--cohere-border)}
  .model-badge::before{content:'';width:6px;height:6px;border-radius:50%;background:currentColor}

  /* ── Score bar ── */
  .score-bar-wrap{display:flex;align-items:center;gap:8px}
  .score-bar{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
  .score-bar-fill{height:100%;border-radius:2px;background:var(--accent);transition:width .4s ease}
  .score-val{font-family:var(--mono);font-size:12px;color:var(--text-2);min-width:44px;text-align:right}

  /* ── Chunks ── */
  .chunk-item{border:1px solid var(--border);border-radius:var(--radius-sm);margin-bottom:10px;overflow:hidden;background:var(--surface)}
  .chunk-item:last-child{margin-bottom:0}
  .chunk-header{padding:8px 12px;background:var(--surface-2);border-bottom:1px solid var(--border-light);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .chunk-rank{font-family:var(--mono);font-size:11px;font-weight:500;color:var(--text-3);background:var(--border-light);padding:1px 6px;border-radius:3px}
  .chunk-source{font-family:var(--mono);font-size:11px;color:var(--text-2)}
  .chunk-meta{font-family:var(--mono);font-size:10px;color:var(--text-3);background:var(--surface-2);padding:2px 6px;border-radius:3px;border:1px solid var(--border-light)}
  .chunk-body{padding:12px;font-size:13px;line-height:1.6;color:var(--text-2);max-height:120px;overflow:hidden}
  .chunk-body.expanded{max-height:none}
  .chunk-expand{font-size:11px;color:var(--accent);cursor:pointer;padding:4px 12px;display:block;border-top:1px solid var(--border-light);background:var(--surface-2);font-family:var(--condensed);font-weight:600;letter-spacing:.3px;text-transform:uppercase;text-align:center}

  /* ── Answer box ── */
  .answer-box{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:var(--radius);padding:18px;font-size:14px;line-height:1.85;color:var(--text);word-break:break-word}

  /* ── Citations ── */
  .cite-sup{display:inline-flex;align-items:center;justify-content:center;min-width:17px;height:17px;padding:0 4px;border-radius:4px;background:var(--accent-bg);border:1px solid #bfdbfe;color:var(--accent);font-family:var(--mono);font-size:10px;font-weight:700;cursor:pointer;vertical-align:super;transition:background .15s,color .15s;margin:0 1px;line-height:1}
  .cite-sup:hover,.cite-sup.active{background:var(--accent);color:#fff;border-color:var(--accent)}
  .citation-item{display:flex;gap:12px;padding:10px 14px;border:1px solid var(--border);border-radius:var(--radius-sm);margin-bottom:8px;background:var(--surface);align-items:flex-start;transition:border-color .2s,background .2s}
  .citation-item.highlighted{border-color:var(--accent);background:var(--accent-bg)}
  .citation-item:last-child{margin-bottom:0}
  .citation-num{font-family:var(--mono);font-size:12px;font-weight:500;width:22px;height:22px;border-radius:50%;background:var(--accent-bg);color:var(--accent);display:flex;align-items:center;justify-content:center;flex-shrink:0;border:1px solid #bfdbfe}
  .citation-content{flex:1;min-width:0}
  .citation-header{display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap}
  .citation-source{font-family:var(--mono);font-size:11.5px;color:var(--text);font-weight:500}
  .citation-loc{font-family:var(--mono);font-size:10.5px;color:var(--text-3)}
  .citation-preview{font-size:12px;color:var(--text-2);line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

  /* ── Compare grid ── */
  .compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}

  /* ── Metric cards ── */
  .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}
  .metric-card{padding:14px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);text-align:center}
  .metric-value{font-family:var(--mono);font-size:24px;font-weight:500;color:var(--accent);line-height:1.2}
  .metric-label{font-family:var(--condensed);font-size:11px;letter-spacing:.6px;text-transform:uppercase;color:var(--text-3);margin-top:4px}
  .stat-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border-light);font-size:13px}
  .stat-row:last-child{border-bottom:none}
  .stat-key{color:var(--text-2)}.stat-val{font-family:var(--mono);font-size:12px;color:var(--text);font-weight:500}

  /* ── RAGAS metric cards ── */
  .ragas-metric-card{padding:16px;border:1px solid var(--ragas-border);border-radius:var(--radius-sm);background:var(--ragas-bg);text-align:center}
  .ragas-metric-value{font-family:var(--mono);font-size:26px;font-weight:500;color:var(--ragas);line-height:1.2}
  .ragas-metric-label{font-family:var(--condensed);font-size:11px;letter-spacing:.6px;text-transform:uppercase;color:var(--ragas);opacity:.7;margin-top:4px}
  .ragas-score-bar{height:6px;background:var(--ragas-border);border-radius:3px;overflow:hidden;margin-top:8px}
  .ragas-score-fill{height:100%;background:var(--ragas);border-radius:3px;transition:width .6s ease}
  .ragas-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:99px;font-family:var(--mono);font-size:11px;font-weight:600;background:var(--ragas-bg);color:var(--ragas);border:1px solid var(--ragas-border)}

  /* ── Eval sub-tabs ── */
  .eval-subtabs{display:flex;gap:2px;margin-bottom:20px;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius);padding:4px}
  .eval-subtab{flex:1;padding:8px 14px;border:none;background:none;font-family:var(--condensed);font-weight:600;font-size:12px;letter-spacing:.5px;text-transform:uppercase;color:var(--text-3);cursor:pointer;border-radius:6px;transition:all .15s;text-align:center}
  .eval-subtab:hover{color:var(--text)}
  .eval-subtab.active{background:var(--surface);color:var(--ragas);box-shadow:var(--shadow);border:1px solid var(--ragas-border)}
  .eval-subpanel{display:none}.eval-subpanel.active{display:block}

  /* ── Data table ── */
  .data-table{width:100%;border-collapse:collapse;font-size:12.5px}
  .data-table th{font-family:var(--condensed);font-size:11px;letter-spacing:.6px;text-transform:uppercase;color:var(--text-3);font-weight:600;padding:8px 12px;text-align:left;border-bottom:1px solid var(--border);background:var(--surface-2)}
  .data-table td{padding:8px 12px;border-bottom:1px solid var(--border-light);vertical-align:top}
  .data-table tr:last-child td{border-bottom:none}
  .data-table tr:hover td{background:var(--surface-2)}
  .mono-val{font-family:var(--mono);font-size:12px}

  /* ── Toast ── */
  #toast{position:fixed;bottom:24px;right:24px;padding:12px 18px;border-radius:var(--radius);font-size:13px;font-weight:500;box-shadow:var(--shadow-md);opacity:0;transform:translateY(8px);transition:all .25s;pointer-events:none;z-index:1000;max-width:340px}
  #toast.show{opacity:1;transform:translateY(0)}
  #toast.success{background:#f0fdf4;color:#15803d;border:1px solid #86efac}
  #toast.error{background:#fef2f2;color:#b91c1c;border:1px solid #fca5a5}
  #toast.info{background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd}

  /* ── Empty / error states ── */
  .empty-state{text-align:center;padding:36px 20px;color:var(--text-3);font-size:13px}
  .empty-state .empty-label{font-family:var(--condensed);font-size:13px;letter-spacing:.3px;margin-bottom:4px;color:var(--text-3)}
  .error-box{padding:12px 16px;background:var(--danger-bg);border:1px solid #fca5a5;border-radius:var(--radius-sm);color:var(--danger);font-size:13px}

  /* ── Latency pills ── */
  .latency-pill{font-family:var(--mono);font-size:10.5px;padding:2px 7px;border-radius:99px;background:var(--surface-2);border:1px solid var(--border);color:var(--text-2)}
  .latency-pill.fast{background:var(--success-bg);border-color:#86efac;color:var(--success)}
  .latency-pill.slow{background:#fff7ed;border-color:#fed7aa;color:#c2410c}

  /* ── Collection boxes ── */
  .coll-box{border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;background:var(--surface)}
  .coll-box-header{padding:14px 18px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border-light)}
  .coll-status{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px}
  .coll-dot{width:7px;height:7px;border-radius:50%;background:#9ca3af}
  .coll-dot.green{background:var(--success)}
  .overlap-bar{height:8px;background:var(--border-light);border-radius:4px;overflow:hidden;margin:4px 0}
  .overlap-fill{height:100%;background:linear-gradient(90deg,var(--openai),var(--cohere));border-radius:4px;transition:width .5s ease}

  /* ── Chat ── */
  .chat-layout{display:flex;flex-direction:column;height:calc(100vh - 158px);min-height:520px}
  .chat-toolbar{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius) var(--radius) 0 0;padding:10px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border-light)}
  .chat-toolbar-title{font-family:var(--condensed);font-weight:600;font-size:14px;letter-spacing:.5px;color:var(--text)}
  .chat-toolbar select,.chat-toolbar input[type="number"]{width:auto;padding:5px 10px;font-size:12.5px}
  .chat-toolbar input[type="number"]{width:60px}
  .chat-spacer{flex:1}
  .chat-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:16px;background:var(--surface);border-left:1px solid var(--border);border-right:1px solid var(--border)}
  .chat-msg{display:flex;gap:10px;max-width:88%}
  .chat-msg.user{align-self:flex-end;flex-direction:row-reverse}
  .chat-msg.assistant{align-self:flex-start}
  .chat-avatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;font-weight:700;font-family:var(--condensed);letter-spacing:.3px}
  .chat-msg.user .chat-avatar{background:var(--accent);color:#fff}
  .chat-msg.assistant .chat-avatar{background:var(--surface-2);border:1px solid var(--border);color:var(--text-2);font-size:11px}
  .chat-msg-wrap{display:flex;flex-direction:column;gap:6px;max-width:100%}
  .chat-bubble{padding:11px 15px;border-radius:12px;font-size:13.5px;line-height:1.7}
  .chat-msg.user .chat-bubble{background:var(--accent);color:#fff;border-radius:12px 12px 4px 12px}
  .chat-msg.assistant .chat-bubble{background:var(--surface-2);border:1px solid var(--border);color:var(--text);border-radius:12px 12px 12px 4px}
  .chat-sources{display:flex;flex-wrap:wrap;gap:5px}
  .chat-source-chip{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;background:var(--surface);border:1px solid var(--border);border-radius:99px;font-family:var(--mono);font-size:10.5px;color:var(--text-2);cursor:default;transition:all .15s}
  .chat-source-chip:hover{border-color:var(--accent);color:var(--accent)}
  .chat-meta{font-family:var(--mono);font-size:10px;color:var(--text-3);display:flex;gap:8px;align-items:center;padding-left:2px}
  .chat-input-bar{background:var(--surface);border:1px solid var(--border);border-radius:0 0 var(--radius) var(--radius);padding:12px 14px;display:flex;gap:8px;align-items:flex-end}
  .chat-input-bar textarea{flex:1;min-height:40px;max-height:120px;resize:none;font-size:13.5px;padding:9px 12px}
  .typing-dot{display:inline-flex;gap:4px;padding:8px 4px;align-items:center}
  .typing-dot span{width:7px;height:7px;border-radius:50%;background:var(--text-3);animation:bounce 1.2s ease-in-out infinite}
  .typing-dot span:nth-child(2){animation-delay:.2s}
  .typing-dot span:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}

  /* ── Accuracy badge ── */
  .accuracy-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:99px;font-family:var(--mono);font-size:11px;font-weight:600}
  .accuracy-badge.high{background:#f0fdf4;color:#15803d;border:1px solid #86efac}
  .accuracy-badge.med{background:#fff7ed;color:#c2410c;border:1px solid #fed7aa}
  .accuracy-badge.low{background:var(--danger-bg);color:var(--danger);border:1px solid #fca5a5}

  /* ── Info banner ── */
  .info-banner{padding:10px 14px;background:var(--ragas-bg);border:1px solid var(--ragas-border);border-radius:var(--radius-sm);font-size:12.5px;color:var(--ragas);margin-bottom:16px;line-height:1.6}

  /* ── Responsive ── */
  @media(max-width:900px){.two-col{grid-template-columns:1fr}.compare-grid{grid-template-columns:1fr}.inline-row.three{grid-template-columns:1fr 1fr}}
  @media(max-width:600px){main{padding:14px}header{padding:0 16px}.tab-nav{padding:0 16px;overflow-x:auto}.inline-row,.inline-row.three{grid-template-columns:1fr}}
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:10px">
    <div class="brand-icon">RAG</div>
    <div>
      <div class="brand-title">Multilingual RAG System</div>
      <div class="brand-sub">text-embedding-3-large &middot; embed-v-4-0 &middot; Pinecone &middot; gpt-5.4-pro</div>
    </div>
  </div>
  <div class="header-spacer"></div>
  <span class="status-dot" id="statusDot"></span>
  <span class="status-label" id="statusLabel">Checking...</span>
</header>

<nav class="tab-nav">
  <button class="tab-btn active" data-tab="ingest">Ingest</button>
  <button class="tab-btn" data-tab="query">Query</button>
  <button class="tab-btn" data-tab="compare">Compare</button>
  <button class="tab-btn" data-tab="chat">Chat</button>
  <button class="tab-btn" data-tab="evaluate">Evaluate (RAGAS)</button>
  <button class="tab-btn" data-tab="collections">Collections</button>
</nav>

<main>

<!-- ══════════════════════════════════════════
     INGEST
     ══════════════════════════════════════════ -->
<div class="tab-panel active" id="tab-ingest">
  <div class="two-col">
    <div>
      <div class="card" style="margin-bottom:16px">
        <div class="card-header"><span class="card-title">Upload Document</span></div>
        <div class="card-body">
          <div class="drop-zone" id="dropZone">
            <input type="file" id="fileInput" accept=".pdf,.csv" multiple>
            <div class="drop-icon">PDF</div>
            <div class="drop-text">Drop files here or click to browse</div>
            <div class="drop-hint">Supported: PDF, CSV &middot; Multiple files &middot; English &amp; Spanish</div>
          </div>
          <div id="fileListDisplay" style="display:none;margin-top:10px"></div>
          <div class="field" style="margin-top:16px">
            <label>Embedding Model</label>
            <select id="ingestModel">
              <option value="both">Both (OpenAI + Cohere)</option>
              <option value="openai">OpenAI only</option>
              <option value="cohere">Cohere only</option>
            </select>
          </div>
          <button class="btn btn-primary btn-full" id="ingestBtn" disabled>Ingest Documents</button>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Quick Start</span></div>
        <div class="card-body">
          <p style="font-size:13px;color:var(--text-2);line-height:1.7;margin-bottom:12px">Upload PDF or CSV files to index into Pinecone. Use the Query or Chat tabs to retrieve answers.</p>
          <p style="font-size:12px;color:var(--text-3);font-family:var(--mono)">Sample files:<br>data/sales_data.csv<br>data/ventas_datos.csv<br>data/corporate_report_en.pdf<br>data/informe_corporativo_es.pdf</p>
        </div>
      </div>
    </div>
    <div>
      <div class="card">
        <div class="card-header">
          <span class="card-title">Ingestion Results</span>
          <span id="ingestStatus"></span>
        </div>
        <div class="card-body" id="ingestResults">
          <div class="empty-state"><div class="empty-label">Upload a document to begin</div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════
     QUERY
     ══════════════════════════════════════════ -->
<div class="tab-panel" id="tab-query">
  <div class="two-col">
    <div>
      <div class="card">
        <div class="card-header"><span class="card-title">Query Parameters</span></div>
        <div class="card-body">
          <div class="field">
            <label>Question</label>
            <textarea id="queryText" rows="3" placeholder="Ask in English or Spanish..."></textarea>
          </div>
          <div class="inline-row">
            <div class="field">
              <label>Embedding Model</label>
              <select id="queryModel">
                <option value="openai">OpenAI</option>
                <option value="cohere">Cohere</option>
                <option value="both">Both (side-by-side)</option>
              </select>
            </div>
            <div class="field">
              <label>Top-K</label>
              <input type="number" id="queryTopK" value="5" min="1" max="20">
            </div>
          </div>
          <div class="field">
            <label>Filter: source_type</label>
            <select id="queryFilterType">
              <option value="">Any</option>
              <option value="pdf">PDF only</option>
              <option value="csv">CSV only</option>
            </select>
          </div>
          <div class="field">
            <label>Score Threshold</label>
            <input type="number" id="queryThreshold" value="0.0" min="0" max="1" step="0.05">
          </div>
          <button class="btn btn-primary btn-full" id="queryBtn">Ask Question</button>
        </div>
      </div>
    </div>
    <div id="queryOutput">
      <div class="empty-state card" style="padding:40px">
        <div class="empty-label">Enter a question and click Ask</div>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════
     COMPARE
     ══════════════════════════════════════════ -->
<div class="tab-panel" id="tab-compare">
  <div class="card" style="margin-bottom:16px">
    <div class="card-header"><span class="card-title">Compare Embeddings</span></div>
    <div class="card-body">
      <div style="display:flex;gap:12px;align-items:flex-end">
        <div class="field" style="margin:0;flex:1">
          <label>Query</label>
          <input type="text" id="compareQuery" placeholder="Query both models simultaneously...">
        </div>
        <div class="field" style="margin:0;width:100px">
          <label>Top-K</label>
          <input type="number" id="compareTopK" value="5" min="1" max="20">
        </div>
        <button class="btn btn-primary" id="compareBtn">Compare</button>
      </div>
    </div>
  </div>
  <div id="compareOutput">
    <div class="empty-state card" style="padding:40px">
      <div class="empty-label">Run a comparison to see side-by-side results</div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════
     CHAT
     ══════════════════════════════════════════ -->
<div class="tab-panel" id="tab-chat">
  <div class="chat-layout">
    <div class="chat-toolbar">
      <span class="chat-toolbar-title">RAG Chat</span>
      <span style="font-size:11px;color:var(--text-3);font-family:var(--mono)">Model:</span>
      <select id="chatModel">
        <option value="openai">OpenAI</option>
        <option value="cohere">Cohere</option>
      </select>
      <span style="font-size:11px;color:var(--text-3);font-family:var(--mono)">Top-K:</span>
      <input type="number" id="chatTopK" value="5" min="1" max="20">
      <div class="chat-spacer"></div>
      <button class="btn btn-secondary" id="chatClearBtn" style="padding:6px 12px;font-size:12px">Clear Chat</button>
    </div>
    <div class="chat-messages" id="chatMessages">
      <div class="chat-msg assistant">
        <div class="chat-avatar">AI</div>
        <div class="chat-msg-wrap">
          <div class="chat-bubble">Hello! I am your RAG assistant. Ask me anything about your ingested documents — I will cite my sources inline.</div>
        </div>
      </div>
    </div>
    <div class="chat-input-bar">
      <textarea id="chatInput" placeholder="Ask a question... (Enter to send, Shift+Enter for new line)" rows="1"></textarea>
      <button class="btn btn-primary" id="chatSendBtn" style="flex-shrink:0;height:40px">Send</button>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════
     EVALUATE (RAGAS)
     ══════════════════════════════════════════ -->
<div class="tab-panel" id="tab-evaluate">

  <div class="info-banner">
    <strong>RAGAS Evaluation Framework</strong> &mdash; All evaluation is powered by RAGAS.
    <strong>Faithfulness</strong> and <strong>Answer Relevancy</strong> are always computed.
    Provide a <strong>Ground Truth</strong> answer to also get <strong>Context Recall</strong> and <strong>Context Precision</strong>.
  </div>

  <div class="eval-subtabs">
    <button class="eval-subtab active" data-evaltab="single">Single Query Eval</button>
    <button class="eval-subtab" data-evaltab="benchmark">Benchmark</button>
  </div>

  <!-- Single Query Eval -->
  <div class="eval-subpanel active" id="evaltab-single">
    <div class="two-col">
      <div>
        <div class="card">
          <div class="card-header">
            <span class="card-title">Single Query RAGAS Eval</span>
            <span class="ragas-badge">RAGAS</span>
          </div>
          <div class="card-body">
            <div class="field">
              <label>Query</label>
              <textarea id="singleEvalQuery" rows="3" placeholder="Enter your evaluation query..."></textarea>
            </div>
            <div class="inline-row">
              <div class="field">
                <label>Embedding Model</label>
                <select id="singleEvalModel">
                  <option value="openai">OpenAI</option>
                  <option value="cohere">Cohere</option>
                </select>
              </div>
              <div class="field">
                <label>Top-K</label>
                <input type="number" id="singleEvalTopK" value="5" min="1" max="20">
              </div>
            </div>
            <div class="field">
              <label>Ground Truth <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3)">(optional — enables Context Recall &amp; Precision)</span></label>
              <textarea id="singleEvalGroundTruth" rows="3" placeholder="Reference answer text... Leave blank to skip reference-based metrics."></textarea>
            </div>
            <div class="field">
              <label>Pre-generated Answer <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3)">(optional — leave blank to auto-generate)</span></label>
              <textarea id="singleEvalAnswer" rows="2" placeholder="Paste your answer here, or leave blank to auto-generate via RAG pipeline..."></textarea>
            </div>
            <button class="btn btn-ragas btn-full" id="singleEvalBtn">Run RAGAS Evaluation</button>
          </div>
        </div>
      </div>
      <div id="singleEvalOutput">
        <div class="empty-state card" style="padding:40px">
          <div class="empty-label">Configure and run a RAGAS evaluation</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Benchmark -->
  <div class="eval-subpanel" id="evaltab-benchmark">
    <div class="two-col">
      <div>
        <div class="card">
          <div class="card-header">
            <span class="card-title">RAGAS Benchmark</span>
            <span class="ragas-badge">RAGAS</span>
          </div>
          <div class="card-body">
            <div class="field">
              <label>Embedding Model</label>
              <select id="benchEvalModel">
                <option value="openai">OpenAI</option>
                <option value="cohere">Cohere</option>
                <option value="both">Both (Side-by-Side)</option>
              </select>
            </div>
            <div class="field">
              <label>K (top results)</label>
              <input type="number" id="benchEvalK" value="5" min="1" max="20">
            </div>
            <button class="btn btn-secondary btn-full" id="loadRagasBenchmarkBtn" style="margin-bottom:10px">Load Sample Benchmark</button>
            <div class="field">
              <label>Queries JSON
                <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3)">— each entry: query (required), ground_truth (optional)</span>
              </label>
              <textarea id="benchEvalQueries" rows="10" placeholder='[{"query":"...", "ground_truth":"..."}, ...]' style="font-family:var(--mono);font-size:11px"></textarea>
            </div>
            <button class="btn btn-ragas btn-full" id="benchEvalBtn">Run RAGAS Benchmark</button>
          </div>
        </div>
      </div>
      <div id="benchEvalOutput">
        <div class="empty-state card" style="padding:40px">
          <div class="empty-label">Run benchmark to see RAGAS metrics</div>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- ══════════════════════════════════════════
     COLLECTIONS
     ══════════════════════════════════════════ -->
<div class="tab-panel" id="tab-collections">
  <div id="collectionsOutput">
    <div class="empty-state card" style="padding:40px">
      <div class="empty-label">Loading collection stats...</div>
    </div>
  </div>
</div>

</main>
<div id="toast"></div>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────
const API = '';

function apiFetch(path, opts) {
  return fetch(API + path, opts || {}).then(function(res) {
    if (!res.ok) {
      return res.json()
        .catch(function() { return { detail: res.statusText }; })
        .then(function(e) { throw new Error(e.detail || JSON.stringify(e)); });
    }
    return res.json();
  });
}

function showToast(msg, type) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show ' + (type || 'info');
  setTimeout(function() { t.className = ''; }, 3500);
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  if (loading) btn.classList.add('loading');
  else btn.classList.remove('loading');
}

function latCls(ms) { return ms < 300 ? 'fast' : ms > 1200 ? 'slow' : ''; }
function scoreColor(s) { return s >= 0.8 ? '#16a34a' : s >= 0.6 ? '#d97706' : '#9ca3af'; }
function ragasColor(s) { return s >= 0.75 ? 'var(--success)' : s >= 0.5 ? '#d97706' : 'var(--danger)'; }

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// Render answer text with inline citation superscripts [N] → clickable badge
function renderWithCites(text, citations, uid) {
  var html = esc(text).replace(/\n/g, '<br>');
  html = html.replace(/\[(\d+)\]/g, function(m, n) {
    var idx = parseInt(n) - 1;
    var cite = citations && citations[idx];
    if (cite) {
      var tip = esc(cite.source) + ' - ' + esc(cite.location);
      return '<sup class="cite-sup" data-idx="' + idx + '" data-uid="' + (uid || '') + '" title="' + tip + '">[' + n + ']</sup>';
    }
    return m;
  });
  return html;
}

function accuracyBadge(citations) {
  if (!citations || !citations.length) return '';
  var avg = citations.reduce(function(s, c) { return s + c.score; }, 0) / citations.length;
  var cls = avg >= 0.75 ? 'high' : avg >= 0.5 ? 'med' : 'low';
  var label = avg >= 0.75 ? 'High Accuracy' : avg >= 0.5 ? 'Medium Accuracy' : 'Low Accuracy';
  return '<span class="accuracy-badge ' + cls + '">' + label + ' (' + avg.toFixed(2) + ')</span>';
}

// ─────────────────────────────────────────────────────────────────────────────
// Health check
// ─────────────────────────────────────────────────────────────────────────────
function checkHealth() {
  apiFetch('/api/collections').then(function() {
    document.getElementById('statusDot').className = 'status-dot ok';
    document.getElementById('statusLabel').textContent = 'Connected';
  }).catch(function() {
    document.getElementById('statusDot').className = 'status-dot error';
    document.getElementById('statusLabel').textContent = 'Offline';
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab navigation
// ─────────────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
    btn.classList.add('active');
    var panel = document.getElementById('tab-' + btn.dataset.tab);
    if (panel) panel.classList.add('active');
    if (btn.dataset.tab === 'collections') loadCollections();
  });
});

// Evaluate sub-tabs
document.querySelectorAll('.eval-subtab').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.eval-subtab').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.eval-subpanel').forEach(function(p) { p.classList.remove('active'); });
    btn.classList.add('active');
    var panel = document.getElementById('evaltab-' + btn.dataset.evaltab);
    if (panel) panel.classList.add('active');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// INGEST
// ─────────────────────────────────────────────────────────────────────────────
var dropZone = document.getElementById('dropZone');
var fileInput = document.getElementById('fileInput');
var fileListDisplay = document.getElementById('fileListDisplay');
var ingestBtn = document.getElementById('ingestBtn');
var selectedFiles = [];

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function renderFileList() {
  if (!selectedFiles.length) {
    fileListDisplay.style.display = 'none';
    ingestBtn.disabled = true;
    return;
  }
  fileListDisplay.style.display = 'block';
  ingestBtn.disabled = false;
  fileListDisplay.innerHTML = selectedFiles.map(function(f, i) {
    return '<div class="file-list-item">'
      + '<span>' + esc(f.name) + '</span>'
      + '<span class="file-size">' + fmtBytes(f.size) + '</span>'
      + '<button class="file-remove" data-idx="' + i + '">&times;</button>'
      + '</div>';
  }).join('');
  fileListDisplay.querySelectorAll('.file-remove').forEach(function(btn) {
    btn.addEventListener('click', function() {
      selectedFiles.splice(parseInt(btn.dataset.idx), 1);
      renderFileList();
    });
  });
}

function addFiles(files) {
  Array.from(files).forEach(function(f) {
    if (!selectedFiles.some(function(s) { return s.name === f.name && s.size === f.size; })) {
      selectedFiles.push(f);
    }
  });
  renderFileList();
}

fileInput.addEventListener('change', function() {
  if (fileInput.files.length) addFiles(fileInput.files);
  fileInput.value = '';
});
dropZone.addEventListener('dragover', function(e) {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', function(e) {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});

ingestBtn.addEventListener('click', async function() {
  if (!selectedFiles.length) return;
  setLoading(ingestBtn, true);
  var total = selectedFiles.length, results = [], errors = [];

  for (var i = 0; i < selectedFiles.length; i++) {
    var f = selectedFiles[i];
    var pid = 'prog-' + i;
    document.getElementById('ingestResults').innerHTML =
      '<div style="padding:20px 18px">'
      + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
      + '<span style="font-family:var(--mono);font-size:12px;color:var(--text-2)">' + esc(f.name) + '</span>'
      + '<span style="font-family:var(--mono);font-size:11px;color:var(--text-3)">File ' + (i + 1) + ' of ' + total + '</span>'
      + '</div>'
      + '<div class="progress-wrap"><div class="progress-bar" id="' + pid + '"></div></div>'
      + '<div class="progress-msg" id="msg-' + pid + '">Uploading...</div>'
      + '</div>';

    try {
      var fd = new FormData();
      fd.append('file', f);
      fd.append('embedding_model', document.getElementById('ingestModel').value);

      var result = await new Promise(function(resolve, reject) {
        fetch('/api/ingest', { method: 'POST', body: fd }).then(function(resp) {
          if (!resp.ok) {
            resp.json()
              .catch(function() { return { detail: resp.statusText }; })
              .then(function(e) { reject(new Error(e.detail || JSON.stringify(e))); });
            return;
          }
          var reader = resp.body.getReader();
          var decoder = new TextDecoder();
          var buf = '';
          function pump() {
            return reader.read().then(function(chunk) {
              if (chunk.done) return;
              buf += decoder.decode(chunk.value, { stream: true });
              var lines = buf.split('\n');
              buf = lines.pop();
              for (var j = 0; j < lines.length; j++) {
                var line = lines[j];
                if (!line.startsWith('data: ')) continue;
                try {
                  var ev = JSON.parse(line.slice(6));
                  if (ev.type === 'progress') {
                    var bar = document.getElementById(pid);
                    var msg = document.getElementById('msg-' + pid);
                    if (bar) bar.style.width = ev.percent + '%';
                    if (msg) msg.textContent = ev.message;
                  } else if (ev.type === 'done') {
                    var bar2 = document.getElementById(pid);
                    if (bar2) bar2.style.width = '100%';
                    resolve(ev);
                  } else if (ev.type === 'error') {
                    reject(new Error(ev.message));
                  }
                } catch (pe) {}
              }
              return pump();
            }).catch(reject);
          }
          pump();
        }).catch(reject);
      });

      results.push(result);
    } catch (e) {
      errors.push({ name: f.name, error: e.message });
    }
  }

  document.getElementById('ingestResults').innerHTML = renderMultiIngest(results, errors);
  if (results.length) showToast('Indexed ' + results.length + ' of ' + total + ' successfully', 'success');
  if (errors.length) showToast(errors.length + ' file(s) failed', 'error');
  selectedFiles = [];
  fileInput.value = '';
  renderFileList();
  setLoading(ingestBtn, false);
});

function renderOneIngest(d) {
  var rows = [
    ['Type', d.source_type.toUpperCase()],
    ['Raw documents', d.raw_documents],
    ['Semantic chunks', d.chunks],
  ];
  if (d.indexed.openai !== undefined) rows.push(['OpenAI vectors', d.indexed.openai]);
  if (d.indexed.cohere !== undefined) rows.push(['Cohere vectors', d.indexed.cohere]);
  return '<div style="border:1px solid var(--border);border-radius:var(--radius-sm);margin-bottom:10px;overflow:hidden">'
    + '<div style="padding:8px 14px;background:var(--surface-2);border-bottom:1px solid var(--border-light);display:flex;align-items:center;gap:8px">'
    + '<span style="font-family:var(--mono);font-size:12px">' + esc(d.filename) + '</span>'
    + '<span style="margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--success);background:var(--success-bg);border:1px solid #86efac;border-radius:99px;padding:1px 8px">done</span>'
    + '</div>'
    + '<div style="padding:6px 14px">'
    + rows.map(function(r) {
        return '<div class="stat-row"><span class="stat-key">' + r[0] + '</span><span class="stat-val">' + r[1] + '</span></div>';
      }).join('')
    + '</div></div>';
}

function renderMultiIngest(results, errors) {
  var total = results.length + errors.length;
  var summary = total > 1
    ? '<div style="margin-bottom:14px;padding:10px 14px;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);font-family:var(--condensed);font-size:13px;color:var(--text-2)">'
      + results.length + ' of ' + total + ' files indexed successfully</div>'
    : '';
  var s = results.map(renderOneIngest).join('');
  var e = errors.map(function(err) {
    return '<div style="border:1px solid #fca5a5;border-radius:var(--radius-sm);margin-bottom:10px;padding:10px 14px;background:var(--danger-bg);font-size:12px;color:var(--danger)">'
      + esc(err.name) + ': ' + esc(err.error) + '</div>';
  }).join('');
  return '<div style="padding-top:4px">' + summary + s + e + '</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// QUERY  (single-model streaming + "both" client-side fan-out)
// ─────────────────────────────────────────────────────────────────────────────
function queryPayload(model) {
  var ft = document.getElementById('queryFilterType').value;
  return {
    query: document.getElementById('queryText').value.trim(),
    embedding_model: model,
    top_k: parseInt(document.getElementById('queryTopK').value),
    filters: ft ? { source_type: ft } : null,
    score_threshold: parseFloat(document.getElementById('queryThreshold').value),
  };
}

document.getElementById('queryBtn').addEventListener('click', async function() {
  var query = document.getElementById('queryText').value.trim();
  if (!query) { showToast('Enter a question first', 'error'); return; }
  var btn = document.getElementById('queryBtn');
  var model = document.getElementById('queryModel').value;
  setLoading(btn, true);

  if (model === 'both') {
    document.getElementById('queryOutput').innerHTML =
      '<div class="empty-state card" style="padding:40px"><div class="empty-label">Querying both models...</div></div>';
    try {
      var results = await Promise.all([
        collectStream(queryPayload('openai')),
        collectStream(queryPayload('cohere')),
      ]);
      document.getElementById('queryOutput').innerHTML = renderBothQuery(results[0], results[1]);
      attachCiteHandlers();
    } catch (e) {
      document.getElementById('queryOutput').innerHTML = '<div class="error-box">' + esc(e.message) + '</div>';
      showToast(e.message, 'error');
    }
    setLoading(btn, false);
    return;
  }

  var uid = model + Date.now();
  var mBadge = '<span class="model-badge ' + model + '">' + (model === 'openai' ? 'OpenAI' : 'Cohere') + '</span>';

  document.getElementById('queryOutput').innerHTML =
    '<div style="display:flex;flex-direction:column;gap:14px">'
    + '<div class="card"><div class="card-header"><span class="card-title">Answer</span>'
    + '<div id="q-meta" style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">' + mBadge + '</div></div>'
    + '<div class="card-body"><div class="answer-box stream-cursor" id="q-answer"></div></div></div>'
    + '<div id="q-sources"></div>'
    + '</div>';

  var answerEl  = document.getElementById('q-answer');
  var metaEl    = document.getElementById('q-meta');
  var sourcesEl = document.getElementById('q-sources');
  var fullAnswer = '', citations = [], ret_ms = 0;

  try {
    var resp = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(queryPayload(model)),
    });
    if (!resp.ok) {
      var e2 = await resp.json().catch(function() { return { detail: resp.statusText }; });
      throw new Error(e2.detail || JSON.stringify(e2));
    }
    var reader = resp.body.getReader(), decoder = new TextDecoder(), buf = '';
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buf += decoder.decode(chunk.value, { stream: true });
      var lines = buf.split('\n'); buf = lines.pop();
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        if (!line.startsWith('data: ')) continue;
        try {
          var ev = JSON.parse(line.slice(6));
          if (ev.type === 'citations') {
            citations = ev.citations;
            ret_ms = ev.retrieval_latency_ms;
            metaEl.innerHTML = mBadge
              + '<span class="latency-pill ' + latCls(ret_ms) + '">Retrieval: ' + ret_ms + 'ms</span>'
              + '<span class="latency-pill">Generating...</span>';
            if (citations.length) {
              sourcesEl.innerHTML = '<div class="card"><div class="card-header"><span class="card-title">Sources (' + citations.length + ')</span></div>'
                + '<div class="card-body" style="padding-top:10px">' + renderCitations(citations, uid) + '</div></div>';
            }
          } else if (ev.type === 'token') {
            fullAnswer += ev.content;
            answerEl.innerHTML = esc(fullAnswer).replace(/\n/g, '<br>');
          } else if (ev.type === 'done') {
            answerEl.classList.remove('stream-cursor');
            answerEl.innerHTML = renderWithCites(fullAnswer, citations, uid);
            var acc = accuracyBadge(citations);
            metaEl.innerHTML = mBadge
              + '<span class="latency-pill ' + latCls(ret_ms) + '">Retrieval: ' + ret_ms + 'ms</span>'
              + '<span class="latency-pill ' + latCls(ev.generation_latency_ms) + '">Generation: ' + ev.generation_latency_ms + 'ms</span>'
              + (ev.tokens_used ? '<span class="latency-pill">Tokens: ' + ev.tokens_used.total + '</span>' : '')
              + acc;
            attachCiteHandlers();
          } else if (ev.type === 'error') {
            answerEl.classList.remove('stream-cursor');
            throw new Error(ev.message);
          }
        } catch (pe) { if (pe.message) throw pe; }
      }
    }
  } catch (e) {
    document.getElementById('queryOutput').innerHTML = '<div class="error-box">' + esc(e.message) + '</div>';
    showToast(e.message, 'error');
  }
  setLoading(btn, false);
});

// Collect a full streaming response into a plain object (used for "both" mode)
async function collectStream(payload) {
  var resp = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    var e2 = await resp.json().catch(function() { return { detail: resp.statusText }; });
    throw new Error(e2.detail || JSON.stringify(e2));
  }
  var reader = resp.body.getReader(), decoder = new TextDecoder(), buf = '';
  var out = {
    embedding_model: payload.embedding_model,
    answer: '', citations: [],
    retrieval_latency_ms: 0, generation_latency_ms: 0, tokens_used: null,
  };
  while (true) {
    var chunk = await reader.read();
    if (chunk.done) break;
    buf += decoder.decode(chunk.value, { stream: true });
    var lines = buf.split('\n'); buf = lines.pop();
    for (var i = 0; i < lines.length; i++) {
      if (!lines[i].startsWith('data: ')) continue;
      try {
        var ev = JSON.parse(lines[i].slice(6));
        if (ev.type === 'citations') { out.citations = ev.citations; out.retrieval_latency_ms = ev.retrieval_latency_ms; }
        else if (ev.type === 'token') { out.answer += ev.content; }
        else if (ev.type === 'done') { out.generation_latency_ms = ev.generation_latency_ms; out.tokens_used = ev.tokens_used; }
        else if (ev.type === 'error') { throw new Error(ev.message); }
      } catch (pe) { if (pe.message) throw pe; }
    }
  }
  return out;
}

function renderCitations(citations, uid) {
  return citations.map(function(c) {
    return '<div class="citation-item" id="cite-' + uid + '-' + c.index + '">'
      + '<div class="citation-num">' + c.index + '</div>'
      + '<div class="citation-content">'
      + '<div class="citation-header">'
      + '<span class="citation-source">' + esc(c.source) + '</span>'
      + '<span class="citation-loc">' + esc(c.location) + '</span>'
      + '<span class="score-val" style="color:' + scoreColor(c.score) + '">' + c.score.toFixed(4) + '</span>'
      + '</div>'
      + '<div class="citation-preview">' + esc(c.preview) + '</div>'
      + '</div></div>';
  }).join('');
}

function renderQuery(d, uid) {
  var model = d.embedding_model;
  var mBadge = '<span class="model-badge ' + model + '">' + (model === 'openai' ? 'OpenAI' : 'Cohere') + '</span>';
  var acc = accuracyBadge(d.citations);
  var cites = renderCitations(d.citations || [], uid);
  return '<div style="display:flex;flex-direction:column;gap:14px">'
    + '<div class="card"><div class="card-header"><span class="card-title">Answer</span>'
    + '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">'
    + mBadge
    + '<span class="latency-pill ' + latCls(d.retrieval_latency_ms) + '">Retrieval: ' + d.retrieval_latency_ms + 'ms</span>'
    + '<span class="latency-pill ' + latCls(d.generation_latency_ms) + '">Generation: ' + d.generation_latency_ms + 'ms</span>'
    + (d.tokens_used ? '<span class="latency-pill">Tokens: ' + d.tokens_used.total + '</span>' : '')
    + acc + '</div></div>'
    + '<div class="card-body"><div class="answer-box">' + renderWithCites(d.answer, d.citations, uid) + '</div></div></div>'
    + (cites
        ? '<div class="card"><div class="card-header"><span class="card-title">Sources (' + d.citations.length + ')</span></div>'
          + '<div class="card-body" style="padding-top:10px">' + cites + '</div></div>'
        : '')
    + '</div>';
}

function renderBothQuery(oa, co) {
  return '<div style="display:flex;flex-direction:column;gap:12px">'
    + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start">'
    + renderQuery(oa, 'oa') + renderQuery(co, 'co')
    + '</div></div>';
}

function renderChunk(chunk, rank, uid) {
  var meta = chunk.metadata || {}, source = meta.source || 'unknown';
  var loc = meta.page
    ? '<span class="chunk-meta">page ' + meta.page + '</span>'
    : meta.row_start
      ? '<span class="chunk-meta">rows ' + meta.row_start + '-' + meta.row_end + '</span>'
      : meta.doc_type
        ? '<span class="chunk-meta">' + meta.doc_type + '</span>'
        : '';
  if (meta.language) loc += ' <span class="chunk-meta">' + meta.language + '</span>';
  var sb = '<div class="score-bar-wrap" style="min-width:80px;flex:1;max-width:120px">'
    + '<div class="score-bar"><div class="score-bar-fill" style="width:' + (chunk.score * 100) + '%;background:' + scoreColor(chunk.score) + '"></div></div>'
    + '<span class="score-val">' + chunk.score.toFixed(4) + '</span></div>';
  var id = 'cb-' + uid + '-' + rank;
  return '<div class="chunk-item">'
    + '<div class="chunk-header"><span class="chunk-rank">#' + rank + '</span><span class="chunk-source">' + esc(source) + '</span>' + loc + '<div style="margin-left:auto">' + sb + '</div></div>'
    + '<div class="chunk-body" id="' + id + '">' + esc(chunk.text) + '</div>'
    + '<span class="chunk-expand" data-target="' + id + '">Show more</span>'
    + '</div>';
}

function attachExpandHandlers() {
  document.querySelectorAll('.chunk-expand').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var t = document.getElementById(btn.dataset.target);
      if (t.classList.contains('expanded')) {
        t.classList.remove('expanded'); btn.textContent = 'Show more';
      } else {
        t.classList.add('expanded'); btn.textContent = 'Show less';
      }
    });
  });
}

function attachCiteHandlers() {
  document.querySelectorAll('.cite-sup').forEach(function(sup) {
    sup.addEventListener('click', function() {
      var idx = parseInt(sup.dataset.idx);
      var uid = sup.dataset.uid || '';
      var citeEl = document.getElementById('cite-' + uid + '-' + (idx + 1));
      if (!citeEl) {
        var all = document.querySelectorAll('.citation-item');
        for (var i = 0; i < all.length; i++) {
          if (all[i].id.endsWith('-' + (idx + 1))) { citeEl = all[i]; break; }
        }
      }
      if (citeEl) {
        document.querySelectorAll('.citation-item').forEach(function(el) { el.classList.remove('highlighted'); });
        document.querySelectorAll('.cite-sup').forEach(function(el) { el.classList.remove('active'); });
        citeEl.classList.add('highlighted');
        sup.classList.add('active');
        citeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPARE
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('compareBtn').addEventListener('click', async function() {
  var query = document.getElementById('compareQuery').value.trim();
  if (!query) { showToast('Enter a query', 'error'); return; }
  var btn = document.getElementById('compareBtn');
  setLoading(btn, true);
  document.getElementById('compareOutput').innerHTML =
    '<div class="empty-state card" style="padding:40px"><div class="empty-label">Querying both models...</div></div>';
  try {
    var data = await apiFetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, top_k: parseInt(document.getElementById('compareTopK').value) }),
    });
    document.getElementById('compareOutput').innerHTML = renderCompare(data);
    attachExpandHandlers();
  } catch (e) {
    document.getElementById('compareOutput').innerHTML = '<div class="error-box">' + esc(e.message) + '</div>';
    showToast(e.message, 'error');
  }
  setLoading(btn, false);
});

function renderCompare(d) {
  var c = d.comparison, op = Math.round(c.result_overlap * 100);
  var summary = '<div class="card" style="margin-bottom:16px"><div class="card-header"><span class="card-title">Comparison Summary</span></div><div class="card-body">'
    + '<div class="metric-grid">'
    + '<div class="metric-card"><div class="metric-value" style="color:var(--openai)">' + c.openai_latency_ms + '<span style="font-size:13px">ms</span></div><div class="metric-label">OpenAI Latency</div></div>'
    + '<div class="metric-card"><div class="metric-value" style="color:var(--cohere)">' + c.cohere_latency_ms + '<span style="font-size:13px">ms</span></div><div class="metric-label">Cohere Latency</div></div>'
    + '<div class="metric-card"><div class="metric-value" style="color:var(--openai)">' + c.openai_top_score.toFixed(4) + '</div><div class="metric-label">OpenAI Top Score</div></div>'
    + '<div class="metric-card"><div class="metric-value" style="color:var(--cohere)">' + c.cohere_top_score.toFixed(4) + '</div><div class="metric-label">Cohere Top Score</div></div>'
    + '</div>'
    + '<div><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">'
    + '<span style="color:var(--text-2);font-family:var(--condensed);font-weight:600;text-transform:uppercase;letter-spacing:.5px">Result Overlap</span>'
    + '<span style="font-family:var(--mono)">' + op + '%</span></div>'
    + '<div class="overlap-bar"><div class="overlap-fill" style="width:' + op + '%"></div></div>'
    + '<div style="font-size:11px;color:var(--text-3);margin-top:4px;font-family:var(--mono)">'
    + c.openai_total + ' OpenAI &middot; ' + c.cohere_total + ' Cohere &middot; Jaccard: ' + c.result_overlap.toFixed(4)
    + '</div></div></div></div>';

  var oc = (d.openai.results || []).map(function(ch, i) { return renderChunk(ch, i + 1, 'oa'); }).join('');
  var cc = (d.cohere.results || []).map(function(ch, i) { return renderChunk(ch, i + 1, 'co'); }).join('');

  return summary + '<div class="compare-grid">'
    + '<div class="card"><div class="card-header"><span style="font-family:var(--condensed);font-weight:600;font-size:14px;color:var(--openai)">OpenAI text-embedding-3-large</span><span class="latency-pill ' + latCls(c.openai_latency_ms) + '">' + c.openai_latency_ms + 'ms</span></div><div class="card-body" style="padding-top:10px">' + (oc || '<div class="empty-state">No results</div>') + '</div></div>'
    + '<div class="card"><div class="card-header"><span style="font-family:var(--condensed);font-weight:600;font-size:14px;color:var(--cohere)">Cohere embed-v-4-0</span><span class="latency-pill ' + latCls(c.cohere_latency_ms) + '">' + c.cohere_latency_ms + 'ms</span></div><div class="card-body" style="padding-top:10px">' + (cc || '<div class="empty-state">No results</div>') + '</div></div>'
    + '</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// CHAT
// ─────────────────────────────────────────────────────────────────────────────
var chatHistory = [];

async function sendChat() {
  var input = document.getElementById('chatInput');
  var query = input.value.trim();
  if (!query) return;
  input.value = '';
  input.style.height = '40px';

  var msgs = document.getElementById('chatMessages');

  var userEl = document.createElement('div');
  userEl.className = 'chat-msg user';
  userEl.innerHTML = '<div class="chat-avatar">U</div>'
    + '<div class="chat-msg-wrap"><div class="chat-bubble">' + esc(query) + '</div></div>';
  msgs.appendChild(userEl);
  msgs.scrollTop = msgs.scrollHeight;

  chatHistory.push({ role: 'user', content: query });

  var btn = document.getElementById('chatSendBtn');
  setLoading(btn, true);

  var msgEl = document.createElement('div');
  msgEl.className = 'chat-msg assistant';
  var sid = 'cs-' + Date.now();
  msgEl.innerHTML = '<div class="chat-avatar">AI</div>'
    + '<div class="chat-msg-wrap"><div class="chat-bubble stream-cursor" id="' + sid + '">'
    + '<div class="typing-dot"><span></span><span></span><span></span></div>'
    + '</div></div>';
  msgs.appendChild(msgEl);
  msgs.scrollTop = msgs.scrollHeight;

  var bubbleEl   = document.getElementById(sid);
  var wrapEl     = msgEl.querySelector('.chat-msg-wrap');
  var fullAnswer = '', citations = [], ret_ms = 0;
  var model      = document.getElementById('chatModel').value;

  try {
    var resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        history: chatHistory.slice(-6),
        embedding_model: model,
        top_k: parseInt(document.getElementById('chatTopK').value),
      }),
    });
    if (!resp.ok) {
      var e2 = await resp.json().catch(function() { return { detail: resp.statusText }; });
      throw new Error(e2.detail || JSON.stringify(e2));
    }
    var reader = resp.body.getReader(), decoder = new TextDecoder(), buf = '';
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buf += decoder.decode(chunk.value, { stream: true });
      var lines = buf.split('\n'); buf = lines.pop();
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        if (!line.startsWith('data: ')) continue;
        try {
          var ev = JSON.parse(line.slice(6));
          if (ev.type === 'citations') {
            citations = ev.citations;
            ret_ms = ev.retrieval_latency_ms;
          } else if (ev.type === 'token') {
            fullAnswer += ev.content;
            bubbleEl.innerHTML = esc(fullAnswer).replace(/\n/g, '<br>');
            msgs.scrollTop = msgs.scrollHeight;
          } else if (ev.type === 'done') {
            bubbleEl.classList.remove('stream-cursor');
            bubbleEl.innerHTML = renderWithCites(fullAnswer, citations, sid);
            var modelLabel = model === 'openai' ? 'OpenAI' : 'Cohere';
            var acc = accuracyBadge(citations);
            var sources = citations.length
              ? '<div class="chat-sources">'
                + citations.map(function(c) {
                    return '<span class="chat-source-chip" title="' + esc(c.preview) + '">'
                      + esc(c.source) + ' &middot; ' + esc(c.location)
                      + ' <span style="opacity:.5">' + c.score.toFixed(2) + '</span>'
                      + '</span>';
                  }).join('')
                + '</div>'
              : '';
            var meta = '<div class="chat-meta">'
              + modelLabel + ' &middot; ' + ret_ms + 'ms ret &middot; '
              + ev.generation_latency_ms + 'ms gen ' + acc
              + '</div>';
            wrapEl.innerHTML += sources + meta;
            chatHistory.push({ role: 'assistant', content: fullAnswer });
            attachCiteHandlers();
          } else if (ev.type === 'error') {
            throw new Error(ev.message);
          }
        } catch (pe) { if (pe.message) throw pe; }
      }
    }
  } catch (e) {
    bubbleEl.classList.remove('stream-cursor');
    bubbleEl.innerHTML = '<span style="color:var(--danger)">Error: ' + esc(e.message) + '</span>';
  }
  setLoading(btn, false);
}

document.getElementById('chatSendBtn').addEventListener('click', sendChat);
document.getElementById('chatInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
document.getElementById('chatInput').addEventListener('input', function() {
  this.style.height = '40px';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});
document.getElementById('chatClearBtn').addEventListener('click', function() {
  chatHistory = [];
  document.getElementById('chatMessages').innerHTML =
    '<div class="chat-msg assistant"><div class="chat-avatar">AI</div>'
    + '<div class="chat-msg-wrap"><div class="chat-bubble">Chat cleared. Ask me anything about your documents.</div></div></div>';
  showToast('Chat cleared', 'info');
});

// ─────────────────────────────────────────────────────────────────────────────
// EVALUATE — Single query (RAGAS)
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('singleEvalBtn').addEventListener('click', async function() {
  var query = document.getElementById('singleEvalQuery').value.trim();
  if (!query) { showToast('Enter a query', 'error'); return; }
  var btn = document.getElementById('singleEvalBtn');
  setLoading(btn, true);
  document.getElementById('singleEvalOutput').innerHTML =
    '<div class="empty-state card" style="padding:40px"><div class="empty-label">Running RAGAS evaluation...</div></div>';

  var payload = {
    query: query,
    embedding_model: document.getElementById('singleEvalModel').value,
    top_k: parseInt(document.getElementById('singleEvalTopK').value),
  };
  var gt  = document.getElementById('singleEvalGroundTruth').value.trim();
  var ans = document.getElementById('singleEvalAnswer').value.trim();
  if (gt)  payload.ground_truth = gt;
  if (ans) payload.answer = ans;

  try {
    var data = await apiFetch('/api/ragas-evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    document.getElementById('singleEvalOutput').innerHTML = renderSingleRagas(data);
    showToast('RAGAS evaluation complete', 'success');
  } catch (e) {
    document.getElementById('singleEvalOutput').innerHTML = '<div class="error-box">' + esc(e.message) + '</div>';
    showToast(e.message, 'error');
  }
  setLoading(btn, false);
});

function renderRagasMetricCard(label, value, hasValue) {
  if (!hasValue || value === null || value === undefined) {
    return '<div class="ragas-metric-card" style="opacity:.45">'
      + '<div class="ragas-metric-value" style="font-size:16px;color:var(--text-3)">N/A</div>'
      + '<div class="ragas-metric-label">' + label + '</div>'
      + '<div style="font-size:10px;color:var(--text-3);margin-top:4px;font-family:var(--mono)">requires ground truth</div>'
      + '</div>';
  }
  var pct = Math.round(value * 100);
  return '<div class="ragas-metric-card">'
    + '<div class="ragas-metric-value">' + value.toFixed(4) + '</div>'
    + '<div class="ragas-metric-label">' + label + '</div>'
    + '<div class="ragas-score-bar"><div class="ragas-score-fill" style="width:' + pct + '%"></div></div>'
    + '</div>';
}

function renderSingleRagas(d) {
  var model = d.embedding_model;
  var mBadge = '<span class="model-badge ' + model + '">' + (model === 'openai' ? 'OpenAI' : 'Cohere') + '</span>';
  var hasRef = d.context_recall !== undefined && d.context_recall !== null;

  var cards = renderRagasMetricCard('Faithfulness', d.faithfulness, true)
    + renderRagasMetricCard('Answer Relevancy', d.answer_relevancy, true)
    + renderRagasMetricCard('Context Recall', d.context_recall, hasRef)
    + renderRagasMetricCard('Context Precision', d.context_precision, hasRef);

  var latency = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">'
    + '<span class="latency-pill ' + latCls(d.retrieval_latency_ms) + '">Retrieval: ' + d.retrieval_latency_ms + 'ms</span>'
    + '<span class="latency-pill ' + latCls(d.generation_latency_ms) + '">Generation: ' + d.generation_latency_ms + 'ms</span>'
    + '<span class="latency-pill ' + latCls(d.eval_latency_ms) + '">RAGAS Eval: ' + d.eval_latency_ms + 'ms</span>'
    + (d.tokens_used ? '<span class="latency-pill">Tokens: ' + d.tokens_used.total + '</span>' : '')
    + '</div>';

  var preview = d.answer_preview
    ? '<div class="card" style="margin-top:14px"><div class="card-header"><span class="card-title">Generated Answer Preview</span></div>'
      + '<div class="card-body"><div class="answer-box" style="font-size:13px">'
      + esc(d.answer_preview) + (d.answer_preview.length >= 300 ? '...' : '')
      + '</div></div></div>'
    : '';

  return '<div style="display:flex;flex-direction:column;gap:14px">'
    + '<div class="card"><div class="card-header"><span class="card-title">RAGAS Scores</span>'
    + '<div style="display:flex;gap:6px;align-items:center">' + mBadge + '<span class="ragas-badge">RAGAS</span></div></div>'
    + '<div class="card-body">'
    + '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px">' + cards + '</div>'
    + '<div class="stat-row"><span class="stat-key">Query</span><span class="stat-val" style="max-width:280px;text-overflow:ellipsis;overflow:hidden;white-space:nowrap">' + esc(d.query) + '</span></div>'
    + '<div class="stat-row"><span class="stat-key">Contexts Retrieved</span><span class="stat-val">' + d.num_contexts + '</span></div>'
    + '<div class="stat-row"><span class="stat-key">Latency</span><span class="stat-val">' + latency + '</span></div>'
    + '</div></div>'
    + preview
    + '</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// EVALUATE — Benchmark (RAGAS)
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('loadRagasBenchmarkBtn').addEventListener('click', async function() {
  try {
    var data = await apiFetch('/api/ragas-benchmark-sample');
    document.getElementById('benchEvalQueries').value = JSON.stringify(data, null, 2);
    showToast('Loaded ' + data.length + ' RAGAS benchmark queries', 'success');
  } catch (e) {
    showToast('Could not load benchmark queries', 'error');
  }
});

document.getElementById('benchEvalBtn').addEventListener('click', async function() {
  var raw = document.getElementById('benchEvalQueries').value.trim();
  if (!raw) { showToast('Enter benchmark queries', 'error'); return; }
  var queries;
  try { queries = JSON.parse(raw); } catch (e) { showToast('Invalid JSON', 'error'); return; }
  var btn = document.getElementById('benchEvalBtn');
  var model = document.getElementById('benchEvalModel').value;
  var k = parseInt(document.getElementById('benchEvalK').value);
  setLoading(btn, true);
  document.getElementById('benchEvalOutput').innerHTML =
    '<div class="empty-state card" style="padding:40px"><div class="empty-label">Running RAGAS benchmark — this may take a few minutes</div></div>';

  try {
    var data = await apiFetch('/api/ragas-benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ benchmark_queries: queries, embedding_model: model, k: k }),
    });
    if (model === 'both') {
      document.getElementById('benchEvalOutput').innerHTML = renderRagasBenchmarkCompare(data, k, queries.length);
    } else {
      document.getElementById('benchEvalOutput').innerHTML = renderRagasBenchmark(data);
    }
    showToast('RAGAS benchmark complete', 'success');
  } catch (e) {
    document.getElementById('benchEvalOutput').innerHTML = '<div class="error-box">' + esc(e.message) + '</div>';
    showToast(e.message, 'error');
  }
  setLoading(btn, false);
});

function renderRagasBenchmark(d) {
  var a = d.aggregate, model = d.embedding_model;
  var mBadge = '<span class="model-badge ' + model + '">' + (model === 'openai' ? 'OpenAI' : 'Cohere') + '</span>';
  var hasRef = a.avg_context_recall !== undefined;

  var cards = '<div class="ragas-metric-card"><div class="ragas-metric-value">' + a.avg_faithfulness.toFixed(4) + '</div><div class="ragas-metric-label">Avg Faithfulness</div><div class="ragas-score-bar"><div class="ragas-score-fill" style="width:' + Math.round(a.avg_faithfulness * 100) + '%"></div></div></div>'
    + '<div class="ragas-metric-card"><div class="ragas-metric-value">' + a.avg_answer_relevancy.toFixed(4) + '</div><div class="ragas-metric-label">Avg Answer Relevancy</div><div class="ragas-score-bar"><div class="ragas-score-fill" style="width:' + Math.round(a.avg_answer_relevancy * 100) + '%"></div></div></div>'
    + (hasRef
        ? '<div class="ragas-metric-card"><div class="ragas-metric-value">' + a.avg_context_recall.toFixed(4) + '</div><div class="ragas-metric-label">Avg Context Recall</div><div class="ragas-score-bar"><div class="ragas-score-fill" style="width:' + Math.round(a.avg_context_recall * 100) + '%"></div></div></div>'
        : '<div class="ragas-metric-card" style="opacity:.45"><div class="ragas-metric-value" style="font-size:16px;color:var(--text-3)">N/A</div><div class="ragas-metric-label">Avg Context Recall</div></div>')
    + (hasRef
        ? '<div class="ragas-metric-card"><div class="ragas-metric-value">' + a.avg_context_precision.toFixed(4) + '</div><div class="ragas-metric-label">Avg Context Precision</div><div class="ragas-score-bar"><div class="ragas-score-fill" style="width:' + Math.round(a.avg_context_precision * 100) + '%"></div></div></div>'
        : '<div class="ragas-metric-card" style="opacity:.45"><div class="ragas-metric-value" style="font-size:16px;color:var(--text-3)">N/A</div><div class="ragas-metric-label">Avg Context Precision</div></div>');

  var rows = (d.per_query || []).map(function(r, i) {
    if (r.error) {
      return '<tr><td>' + (i + 1) + '</td><td style="max-width:200px;word-break:break-word">' + esc(r.query) + '</td><td colspan="5" style="color:var(--danger);font-size:11px">' + esc(r.error) + '</td></tr>';
    }
    return '<tr>'
      + '<td>' + (i + 1) + '</td>'
      + '<td style="max-width:200px;word-break:break-word;font-size:12px">' + esc(r.query) + '</td>'
      + '<td class="mono-val" style="color:' + ragasColor(r.faithfulness) + '">' + r.faithfulness.toFixed(4) + '</td>'
      + '<td class="mono-val" style="color:' + ragasColor(r.answer_relevancy) + '">' + r.answer_relevancy.toFixed(4) + '</td>'
      + '<td class="mono-val">' + (r.context_recall !== null && r.context_recall !== undefined ? '<span style="color:' + ragasColor(r.context_recall) + '">' + r.context_recall.toFixed(4) + '</span>' : '<span style="color:var(--text-3)">—</span>') + '</td>'
      + '<td class="mono-val">' + (r.context_precision !== null && r.context_precision !== undefined ? '<span style="color:' + ragasColor(r.context_precision) + '">' + r.context_precision.toFixed(4) + '</span>' : '<span style="color:var(--text-3)">—</span>') + '</td>'
      + '<td class="mono-val">' + (r.retrieval_latency_ms || 0) + 'ms</td>'
      + '</tr>';
  }).join('');

  return '<div style="display:flex;flex-direction:column;gap:16px">'
    + '<div class="card"><div class="card-header"><span class="card-title">RAGAS Aggregate Metrics</span>'
    + '<div style="display:flex;gap:8px;align-items:center">' + mBadge + '<span class="ragas-badge">RAGAS</span><span class="latency-pill">K=' + d.k + '</span><span class="latency-pill">' + d.num_queries + ' queries</span><span class="latency-pill">' + d.valid_evaluations + ' valid / ' + d.failed_evaluations + ' failed</span></div></div>'
    + '<div class="card-body"><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">' + cards + '</div></div></div>'
    + '<div class="card"><div class="card-header"><span class="card-title">Per-Query RAGAS Results</span></div>'
    + '<div style="overflow-x:auto"><table class="data-table"><thead><tr>'
    + '<th>#</th><th>Query</th>'
    + '<th style="color:var(--ragas)">Faithfulness</th>'
    + '<th style="color:var(--ragas)">Ans. Relevancy</th>'
    + '<th style="color:var(--ragas)">Ctx Recall</th>'
    + '<th style="color:var(--ragas)">Ctx Precision</th>'
    + '<th>Ret. Latency</th>'
    + '</tr></thead><tbody>' + rows + '</tbody></table></div></div>'
    + '</div>';
}

function renderRagasBenchmarkCompare(data, k, numQ) {
  var oa = data.openai, co = data.cohere;
  var metrics = [
    ['Faithfulness',       'avg_faithfulness'],
    ['Answer Relevancy',   'avg_answer_relevancy'],
    ['Context Recall',     'avg_context_recall'],
    ['Context Precision',  'avg_context_precision'],
  ];

  var summaryRows = metrics.map(function(m) {
    var ov = oa.aggregate[m[1]], cv = co.aggregate[m[1]];
    if (ov === undefined && cv === undefined) {
      return '<tr><td style="font-family:var(--condensed);font-weight:600">' + m[0] + '</td>'
        + '<td class="mono-val" style="color:var(--text-3)">N/A</td>'
        + '<td class="mono-val" style="color:var(--text-3)">N/A</td></tr>';
    }
    ov = ov || 0; cv = cv || 0;
    var w = ov > cv ? 'openai' : cv > ov ? 'cohere' : '';
    return '<tr>'
      + '<td style="font-family:var(--condensed);font-weight:600">' + m[0] + '</td>'
      + '<td class="mono-val" style="color:' + (w === 'openai' ? 'var(--openai)' : 'var(--text-2)') + '"><b>' + ov.toFixed(4) + '</b>' + (w === 'openai' ? ' &#10003;' : '') + '</td>'
      + '<td class="mono-val" style="color:' + (w === 'cohere' ? 'var(--cohere)' : 'var(--text-2)') + '"><b>' + cv.toFixed(4) + '</b>' + (w === 'cohere' ? ' &#10003;' : '') + '</td>'
      + '</tr>';
  }).join('');

  var perQRows = (oa.per_query || []).map(function(r, i) {
    var cr = co.per_query && co.per_query[i] ? co.per_query[i] : {};
    var of_ = r.faithfulness || 0, cf_ = cr.faithfulness || 0;
    var or_ = r.answer_relevancy || 0, crr = cr.answer_relevancy || 0;
    return '<tr>'
      + '<td>' + (i + 1) + '</td>'
      + '<td style="max-width:160px;word-break:break-word;font-size:12px">' + esc(r.query) + '</td>'
      + '<td class="mono-val" style="color:' + (of_ >= cf_ ? 'var(--openai)' : 'var(--text-2)') + '">' + of_.toFixed(4) + '</td>'
      + '<td class="mono-val" style="color:' + (cf_ >= of_ ? 'var(--cohere)' : 'var(--text-2)') + '">' + cf_.toFixed(4) + '</td>'
      + '<td class="mono-val" style="color:' + (or_ >= crr ? 'var(--openai)' : 'var(--text-2)') + '">' + or_.toFixed(4) + '</td>'
      + '<td class="mono-val" style="color:' + (crr >= or_ ? 'var(--cohere)' : 'var(--text-2)') + '">' + crr.toFixed(4) + '</td>'
      + '</tr>';
  }).join('');

  return '<div style="display:flex;flex-direction:column;gap:16px">'
    + '<div class="card"><div class="card-header"><span class="card-title">OpenAI vs Cohere — RAGAS Benchmark</span>'
    + '<div style="display:flex;gap:8px"><span class="ragas-badge">RAGAS</span><span class="latency-pill">K=' + k + '</span><span class="latency-pill">' + numQ + ' queries</span></div></div>'
    + '<div class="card-body"><table class="data-table"><thead><tr><th>Metric</th><th style="color:var(--openai)">OpenAI</th><th style="color:var(--cohere)">Cohere</th></tr></thead><tbody>' + summaryRows + '</tbody></table></div></div>'
    + '<div class="card"><div class="card-header"><span class="card-title">Per-Query Detail (Faithfulness &amp; Relevancy)</span></div>'
    + '<div style="overflow-x:auto"><table class="data-table"><thead><tr>'
    + '<th>#</th><th>Query</th>'
    + '<th style="color:var(--openai)">OAI Faith.</th><th style="color:var(--cohere)">Coh. Faith.</th>'
    + '<th style="color:var(--openai)">OAI Relev.</th><th style="color:var(--cohere)">Coh. Relev.</th>'
    + '</tr></thead><tbody>' + perQRows + '</tbody></table></div></div>'
    + '</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// COLLECTIONS
// ─────────────────────────────────────────────────────────────────────────────
async function loadCollections() {
  try {
    var data = await apiFetch('/api/collections');
    document.getElementById('collectionsOutput').innerHTML = renderCollections(data);
    attachClearHandlers();
  } catch (e) {
    document.getElementById('collectionsOutput').innerHTML = '<div class="error-box">' + esc(e.message) + '</div>';
  }
}

function renderCollections(data) {
  function renderColl(key, info, color, label) {
    var pts = info.points_count || 0, hasData = pts > 0;
    return '<div class="coll-box" style="margin-bottom:16px">'
      + '<div class="coll-box-header">'
      + '<div style="display:flex;align-items:center;gap:10px">'
      + '<span class="model-badge ' + key + '">' + label + '</span>'
      + '<span style="font-family:var(--condensed);font-weight:600;font-size:14px;color:' + color + '">' + info.name + '</span>'
      + '</div>'
      + '<div style="display:flex;align-items:center;gap:8px">'
      + '<div class="coll-status"><div class="coll-dot ' + (hasData ? 'green' : '') + '"></div>' + (hasData ? 'Active' : 'Empty') + '</div>'
      + '<button class="btn btn-danger" data-clear="' + key + '" style="padding:6px 12px;font-size:12px">Clear</button>'
      + '</div></div>'
      + '<div style="padding:14px 18px">'
      + '<div class="stat-row"><span class="stat-key">Points (chunks)</span><span class="stat-val">' + pts.toLocaleString() + '</span></div>'
      + '<div class="stat-row"><span class="stat-key">Indexed vectors</span><span class="stat-val">' + (info.vectors_count || 0).toLocaleString() + '</span></div>'
      + '<div class="stat-row"><span class="stat-key">Status</span><span class="stat-val">' + (info.status || 'unknown') + '</span></div>'
      + '<div class="stat-row"><span class="stat-key">Error</span><span class="stat-val" style="color:var(--danger)">' + (info.error || '&mdash;') + '</span></div>'
      + '</div></div>';
  }
  return '<div style="max-width:640px">'
    + renderColl('openai', data.openai, 'var(--openai)', 'OpenAI')
    + renderColl('cohere', data.cohere, 'var(--cohere)', 'Cohere')
    + '<div style="display:flex;gap:10px">'
    + '<button class="btn btn-danger" data-clear="both">Clear All Collections</button>'
    + '<button class="btn btn-secondary" id="refreshCollBtn">Refresh</button>'
    + '</div></div>';
}

function attachClearHandlers() {
  document.querySelectorAll('[data-clear]').forEach(function(btn) {
    btn.addEventListener('click', async function() {
      var model = btn.dataset.clear;
      if (!confirm('Clear ' + model + ' collection(s)? This cannot be undone.')) return;
      try {
        await apiFetch('/api/collections/' + model, { method: 'DELETE' });
        showToast('Cleared ' + model + ' collection(s)', 'success');
        await loadCollections();
      } catch (e) {
        showToast(e.message, 'error');
      }
    });
  });
  var r = document.getElementById('refreshCollBtn');
  if (r) r.addEventListener('click', loadCollections);
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000);
</script>
</body>
</html>
"""