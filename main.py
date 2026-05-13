import ssl
import certifi

def fixed_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = fixed_ssl_context

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from api.routes import router
import uvicorn

app = FastAPI(title="Enterprise RAG Platform", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    return HTMLResponse(content=HTML_UI)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")


HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Enterprise RAG Platform</title>
<style>
:root {
  --bg:#f2f4f9;--surface:#fff;--surface-2:#f7f9fc;--surface-3:#eef1f8;
  --border:#dde3ef;--border-2:#e8ecf5;
  --text:#0c1526;--text-2:#3d4f6b;--text-3:#8796b2;
  --blue:#1e3fae;--blue-2:#2551d6;--blue-bg:#eef2ff;--blue-border:#c0cfff;
  --oa:#047857;--oa-bg:#ecfdf5;--oa-border:#6ee7b7;--oa-text:#065f46;
  --co:#b45309;--co-bg:#fffbeb;--co-border:#fcd34d;--co-text:#92400e;
  --ragas:#5b21b6;--ragas-bg:#f5f3ff;--ragas-border:#c4b5fd;
  --ok:#059669;--ok-bg:#f0fdf4;
  --err:#dc2626;--err-bg:#fef2f2;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --shadow-md:0 4px 14px rgba(0,0,0,.08),0 2px 4px rgba(0,0,0,.04);
  --r:8px;--r-sm:5px;
  --font:system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono','SF Mono',ui-monospace,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font);background:var(--bg);color:var(--text);font-size:13.5px;line-height:1.6;min-height:100vh}

/* ── Header ── */
header{background:var(--surface);border-bottom:1px solid var(--border);height:54px;padding:0 24px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:200;box-shadow:var(--shadow)}
.brand{font-size:15px;font-weight:700;letter-spacing:-.2px;color:var(--text)}
.brand-sep{width:1px;height:18px;background:var(--border)}
.brand-meta{font-family:var(--mono);font-size:10.5px;color:var(--text-3)}
.hdr-space{flex:1}
.status-wrap{display:flex;align-items:center;gap:7px}
.status-dot{width:7px;height:7px;border-radius:50%;background:#cbd5e1;transition:background .3s}
.status-dot.ok{background:var(--ok)}.status-dot.err{background:var(--err)}
.status-lbl{font-family:var(--mono);font-size:11px;color:var(--text-3)}

/* ── Tabs ── */
.tab-nav{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;display:flex;gap:2px}
.tab-btn{padding:13px 20px;border:none;background:none;font-size:12.5px;font-weight:600;letter-spacing:.3px;color:var(--text-3);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s,border-color .15s;white-space:nowrap}
.tab-btn:hover{color:var(--text-2)}
.tab-btn.active{color:var(--blue-2);border-bottom-color:var(--blue-2)}

/* ── Layout ── */
main{max-width:1360px;margin:0 auto;padding:24px}
.tab-panel{display:none}.tab-panel.active{display:block}
.two-col{display:grid;grid-template-columns:360px 1fr;gap:20px;align-items:start}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}

/* ── Card ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.card-hdr{padding:13px 18px;border-bottom:1px solid var(--border-2);display:flex;align-items:center;justify-content:space-between;gap:10px;background:var(--surface)}
.card-title{font-size:11px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:var(--text-3)}
.card-body{padding:18px}
.card-body.pt-0{padding-top:10px}

/* ── Form ── */
.field{margin-bottom:14px}.field:last-child{margin-bottom:0}
label{display:block;font-size:11px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;color:var(--text-3);margin-bottom:6px}
input[type=text],input[type=number],select,textarea{width:100%;padding:8px 11px;border:1px solid var(--border);border-radius:var(--r-sm);font-family:var(--font);font-size:13px;color:var(--text);background:var(--surface);outline:none;transition:border-color .15s,box-shadow .15s;-webkit-appearance:none}
input:focus,select:focus,textarea:focus{border-color:var(--blue-2);box-shadow:0 0 0 3px rgba(37,81,214,.1)}
textarea{resize:vertical;min-height:70px}
select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238796b2' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;padding-right:30px;cursor:pointer}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}

/* ── Buttons ── */
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:var(--r-sm);border:1px solid transparent;font-size:12px;font-weight:700;letter-spacing:.4px;cursor:pointer;transition:all .15s;white-space:nowrap}
.btn-primary{background:var(--blue-2);color:#fff;border-color:var(--blue-2)}
.btn-primary:hover:not(:disabled){background:var(--blue);border-color:var(--blue)}
.btn-secondary{background:var(--surface);color:var(--text-2);border-color:var(--border)}
.btn-secondary:hover:not(:disabled){background:var(--surface-2)}
.btn-ragas{background:var(--ragas);color:#fff;border-color:var(--ragas)}
.btn-ragas:hover:not(:disabled){background:#4c1d95}
.btn-danger{background:var(--err-bg);color:var(--err);border-color:#fca5a5}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn-full{width:100%;justify-content:center}
.btn.spin::after{content:'';width:11px;height:11px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:_spin .6s linear infinite;display:inline-block}
@keyframes _spin{to{transform:rotate(360deg)}}

/* ── Drop zone ── */
.drop-zone{border:2px dashed var(--border);border-radius:var(--r);padding:28px 20px;text-align:center;cursor:pointer;background:var(--surface-2);position:relative;transition:all .2s}
.drop-zone:hover,.drop-zone.over{border-color:var(--blue-2);background:var(--blue-bg)}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-title{font-size:13.5px;color:var(--text-2);margin-bottom:4px;font-weight:500}
.drop-sub{font-family:var(--mono);font-size:11px;color:var(--text-3)}
.file-chip{display:flex;align-items:center;gap:8px;padding:6px 12px;background:var(--blue-bg);border:1px solid var(--blue-border);border-radius:var(--r-sm);margin-bottom:6px;font-size:12px;color:var(--blue-2)}
.file-chip .fsz{font-family:var(--mono);font-size:11px;color:var(--text-3);margin-left:4px}
.file-chip .rm{margin-left:auto;cursor:pointer;background:none;border:none;color:var(--text-3);font-size:14px;line-height:1}

/* ── Progress ── */
.prog-wrap{height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin:8px 0}
.prog-bar{height:100%;background:linear-gradient(90deg,var(--blue-2),#60a5fa);border-radius:3px;transition:width .35s ease;width:0%}
.prog-msg{font-family:var(--mono);font-size:11px;color:var(--text-2)}

/* ── Badges / pills ── */
.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:99px;font-family:var(--mono);font-size:11px;font-weight:600}
.badge::before{content:'';width:5px;height:5px;border-radius:50%;background:currentColor}
.badge-oa{background:var(--oa-bg);color:var(--oa);border:1px solid var(--oa-border)}
.badge-co{background:var(--co-bg);color:var(--co);border:1px solid var(--co-border)}
.badge-blue{background:var(--blue-bg);color:var(--blue-2);border:1px solid var(--blue-border)}
.badge-ragas{background:var(--ragas-bg);color:var(--ragas);border:1px solid var(--ragas-border)}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:99px;background:var(--surface-3);border:1px solid var(--border-2);color:var(--text-2)}
.pill.fast{background:var(--ok-bg);border-color:#86efac;color:var(--ok)}
.pill.slow{background:#fff7ed;border-color:#fed7aa;color:#c2410c}

/* ── Score bar ── */
.sbar-wrap{display:flex;align-items:center;gap:7px}
.sbar{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.sbar-fill{height:100%;border-radius:2px;transition:width .4s ease}
.sval{font-family:var(--mono);font-size:12px;color:var(--text-2);min-width:44px;text-align:right}

/* ── Stat row ── */
.stat-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border-2);font-size:13px}
.stat-row:last-child{border-bottom:none}
.skey{color:var(--text-2)}.sval2{font-family:var(--mono);font-size:12px;color:var(--text);font-weight:600}

/* ── Chunk cards ── */
.chunk-card{border:1px solid var(--border);border-radius:var(--r-sm);overflow:hidden;margin-bottom:8px;background:var(--surface)}
.chunk-card:last-child{margin-bottom:0}
.chunk-card-hdr{padding:7px 12px;background:var(--surface-2);border-bottom:1px solid var(--border-2);display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.chunk-rank{font-family:var(--mono);font-size:10.5px;color:var(--text-3);background:var(--border-2);padding:1px 6px;border-radius:3px}
.chunk-source{font-family:var(--mono);font-size:11px;color:var(--text-2)}
.chunk-body{padding:10px 12px;font-size:12.5px;color:var(--text-2);line-height:1.6;max-height:110px;overflow:hidden}
.chunk-body.exp{max-height:none}
.chunk-more{font-size:11px;color:var(--blue-2);cursor:pointer;padding:4px 12px;display:block;border-top:1px solid var(--border-2);background:var(--surface-2);text-align:center;font-weight:700;letter-spacing:.3px}

/* ── Chat layout ── */
.chat-wrap{display:flex;flex-direction:column;height:calc(100vh - 152px);min-height:560px}
.chat-toolbar{background:var(--surface);border:1px solid var(--border);border-radius:var(--r) var(--r) 0 0;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tbar-label{font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--text-3)}
.chat-toolbar select,.chat-toolbar input[type=number]{width:auto;padding:5px 10px;font-size:12px}
.chat-toolbar input[type=number]{width:58px}
.tbar-space{flex:1}
.chat-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px;background:var(--surface);border-left:1px solid var(--border);border-right:1px solid var(--border)}
.chat-input-bar{background:var(--surface);border:1px solid var(--border);border-top:0;border-radius:0 0 var(--r) var(--r);padding:10px 12px;display:flex;gap:8px;align-items:flex-end}
.chat-input-bar textarea{flex:1;min-height:40px;max-height:110px;resize:none;font-size:13px;padding:8px 11px}

/* ── Chat messages ── */
.msg{display:flex;gap:9px;max-width:92%}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.msg.asst{align-self:flex-start}
.avatar{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10.5px;font-weight:700;flex-shrink:0}
.msg.user .avatar{background:var(--blue-2);color:#fff}
.msg.asst .avatar{background:var(--surface-3);border:1px solid var(--border);color:var(--text-2)}
.bubble{padding:10px 14px;border-radius:10px;font-size:13.5px;line-height:1.7;word-break:break-word}
.msg.user .bubble{background:var(--blue-2);color:#fff;border-radius:10px 10px 3px 10px}
.msg.asst .bubble{background:var(--surface-2);border:1px solid var(--border-2);color:var(--text);border-radius:10px 10px 10px 3px}

/* ── Response card (assistant full) ── */
.resp-card{border:1px solid var(--border);border-radius:var(--r);overflow:hidden;background:var(--surface);margin-bottom:4px}
.resp-card-hdr{padding:10px 15px;background:var(--surface-2);border-bottom:1px solid var(--border-2);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.resp-card-body{padding:16px;font-size:13.5px;line-height:1.8;color:var(--text)}
.resp-card-body p+p{margin-top:10px}
.stream-cur::after{content:'|';animation:_blink .7s step-end infinite;color:var(--blue-2)}
@keyframes _blink{0%,100%{opacity:1}50%{opacity:0}}
.typing{display:inline-flex;gap:4px;align-items:center;padding:4px 0}
.typing span{width:6px;height:6px;border-radius:50%;background:var(--text-3);animation:_bounce 1.1s ease-in-out infinite}
.typing span:nth-child(2){animation-delay:.18s}.typing span:nth-child(3){animation-delay:.36s}
@keyframes _bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}

/* ── Citations (Perplexity style) ── */
.cite-sup{display:inline-flex;align-items:center;justify-content:center;min-width:16px;height:16px;padding:0 4px;border-radius:4px;background:var(--blue-bg);border:1px solid var(--blue-border);color:var(--blue-2);font-family:var(--mono);font-size:10px;font-weight:700;cursor:pointer;vertical-align:super;transition:all .15s;margin:0 1px;line-height:1}
.cite-sup:hover,.cite-sup.lit{background:var(--blue-2);color:#fff;border-color:var(--blue-2)}
.sources-panel{border-top:1px solid var(--border-2);padding:12px 15px 14px;background:var(--surface-2)}
.sources-title{font-size:10.5px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;color:var(--text-3);margin-bottom:10px}
.source-item{display:flex;gap:11px;padding:9px 13px;border:1px solid var(--border);border-radius:var(--r-sm);margin-bottom:7px;background:var(--surface);align-items:flex-start;cursor:pointer;transition:border-color .15s,background .15s}
.source-item:last-child{margin-bottom:0}
.source-item:hover,.source-item.lit{border-color:var(--blue-2);background:var(--blue-bg)}
.src-num{font-family:var(--mono);font-size:11px;font-weight:700;width:22px;height:22px;border-radius:50%;background:var(--blue-bg);color:var(--blue-2);border:1px solid var(--blue-border);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.src-info{flex:1;min-width:0}
.src-meta{display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap}
.src-file{font-family:var(--mono);font-size:11.5px;color:var(--text);font-weight:600}
.src-loc{font-family:var(--mono);font-size:10.5px;color:var(--text-3)}
.src-score{font-family:var(--mono);font-size:10.5px;padding:1px 6px;border-radius:3px;font-weight:600}
.src-preview{font-size:12px;color:var(--text-2);line-height:1.5;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}

/* ── Both-mode comparison ── */
.compare-split{display:grid;grid-template-columns:1fr 1fr;gap:16px;width:100%;margin-top:12px}
.compare-col{min-width:0}
.compare-col-hdr{font-size:12px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.compare-col-hdr.oa{color:var(--oa-text)}
.compare-col-hdr.co{color:var(--co-text)}

/* ── DeepEval metrics ── */
.ragas-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
.ragas-card{padding:16px;border:1px solid var(--ragas-border);border-radius:var(--r-sm);background:var(--ragas-bg);text-align:center}
.ragas-val{font-family:var(--mono);font-size:26px;font-weight:600;color:var(--ragas);line-height:1.2}
.ragas-lbl{font-size:10.5px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;color:var(--ragas);opacity:.65;margin-top:4px}
.ragas-bar{height:5px;background:var(--ragas-border);border-radius:3px;overflow:hidden;margin-top:9px}
.ragas-fill{height:100%;background:var(--ragas);border-radius:3px;transition:width .5s ease}
.ragas-na{opacity:.4}
.ragas-na .ragas-val{font-size:18px;color:var(--text-3)}

/* ── Eval sub-tabs ── */
.subtabs{display:flex;gap:2px;background:var(--surface-3);border:1px solid var(--border);border-radius:var(--r);padding:4px;margin-bottom:20px}
.subtab{flex:1;padding:7px 14px;border:none;background:none;font-size:11.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:var(--text-3);cursor:pointer;border-radius:5px;transition:all .15s;text-align:center}
.subtab.active{background:var(--surface);color:var(--ragas);box-shadow:var(--shadow);border:1px solid var(--ragas-border)}
.subpanel{display:none}.subpanel.active{display:block}

/* ── Data table ── */
.data-tbl{width:100%;border-collapse:collapse;font-size:12px}
.data-tbl th{font-size:10.5px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--text-3);padding:7px 12px;text-align:left;border-bottom:1px solid var(--border);background:var(--surface-2)}
.data-tbl td{padding:7px 12px;border-bottom:1px solid var(--border-2);vertical-align:top}
.data-tbl tr:last-child td{border-bottom:none}
.data-tbl tr:hover td{background:var(--surface-2)}
.mono{font-family:var(--mono);font-size:11.5px}

/* ── Misc ── */
.empty{text-align:center;padding:36px 20px;color:var(--text-3);font-size:13px}
.err-box{padding:11px 15px;background:var(--err-bg);border:1px solid #fca5a5;border-radius:var(--r-sm);color:var(--err);font-size:13px}
.info-box{padding:10px 14px;background:var(--blue-bg);border:1px solid var(--blue-border);border-radius:var(--r-sm);font-size:12.5px;color:var(--blue);line-height:1.6}
#toast{position:fixed;bottom:20px;right:20px;padding:10px 16px;border-radius:var(--r);font-size:12.5px;font-weight:600;box-shadow:var(--shadow-md);opacity:0;transform:translateY(6px);transition:all .22s;pointer-events:none;z-index:999;max-width:320px}
#toast.show{opacity:1;transform:translateY(0)}
#toast.ok{background:var(--ok-bg);color:#065f46;border:1px solid #86efac}
#toast.err{background:var(--err-bg);color:#991b1b;border:1px solid #fca5a5}
#toast.info{background:var(--blue-bg);color:var(--blue);border:1px solid var(--blue-border)}

@media(max-width:900px){.two-col{grid-template-columns:1fr}.compare-split{grid-template-columns:1fr}.ragas-grid{grid-template-columns:1fr 1fr 1fr}}
@media(max-width:600px){main{padding:14px}.row2,.row3{grid-template-columns:1fr}}

/* Markdown rendered content */
.md-body { line-height: 1.75; }
.md-body h1, .md-body h2, .md-body h3 {
  font-weight: 700; margin: 14px 0 6px; color: var(--text);
}
.md-body h2 { font-size: 14.5px; }
.md-body h3 { font-size: 13.5px; }
.md-body p { margin: 6px 0; }
.md-body ul, .md-body ol { padding-left: 20px; margin: 6px 0; }
.md-body li { margin: 3px 0; }
.md-body strong { font-weight: 700; color: var(--text); }
.md-body em { font-style: italic; }
.md-body code {
  font-family: var(--mono); font-size: 11.5px;
  background: var(--surface-3); padding: 1px 5px;
  border-radius: 3px; border: 1px solid var(--border-2);
}
.md-body pre {
  background: var(--surface-3); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: 12px 14px;
  overflow-x: auto; margin: 8px 0;
}
.md-body pre code { background: none; border: none; padding: 0; }
.md-body table {
  width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0;
}
.md-body th {
  background: var(--surface-2); font-weight: 700; font-size: 11px;
  letter-spacing: .4px; text-transform: uppercase; color: var(--text-3);
  padding: 6px 10px; border: 1px solid var(--border);
}
.md-body td { padding: 6px 10px; border: 1px solid var(--border-2); }
.md-body tr:hover td { background: var(--surface-2); }
.md-body blockquote {
  border-left: 3px solid var(--blue-2); padding: 6px 12px;
  margin: 8px 0; background: var(--blue-bg); color: var(--text-2);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
}
</style>

<!-- marked.js for Markdown rendering in chat responses -->
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<script>
  marked.setOptions({ breaks: true, gfm: true });
</script>
</head>
<body>

<header>
  <span class="brand">Enterprise RAG Platform</span>
  <div class="brand-sep"></div>
  <span class="brand-meta">text-embedding-3-large &middot; embed-v-4-0 &middot; Pinecone &middot; gpt-5.4-pro</span>
  <div class="hdr-space"></div>
  <div class="status-wrap">
    <span class="status-dot" id="sDot"></span>
    <span class="status-lbl" id="sLbl">Checking...</span>
  </div>
</header>

<nav class="tab-nav">
  <button class="tab-btn active" data-tab="ingest">Data Ingestion</button>
  <button class="tab-btn" data-tab="chat">Chat Comparison</button>
  <button class="tab-btn" data-tab="eval">DeepEval Evaluation</button>
</nav>

<main>

<!-- =====================================================================
     INGEST
     ===================================================================== -->
<div class="tab-panel active" id="tab-ingest">
  <div class="two-col">
    <div>
      <div class="card" style="margin-bottom:16px">
        <div class="card-hdr"><span class="card-title">Upload Documents</span></div>
        <div class="card-body">
          <div class="drop-zone" id="dropZone">
            <input type="file" id="fileInput" accept=".pdf,.csv" multiple>
            <div class="drop-title">Drop files here or click to browse</div>
            <div class="drop-sub">Supported: PDF, CSV &nbsp;&middot;&nbsp; English &amp; Spanish</div>
          </div>
          <div id="fileList" style="display:none;margin-top:10px"></div>
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
        <div class="card-hdr"><span class="card-title">Pipeline Summary</span></div>
        <div class="card-body">
          <div class="stat-row"><span class="skey">PDF pipeline</span><span class="sval2">OCR &rarr; semantic chunk &rarr; embed &rarr; Pinecone</span></div>
          <div class="stat-row"><span class="skey">CSV pipeline</span><span class="sval2">Schema detect &rarr; row chunk &rarr; embed &rarr; metadata filter</span></div>
          <div class="stat-row"><span class="skey">Chunker</span><span class="sval2">all-MiniLM-L6-v2 semantic breakpoints</span></div>
          <div class="stat-row"><span class="skey">Metadata</span><span class="sval2">Auto-extracted (numeric, date, categorical)</span></div>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="card-hdr">
          <span class="card-title">Ingestion Results</span>
          <span id="ingestStatus"></span>
        </div>
        <div class="card-body" id="ingestResults">
          <div class="empty">Upload documents to begin</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- =====================================================================
     CHAT COMPARISON
     ===================================================================== -->
<div class="tab-panel" id="tab-chat">
  <div class="chat-wrap">
    <div class="chat-toolbar">
      <span class="tbar-label">Embedding Model</span>
      <select id="chatModel">
        <option value="openai">OpenAI</option>
        <option value="cohere">Cohere</option>
        <option value="both">Both (Side-by-Side)</option>
      </select>
      <span class="tbar-label" style="margin-left:8px">Top-K</span>
      <input type="number" id="chatTopK" value="5" min="1" max="20">
      <div class="tbar-space"></div>
      <button class="btn btn-secondary" id="chatClear" style="padding:6px 12px">Clear Chat</button>
    </div>

    <div class="chat-messages" id="chatMsgs">
      <div class="msg asst">
        <div class="avatar">AI</div>
        <div>
          <div class="bubble">Ready. Ask questions about your ingested documents. Select a model above, or choose "Both" for a side-by-side comparison.</div>
        </div>
      </div>
    </div>

    <div class="chat-input-bar">
      <textarea id="chatInput" rows="1" placeholder="Ask a question... (Enter to send, Shift+Enter for new line)"></textarea>
      <button class="btn btn-primary" id="chatSend" style="height:40px;flex-shrink:0">Send</button>
    </div>
  </div>
</div>

<!-- =====================================================================
     DEEPEVAL EVALUATION
     ===================================================================== -->
<div class="tab-panel" id="tab-eval">

  <div class="info-box" style="margin-bottom:20px">
    <strong>DeepEval Evaluation</strong> &mdash;
    <strong>Faithfulness</strong>, <strong>Answer Relevancy</strong>, and <strong>Hallucination</strong> are always computed.
    Provide a ground-truth answer to also compute <strong>Context Precision</strong> and <strong>Context Recall</strong>.
    Set model to <strong>Both</strong> for a direct OpenAI vs Cohere comparison.
  </div>

  <div class="subtabs">
    <button class="subtab active" data-sub="single">Single Query</button>
    <button class="subtab" data-sub="bench">Benchmark</button>
  </div>

  <!-- Single Query -->
  <div class="subpanel active" id="sub-single">
    <div class="two-col">
      <div class="card">
        <div class="card-hdr">
          <span class="card-title">Query Configuration</span>
          <span class="badge badge-ragas">DeepEval</span>
        </div>
        <div class="card-body">
          <div class="field">
            <label>Query</label>
            <textarea id="sQuery" rows="3" placeholder="Enter evaluation query..."></textarea>
          </div>
          <div class="row2">
            <div class="field">
              <label>Embedding Model</label>
              <select id="sModel">
                <option value="openai">OpenAI</option>
                <option value="cohere">Cohere</option>
                <option value="both">Both (Side-by-Side)</option>
              </select>
            </div>
            <div class="field">
              <label>Top-K</label>
              <input type="number" id="sTopK" value="5" min="1" max="20">
            </div>
          </div>
          <div class="field">
            <label>Ground Truth <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3)">(optional)</span></label>
            <textarea id="sGT" rows="3" placeholder="Reference answer — enables Context Precision &amp; Context Recall"></textarea>
          </div>
          <div class="field">
            <label>Pre-generated Answer <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3)">(optional — leave blank to auto-generate)</span></label>
            <textarea id="sAnswer" rows="2" placeholder="Paste an existing answer, or leave blank to run the full RAG pipeline..."></textarea>
          </div>
          <button class="btn btn-ragas btn-full" id="sRunBtn">Run DeepEval Evaluation</button>
        </div>
      </div>
      <div id="sEvalOut">
        <div class="empty card" style="padding:40px">Configure and run an evaluation</div>
      </div>
    </div>
  </div>

  <!-- Benchmark -->
  <div class="subpanel" id="sub-bench">
    <div class="two-col">
      <div class="card">
        <div class="card-hdr">
          <span class="card-title">Benchmark Configuration</span>
          <span class="badge badge-ragas">DeepEval</span>
        </div>
        <div class="card-body">
          <div class="row2">
            <div class="field">
              <label>Embedding Model</label>
              <select id="bModel">
                <option value="openai">OpenAI</option>
                <option value="cohere">Cohere</option>
                <option value="both">Both</option>
              </select>
            </div>
            <div class="field">
              <label>Top-K</label>
              <input type="number" id="bK" value="5" min="1" max="20">
            </div>
          </div>
          <div style="margin-bottom:10px">
            <button class="btn btn-secondary btn-full" id="bLoadSample">Load Sample Queries</button>
          </div>
          <div class="field">
            <label>Queries JSON</label>
            <textarea id="bQueries" rows="12" placeholder='[{"query":"...","ground_truth":"..."}]' style="font-family:var(--mono);font-size:11px"></textarea>
          </div>
          <button class="btn btn-ragas btn-full" id="bRunBtn">Run Benchmark</button>
        </div>
      </div>
      <div id="bEvalOut">
        <div class="empty card" style="padding:40px">Run benchmark to see DeepEval metrics</div>
      </div>
    </div>
  </div>

</div><!-- /tab-eval -->

</main>
<div id="toast"></div>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// Core utilities
// ─────────────────────────────────────────────────────────────────────────────
function esc(s){
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function apiFetch(path, opts){
  return fetch(path, opts||{}).then(function(r){
    if(!r.ok) return r.json().catch(function(){return{detail:r.statusText};})
      .then(function(e){throw new Error(e.detail||JSON.stringify(e));});
    return r.json();
  });
}

function toast(msg, type){
  var t=document.getElementById('toast');
  t.textContent=msg; t.className='show '+(type||'info');
  clearTimeout(t._tid);
  t._tid=setTimeout(function(){t.className='';},3200);
}

function spin(btn, on){
  btn.disabled=on;
  on ? btn.classList.add('spin') : btn.classList.remove('spin');
}

function latCls(ms){ return ms<350?'fast':ms>1400?'slow':''; }

function scoreColor(s){ return s>=0.8?'#059669':s>=0.55?'#d97706':'#94a3b8'; }
function ragasColor(s){ return s>=0.75?'var(--ok)':s>=0.5?'#d97706':'var(--err)'; }

// Render [N] cite marks in answer text
function renderCites(text, citations, uid){
  var html = esc(text).replace(/\n/g,'<br>');
  html = html.replace(/\[(\d+)\]/g, function(m,n){
    var idx=parseInt(n)-1;
    if(citations && citations[idx]){
      return '<sup class="cite-sup" data-idx="'+idx+'" data-uid="'+(uid||'')+'" title="'+esc(citations[idx].source)+'">['+n+']</sup>';
    }
    return m;
  });
  return html;
}

// Render answer with inline citation superscripts, markdown-aware
function renderCitesMarkdown(text, citations, uid){
  // Replace [N] citation markers with superscript badges before markdown parsing
  var withBadges = text.replace(/\[(\d+)\]/g, function(m, n){
    var idx = parseInt(n) - 1;
    if(!citations || !citations[idx]) return m;
    return '<sup class="cite-sup" data-idx="'+idx+'" data-uid="'+(uid||'')+'" title="'+esc(citations[idx].source)+'">'+n+'</sup>';
  });
  // Parse markdown (citations are already HTML, preserve them)
  var renderer = new marked.Renderer();
  return marked.parse(withBadges, { renderer: renderer });
}

function bindCiteHandlers(){
  document.querySelectorAll('.cite-sup').forEach(function(el){
    el.addEventListener('click',function(){
      var idx=parseInt(el.dataset.idx);
      var uid=el.dataset.uid||'';
      // clear all
      document.querySelectorAll('.cite-sup').forEach(function(e){e.classList.remove('lit');});
      document.querySelectorAll('.source-item').forEach(function(e){e.classList.remove('lit');});
      el.classList.add('lit');
      // find matching source item
      var si=document.getElementById('src-'+uid+'-'+(idx+1));
      if(!si){
        var all=document.querySelectorAll('.source-item');
        for(var i=0;i<all.length;i++){if(all[i].id.endsWith('-'+(idx+1))){si=all[i];break;}}
      }
      if(si){si.classList.add('lit');si.scrollIntoView({behavior:'smooth',block:'nearest'});}
    });
  });
}

function bindChunkExpand(){
  document.querySelectorAll('.chunk-more').forEach(function(btn){
    btn.addEventListener('click',function(){
      var t=document.getElementById(btn.dataset.t);
      var exp=t.classList.toggle('exp');
      btn.textContent=exp?'Show less':'Show more';
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Status check
// ─────────────────────────────────────────────────────────────────────────────
function checkStatus(){
  apiFetch('/api/collections').then(function(){
    document.getElementById('sDot').className='status-dot ok';
    document.getElementById('sLbl').textContent='Connected';
  }).catch(function(){
    document.getElementById('sDot').className='status-dot err';
    document.getElementById('sLbl').textContent='Offline';
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab navigation
// ─────────────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(function(btn){
  btn.addEventListener('click',function(){
    document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
    document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active');});
    btn.classList.add('active');
    var p=document.getElementById('tab-'+btn.dataset.tab);
    if(p) p.classList.add('active');
  });
});

document.querySelectorAll('.subtab').forEach(function(btn){
  btn.addEventListener('click',function(){
    document.querySelectorAll('.subtab').forEach(function(b){b.classList.remove('active');});
    document.querySelectorAll('.subpanel').forEach(function(p){p.classList.remove('active');});
    btn.classList.add('active');
    var p=document.getElementById('sub-'+btn.dataset.sub);
    if(p) p.classList.add('active');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// INGEST
// ─────────────────────────────────────────────────────────────────────────────
var dropZone=document.getElementById('dropZone');
var fileInput=document.getElementById('fileInput');
var ingestBtn=document.getElementById('ingestBtn');
var selectedFiles=[];

function fmtBytes(b){
  if(b<1024) return b+' B';
  if(b<1048576) return (b/1024).toFixed(1)+' KB';
  return (b/1048576).toFixed(1)+' MB';
}

function renderFileList(){
  var d=document.getElementById('fileList');
  if(!selectedFiles.length){d.style.display='none';ingestBtn.disabled=true;return;}
  d.style.display='block';
  ingestBtn.disabled=false;
  d.innerHTML=selectedFiles.map(function(f,i){
    return '<div class="file-chip">'
      +'<span>'+esc(f.name)+'</span><span class="fsz">'+fmtBytes(f.size)+'</span>'
      +'<button class="rm" data-i="'+i+'">&times;</button>'
      +'</div>';
  }).join('');
  d.querySelectorAll('.rm').forEach(function(b){
    b.addEventListener('click',function(){selectedFiles.splice(parseInt(b.dataset.i),1);renderFileList();});
  });
}

function addFiles(fl){
  Array.from(fl).forEach(function(f){
    var ext=f.name.split('.').pop().toLowerCase();
    if(!['pdf','csv'].includes(ext)){toast('Unsupported: '+f.name,'err');return;}
    if(!selectedFiles.some(function(s){return s.name===f.name&&s.size===f.size;}))
      selectedFiles.push(f);
  });
  renderFileList();
}

fileInput.addEventListener('change',function(){if(fileInput.files.length)addFiles(fileInput.files);fileInput.value='';});
dropZone.addEventListener('dragover',function(e){e.preventDefault();dropZone.classList.add('over');});
dropZone.addEventListener('dragleave',function(){dropZone.classList.remove('over');});
dropZone.addEventListener('drop',function(e){e.preventDefault();dropZone.classList.remove('over');if(e.dataTransfer.files.length)addFiles(e.dataTransfer.files);});

ingestBtn.addEventListener('click',async function(){
  if(!selectedFiles.length) return;
  spin(ingestBtn,true);
  var total=selectedFiles.length, results=[], errors=[];

  for(var i=0;i<selectedFiles.length;i++){
    var f=selectedFiles[i];
    var pid='pb'+i;
    document.getElementById('ingestResults').innerHTML=
      '<div style="padding:16px">'
      +'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
      +'<span style="font-family:var(--mono);font-size:12px">'+esc(f.name)+'</span>'
      +'<span style="font-family:var(--mono);font-size:11px;color:var(--text-3)">'+( i+1)+' / '+total+'</span>'
      +'</div>'
      +'<div class="prog-wrap"><div class="prog-bar" id="'+pid+'"></div></div>'
      +'<div class="prog-msg" id="msg'+pid+'">Uploading...</div>'
      +'</div>';

    try{
      var fd=new FormData();
      fd.append('file',f);
      fd.append('embedding_model',document.getElementById('ingestModel').value);

      var result=await new Promise(function(res,rej){
        fetch('/api/ingest',{method:'POST',body:fd}).then(function(resp){
          if(!resp.ok){resp.json().catch(function(){return{detail:resp.statusText};}).then(function(e){rej(new Error(e.detail||JSON.stringify(e)));});return;}
          var reader=resp.body.getReader(),dec=new TextDecoder(),buf='';
          function pump(){
            return reader.read().then(function(ch){
              if(ch.done) return;
              buf+=dec.decode(ch.value,{stream:true});
              var lines=buf.split('\n');buf=lines.pop();
              for(var j=0;j<lines.length;j++){
                if(!lines[j].startsWith('data: ')) continue;
                try{
                  var ev=JSON.parse(lines[j].slice(6));
                  var bar=document.getElementById(pid),msg=document.getElementById('msg'+pid);
                  if(ev.type==='progress'){if(bar)bar.style.width=ev.percent+'%';if(msg)msg.textContent=ev.message;}
                  else if(ev.type==='done'){if(bar)bar.style.width='100%';res(ev);}
                  else if(ev.type==='error'){rej(new Error(ev.message));}
                }catch(pe){}
              }
              return pump();
            }).catch(rej);
          }
          pump();
        }).catch(rej);
      });
      results.push(result);
    }catch(e){errors.push({name:f.name,error:e.message});}
  }

  document.getElementById('ingestResults').innerHTML=renderIngestResults(results,errors);
  if(results.length) toast(results.length+' file(s) indexed successfully','ok');
  if(errors.length)  toast(errors.length+' file(s) failed','err');
  selectedFiles=[];renderFileList();
  spin(ingestBtn,false);
});

function renderOneResult(d){
  var rows=[
    ['File type', d.source_type.toUpperCase()],
    ['Raw documents', d.raw_documents],
    ['Semantic chunks', d.chunks],
  ];
  if(d.indexed.openai!==undefined) rows.push(['OpenAI vectors indexed', d.indexed.openai]);
  if(d.indexed.cohere!==undefined) rows.push(['Cohere vectors indexed', d.indexed.cohere]);
  return '<div style="border:1px solid var(--border);border-radius:var(--r-sm);margin-bottom:10px;overflow:hidden">'
    +'<div style="padding:7px 13px;background:var(--surface-2);border-bottom:1px solid var(--border-2);display:flex;align-items:center;gap:8px">'
    +'<span style="font-family:var(--mono);font-size:12px;color:var(--text)">'+esc(d.filename)+'</span>'
    +'<span style="margin-left:auto;background:var(--ok-bg);color:var(--ok);border:1px solid #86efac;border-radius:99px;font-family:var(--mono);font-size:10.5px;padding:1px 8px;font-weight:700">Complete</span>'
    +'</div>'
    +'<div style="padding:4px 13px">'
    +rows.map(function(r){return '<div class="stat-row"><span class="skey">'+r[0]+'</span><span class="sval2">'+r[1]+'</span></div>';}).join('')
    +'</div></div>';
}

function renderIngestResults(results,errors){
  var total=results.length+errors.length;
  var summary=total>1
    ?'<div style="margin-bottom:12px;padding:9px 13px;background:var(--surface-3);border:1px solid var(--border);border-radius:var(--r-sm);font-size:13px;color:var(--text-2)">'
      +results.length+' of '+total+' files indexed</div>'
    :'';
  return '<div style="padding-top:4px">'
    +summary
    +results.map(renderOneResult).join('')
    +errors.map(function(e){
      return '<div style="border:1px solid #fca5a5;border-radius:var(--r-sm);margin-bottom:8px;padding:9px 13px;background:var(--err-bg);font-size:12px;color:var(--err)">'
        +esc(e.name)+': '+esc(e.error)+'</div>';
    }).join('')
    +'</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// Source / citation panel renderer
// ─────────────────────────────────────────────────────────────────────────────
function renderSources(citations, uid){
  if(!citations||!citations.length) return '';
  var items=citations.map(function(c){
    var scoreW=Math.round(c.score*100);
    var scoreClr=scoreColor(c.score);
    return '<div class="source-item" id="src-'+uid+'-'+c.index+'">'
      +'<div class="src-num">'+c.index+'</div>'
      +'<div class="src-info">'
      +'<div class="src-meta">'
      +'<span class="src-file">'+esc(c.source)+'</span>'
      +'<span class="src-loc">'+esc(c.location)+'</span>'
      +'<span class="src-score" style="color:'+scoreClr+';background:'+scoreClr+'18">'+c.score.toFixed(4)+'</span>'
      +'<div class="sbar-wrap" style="max-width:80px;flex:1"><div class="sbar"><div class="sbar-fill" style="width:'+scoreW+'%;background:'+scoreClr+'"></div></div></div>'
      +'</div>'
      +'<div class="src-preview">'+esc(c.preview)+'</div>'
      +'</div></div>';
  }).join('');
  return '<div class="sources-panel">'
    +'<div class="sources-title">Sources ('+citations.length+')</div>'
    +items+'</div>';
}

// Build a full response card (used after streaming completes)
function buildRespCard(model, answer, citations, uid, retMs, genMs, tokensUsed){
  var badgeCls = model==='openai'?'badge-oa':'badge-co';
  var badgeLbl = model==='openai'?'OpenAI':'Cohere';
  var retCls=latCls(retMs), genCls=latCls(genMs);
  var hdr='<div class="resp-card-hdr">'
    +'<span class="badge '+badgeCls+'">'+badgeLbl+'</span>'
    +'<span class="pill '+retCls+'">Retrieval '+retMs+'ms</span>'
    +'<span class="pill '+genCls+'">Generation '+genMs+'ms</span>'
    +(tokensUsed?'<span class="pill">Tokens '+tokensUsed.total+'</span>':'')
    +'</div>';
  var body='<div class="resp-card-body">'+renderCites(answer, citations, uid)+'</div>';
  var srcs=renderSources(citations, uid);
  return '<div class="resp-card">'+hdr+body+srcs+'</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat — single model mode
// ─────────────────────────────────────────────────────────────────────────────
var chatHistory=[];

async function sendSingleModel(query, model, topK){
  var msgs=document.getElementById('chatMsgs');
  var uid=model+Date.now();

  // append user bubble
  var uEl=document.createElement('div');
  uEl.className='msg user';
  uEl.innerHTML='<div class="avatar">U</div><div><div class="bubble">'+esc(query)+'</div></div>';
  msgs.appendChild(uEl);
  msgs.scrollTop=msgs.scrollHeight;

  // streaming assistant placeholder
  var aEl=document.createElement('div');
  aEl.className='msg asst';
  var badgeCls=model==='openai'?'badge-oa':'badge-co';
  var badgeLbl=model==='openai'?'OpenAI':'Cohere';
  var cardId='card-'+uid;
  var bodyId='body-'+uid;
  aEl.innerHTML='<div class="avatar">AI</div>'
    +'<div style="flex:1;min-width:0">'
    +'<div class="resp-card" id="'+cardId+'">'
    +'<div class="resp-card-hdr" id="hdr-'+uid+'">'
    +'<span class="badge '+badgeCls+'">'+badgeLbl+'</span>'
    +'<span class="pill">Retrieving...</span>'
    +'</div>'
    +'<div class="resp-card-body stream-cur" id="'+bodyId+'">'
    +'<div class="typing"><span></span><span></span><span></span></div>'
    +'</div>'
    +'</div></div>';
  msgs.appendChild(aEl);
  msgs.scrollTop=msgs.scrollHeight;

  var bodyEl=document.getElementById(bodyId);
  var hdrEl=document.getElementById('hdr-'+uid);
  var cardEl=document.getElementById(cardId);
  var fullAnswer='', citations=[], retMs=0, genMs=0, tokensUsed=null;

  try{
    var resp=await fetch('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:query,history:chatHistory.slice(-6),embedding_model:model,top_k:topK})
    });
    if(!resp.ok){var e2=await resp.json().catch(function(){return{detail:resp.statusText};});throw new Error(e2.detail||JSON.stringify(e2));}

    var reader=resp.body.getReader(),dec=new TextDecoder(),buf='';
    while(true){
      var ch=await reader.read();
      if(ch.done) break;
      buf+=dec.decode(ch.value,{stream:true});
      var lines=buf.split('\n');buf=lines.pop();
      for(var i=0;i<lines.length;i++){
        if(!lines[i].startsWith('data: ')) continue;
        try{
          var ev=JSON.parse(lines[i].slice(6));
          if(ev.type==='citations'){
            citations=ev.citations; retMs=ev.retrieval_latency_ms;
            hdrEl.innerHTML='<span class="badge '+badgeCls+'">'+badgeLbl+'</span>'
              +'<span class="pill '+(latCls(retMs))+'">Retrieval '+retMs+'ms</span>'
              +'<span class="pill">Generating...</span>';
          } else if(ev.type==='token'){
            fullAnswer+=ev.content;
            bodyEl.innerHTML='<div class="md-body">'+marked.parse(fullAnswer)+'</div>';
            msgs.scrollTop=msgs.scrollHeight;
          } else if(ev.type==='done'){
            genMs=ev.generation_latency_ms;
            tokensUsed=ev.tokens_used;
            bodyEl.classList.remove('stream-cur');
            bodyEl.innerHTML='<div class="md-body">'+renderCitesMarkdown(fullAnswer,citations,uid)+'</div>';
            hdrEl.innerHTML='<span class="badge '+badgeCls+'">'+badgeLbl+'</span>'
              +'<span class="pill '+(latCls(retMs))+'">Retrieval '+retMs+'ms</span>'
              +'<span class="pill '+(latCls(genMs))+'">Generation '+genMs+'ms</span>'
              +(tokensUsed?'<span class="pill">Tokens '+tokensUsed.total+'</span>':'');
            cardEl.innerHTML=cardEl.innerHTML+renderSources(citations,uid);
            chatHistory.push({role:'assistant',content:fullAnswer});
            bindCiteHandlers();
          } else if(ev.type==='error'){
            throw new Error(ev.message);
          }
        }catch(pe){if(pe.message) throw pe;}
      }
    }
  }catch(e){
    bodyEl.classList.remove('stream-cur');
    bodyEl.innerHTML='<span style="color:var(--err)">Error: '+esc(e.message)+'</span>';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat — both models side-by-side (comparison mode, no persistent history)
// ─────────────────────────────────────────────────────────────────────────────
async function sendBothModels(query, topK){
  var msgs=document.getElementById('chatMsgs');
  var uid='both'+Date.now();

  // User question banner
  var uEl=document.createElement('div');
  uEl.innerHTML='<div style="padding:10px 16px;background:var(--blue-bg);border:1px solid var(--blue-border);border-radius:var(--r);font-size:13.5px;font-weight:600;color:var(--blue-2);margin-bottom:4px">'+esc(query)+'</div>';
  msgs.appendChild(uEl.firstElementChild);
  msgs.scrollTop=msgs.scrollHeight;

  var uidOA=uid+'oa', uidCO=uid+'co';

  // Split container
  var splitEl=document.createElement('div');
  splitEl.className='compare-split';
  splitEl.innerHTML=
    '<div class="compare-col">'
    +'<div class="compare-col-hdr oa">OpenAI text-embedding-3-large</div>'
    +'<div id="col-oa-'+uid+'">'
    +'<div class="resp-card">'
    +'<div class="resp-card-hdr" id="hoa-'+uid+'"><span class="badge badge-oa">OpenAI</span><span class="pill">Retrieving...</span></div>'
    +'<div class="resp-card-body stream-cur" id="boa-'+uid+'"><div class="typing"><span></span><span></span><span></span></div></div>'
    +'</div></div>'
    +'</div>'
    +'<div class="compare-col">'
    +'<div class="compare-col-hdr co">Cohere embed-v-4-0</div>'
    +'<div id="col-co-'+uid+'">'
    +'<div class="resp-card">'
    +'<div class="resp-card-hdr" id="hco-'+uid+'"><span class="badge badge-co">Cohere</span><span class="pill">Retrieving...</span></div>'
    +'<div class="resp-card-body stream-cur" id="bco-'+uid+'"><div class="typing"><span></span><span></span><span></span></div></div>'
    +'</div></div>'
    +'</div>';
  msgs.appendChild(splitEl);
  msgs.scrollTop=msgs.scrollHeight;

  // Start both fetches simultaneously
  var payloadOA={query:query,history:[],embedding_model:'openai',top_k:topK};
  var payloadCO={query:query,history:[],embedding_model:'cohere',top_k:topK};

  var [respOA, respCO]=await Promise.all([
    fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payloadOA)}),
    fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payloadCO)})
  ]).catch(function(e){
    toast(e.message,'err'); return [null,null];
  });

  if(!respOA||!respCO) return;

  async function processStream(resp, model, bodyId, hdrId, colId, uid2){
    var bodyEl=document.getElementById(bodyId);
    var hdrEl=document.getElementById(hdrId);
    var colEl=document.getElementById(colId);
    var badgeCls=model==='openai'?'badge-oa':'badge-co';
    var badgeLbl=model==='openai'?'OpenAI':'Cohere';
    var fullAnswer='', citations=[], retMs=0, genMs=0, tokensUsed=null;

    if(!resp.ok){
      var e2=await resp.json().catch(function(){return{detail:resp.statusText};});
      bodyEl.innerHTML='<span style="color:var(--err)">Error: '+esc(e2.detail||'Unknown error')+'</span>';
      bodyEl.classList.remove('stream-cur');
      return;
    }

    try{
      var reader=resp.body.getReader(),dec=new TextDecoder(),buf='';
      while(true){
        var ch=await reader.read();
        if(ch.done) break;
        buf+=dec.decode(ch.value,{stream:true});
        var lines=buf.split('\n');buf=lines.pop();
        for(var i=0;i<lines.length;i++){
          if(!lines[i].startsWith('data: ')) continue;
          try{
            var ev=JSON.parse(lines[i].slice(6));
            if(ev.type==='citations'){
              citations=ev.citations; retMs=ev.retrieval_latency_ms;
              hdrEl.innerHTML='<span class="badge '+badgeCls+'">'+badgeLbl+'</span>'
                +'<span class="pill '+(latCls(retMs))+'">Retrieval '+retMs+'ms</span>'
                +'<span class="pill">Generating...</span>';
            } else if(ev.type==='token'){
              fullAnswer+=ev.content;
              bodyEl.innerHTML='<div class="md-body">'+marked.parse(fullAnswer)+'</div>';
              document.getElementById('chatMsgs').scrollTop=99999;
            } else if(ev.type==='done'){
              genMs=ev.generation_latency_ms; tokensUsed=ev.tokens_used;
              bodyEl.classList.remove('stream-cur');
              bodyEl.innerHTML='<div class="md-body">'+renderCitesMarkdown(fullAnswer,citations,uid2)+'</div>';
              hdrEl.innerHTML='<span class="badge '+badgeCls+'">'+badgeLbl+'</span>'
                +'<span class="pill '+(latCls(retMs))+'">Retrieval '+retMs+'ms</span>'
                +'<span class="pill '+(latCls(genMs))+'">Generation '+genMs+'ms</span>'
                +(tokensUsed?'<span class="pill">Tokens '+tokensUsed.total+'</span>':'');
              // append sources inside the card
              var card=colEl.querySelector('.resp-card');
              if(card) card.innerHTML+=renderSources(citations,uid2);
              bindCiteHandlers();
            } else if(ev.type==='error'){
              bodyEl.classList.remove('stream-cur');
              bodyEl.innerHTML='<span style="color:var(--err)">'+esc(ev.message)+'</span>';
            }
          }catch(pe){if(pe.message&&pe.message!==pe) throw pe;}
        }
      }
    }catch(e){
      bodyEl.classList.remove('stream-cur');
      bodyEl.innerHTML='<span style="color:var(--err)">Error: '+esc(e.message)+'</span>';
    }
  }

  await Promise.all([
    processStream(respOA,'openai','boa-'+uid,'hoa-'+uid,'col-oa-'+uid,uidOA),
    processStream(respCO,'cohere','bco-'+uid,'hco-'+uid,'col-co-'+uid,uidCO)
  ]);
}

// Chat send dispatcher
async function sendChat(){
  var input=document.getElementById('chatInput');
  var query=input.value.trim();
  if(!query) return;
  var model=document.getElementById('chatModel').value;
  var topK=parseInt(document.getElementById('chatTopK').value)||5;

  input.value=''; input.style.height='40px';
  var btn=document.getElementById('chatSend');
  spin(btn,true);

  if(model==='both'){
    await sendBothModels(query,topK);
  } else {
    chatHistory.push({role:'user',content:query});
    await sendSingleModel(query,model,topK);
  }
  spin(btn,false);
  document.getElementById('chatMsgs').scrollTop=99999;
}

document.getElementById('chatSend').addEventListener('click',sendChat);
document.getElementById('chatInput').addEventListener('keydown',function(e){
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();}
});
document.getElementById('chatInput').addEventListener('input',function(){
  this.style.height='40px';
  this.style.height=Math.min(this.scrollHeight,110)+'px';
});
document.getElementById('chatClear').addEventListener('click',function(){
  chatHistory=[];
  document.getElementById('chatMsgs').innerHTML=
    '<div class="msg asst"><div class="avatar">AI</div>'
    +'<div><div class="bubble">Chat cleared. Ask a question about your ingested documents.</div></div></div>';
  toast('Chat cleared','info');
});

// ─────────────────────────────────────────────────────────────────────────────
// DeepEval — Single query
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('sRunBtn').addEventListener('click',async function(){
  var query=document.getElementById('sQuery').value.trim();
  if(!query){toast('Enter a query','err');return;}
  var btn=document.getElementById('sRunBtn');
  spin(btn,true);
  document.getElementById('sEvalOut').innerHTML='<div class="empty card" style="padding:40px">Running DeepEval evaluation...</div>';

  var gt=document.getElementById('sGT').value.trim();
  var ans=document.getElementById('sAnswer').value.trim();
  var selectedModel=document.getElementById('sModel').value;

  // If "both" selected, run two evaluations in parallel
  if(selectedModel==='both'){
    var payloadOA={query:query,embedding_model:'openai',top_k:parseInt(document.getElementById('sTopK').value)||5};
    var payloadCO={query:query,embedding_model:'cohere',top_k:parseInt(document.getElementById('sTopK').value)||5};
    if(gt){payloadOA.ground_truth=gt;payloadCO.ground_truth=gt;}
    if(ans){payloadOA.answer=ans;payloadCO.answer=ans;}
    try{
      var [dataOA,dataCO]=await Promise.all([
        apiFetch('/api/ragas-evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payloadOA)}),
        apiFetch('/api/ragas-evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payloadCO)})
      ]);
      document.getElementById('sEvalOut').innerHTML=renderSingleEvalBoth(dataOA,dataCO);
      toast('DeepEval evaluation complete','ok');
    }catch(e){
      document.getElementById('sEvalOut').innerHTML='<div class="err-box">'+esc(e.message)+'</div>';
      toast(e.message,'err');
    }
    spin(btn,false);
    return;
  }

  var payload={
    query:query,
    embedding_model:selectedModel,
    top_k:parseInt(document.getElementById('sTopK').value)||5
  };
  if(gt) payload.ground_truth=gt;
  if(ans) payload.answer=ans;

  try{
    var data=await apiFetch('/api/ragas-evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    document.getElementById('sEvalOut').innerHTML=renderSingleEval(data);
    toast('DeepEval evaluation complete','ok');
  }catch(e){
    document.getElementById('sEvalOut').innerHTML='<div class="err-box">'+esc(e.message)+'</div>';
    toast(e.message,'err');
  }
  spin(btn,false);
});

function ragasCard(label, value, has){
  if(!has||value===null||value===undefined){
    return '<div class="ragas-card ragas-na">'
      +'<div class="ragas-val">N/A</div>'
      +'<div class="ragas-lbl">'+label+'</div>'
      +'<div style="font-size:10px;color:var(--text-3);margin-top:4px;font-family:var(--mono)">requires ground truth</div>'
      +'</div>';
  }
  var pct=Math.round(value*100);
  return '<div class="ragas-card">'
    +'<div class="ragas-val">'+value.toFixed(4)+'</div>'
    +'<div class="ragas-lbl">'+label+'</div>'
    +'<div class="ragas-bar"><div class="ragas-fill" style="width:'+pct+'%"></div></div>'
    +'</div>';
}

function renderSingleEval(d){
  var model=d.embedding_model;
  var badgeCls=model==='openai'?'badge-oa':'badge-co';
  var badgeLbl=model==='openai'?'OpenAI':'Cohere';
  var hasRef=d.contextrecall!==undefined&&d.contextrecall!==null;
  var hasPrecision=d.contextprecision!==undefined&&d.contextprecision!==null;

  return '<div style="display:flex;flex-direction:column;gap:14px">'
    +'<div class="card"><div class="card-hdr"><span class="card-title">DeepEval Scores</span>'
    +'<div style="display:flex;gap:6px"><span class="badge '+badgeCls+'">'+badgeLbl+'</span><span class="badge badge-ragas">DeepEval</span></div>'
    +'</div><div class="card-body">'
    +'<div class="ragas-grid">'
    +ragasCard('Faithfulness',d.faithfulness,true)
    +ragasCard('Answer Relevancy',d.answerrelevancy,true)
    +ragasCard('Hallucination',d.hallucination,true)
    +ragasCard('Ctx Precision',d.contextprecision,hasPrecision)
    +ragasCard('Ctx Recall',d.contextrecall,hasRef)
    +'</div>'
    +'<div class="stat-row"><span class="skey">Contexts retrieved</span><span class="sval2">'+d.num_contexts+'</span></div>'
    +'<div class="stat-row"><span class="skey">Retrieval latency</span>'
    +'<span class="sval2"><span class="pill '+(latCls(d.retrieval_latency_ms))+'">'+d.retrieval_latency_ms+'ms</span></span></div>'
    +'<div class="stat-row"><span class="skey">Generation latency</span>'
    +'<span class="sval2"><span class="pill '+(latCls(d.generation_latency_ms))+'">'+d.generation_latency_ms+'ms</span></span></div>'
    +'<div class="stat-row"><span class="skey">DeepEval latency</span>'
    +'<span class="sval2"><span class="pill">'+d.eval_latency_ms+'ms</span></span></div>'
    +(d.tokens_used?'<div class="stat-row"><span class="skey">Tokens used</span><span class="sval2">'+d.tokens_used.total+'</span></div>':'')
    +'</div></div>'
    +(d.answer_preview
      ?'<div class="card"><div class="card-hdr"><span class="card-title">Answer Preview</span></div>'
       +'<div class="card-body"><div style="font-size:13px;line-height:1.75;color:var(--text-2)">'+esc(d.answer_preview)+(d.answer_preview.length>=300?'...':'')+'</div></div></div>'
      :'')
    +'</div>';
}

function renderSingleEvalBoth(oa, co){
  var metrics=[
    ['Faithfulness',      'faithfulness',      true],
    ['Answer Relevancy',  'answerrelevancy',   true],
    ['Hallucination',     'hallucination',     true],
    ['Ctx Precision',     'contextprecision',  oa.contextprecision!==undefined&&oa.contextprecision!==null],
    ['Ctx Recall',        'contextrecall',     oa.contextrecall!==undefined&&oa.contextrecall!==null],
  ];
  var rows=metrics.map(function(m){
    var ov=oa[m[1]], cv=co[m[1]];
    if(!m[2]||ov===undefined||ov===null){
      return '<tr><td style="font-weight:700">'+m[0]+'</td>'
        +'<td class="mono" style="color:var(--text-3)">N/A</td>'
        +'<td class="mono" style="color:var(--text-3)">N/A</td></tr>';
    }
    // For hallucination, lower is better
    var won = m[1]==='hallucination'
      ? (ov<cv?'oa':cv<ov?'co':'')
      : (ov>cv?'oa':cv>ov?'co':'');
    return '<tr>'
      +'<td style="font-weight:700;font-size:13px">'+m[0]+'</td>'
      +'<td class="mono" style="color:'+(won==='oa'?'var(--oa)':'var(--text-2)')+'"><strong>'+ov.toFixed(4)+'</strong>'+(won==='oa'?' 🏆':'')+'</td>'
      +'<td class="mono" style="color:'+(won==='co'?'var(--co)':'var(--text-2)')+'"><strong>'+cv.toFixed(4)+'</strong>'+(won==='co'?' 🏆':'')+'</td>'
      +'</tr>';
  }).join('');
  return '<div class="card">'
    +'<div class="card-hdr">'
    +'<span class="card-title">DeepEval — OpenAI vs Cohere</span>'
    +'<div style="display:flex;gap:6px">'
    +'<span class="badge badge-oa">OpenAI</span>'
    +'<span class="badge badge-co">Cohere</span>'
    +'<span class="badge badge-ragas">DeepEval</span>'
    +'</div></div>'
    +'<div class="card-body">'
    +'<table class="data-tbl">'
    +'<thead><tr><th>Metric</th><th style="color:var(--oa)">OpenAI</th><th style="color:var(--co)">Cohere</th></tr></thead>'
    +'<tbody>'+rows+'</tbody>'
    +'</table></div></div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// DeepEval — Benchmark
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('bLoadSample').addEventListener('click',async function(){
  try{
    var data=await apiFetch('/api/ragas-benchmark-sample');
    document.getElementById('bQueries').value=JSON.stringify(data,null,2);
    toast('Loaded '+data.length+' benchmark queries','ok');
  }catch(e){toast('Could not load sample queries','err');}
});

document.getElementById('bRunBtn').addEventListener('click',async function(){
  var raw=document.getElementById('bQueries').value.trim();
  if(!raw){toast('Enter benchmark queries','err');return;}
  var queries;
  try{queries=JSON.parse(raw);}catch(e){toast('Invalid JSON','err');return;}
  var btn=document.getElementById('bRunBtn');
  var model=document.getElementById('bModel').value;
  var k=parseInt(document.getElementById('bK').value)||5;
  spin(btn,true);
  document.getElementById('bEvalOut').innerHTML='<div class="empty card" style="padding:40px">Running DeepEval benchmark — this may take a few minutes...</div>';

  try{
    var data=await apiFetch('/api/ragas-benchmark',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({benchmark_queries:queries,embedding_model:model,k:k})
    });
    document.getElementById('bEvalOut').innerHTML=
      model==='both'
        ? renderBenchCompare(data,k,queries.length)
        : renderBench(data);
    toast('Benchmark complete','ok');
  }catch(e){
    document.getElementById('bEvalOut').innerHTML='<div class="err-box">'+esc(e.message)+'</div>';
    toast(e.message,'err');
  }
  spin(btn,false);
});

function renderBench(d){
  var s=d.summary, model=d.embedding_model;
  var badgeCls=model==='openai'?'badge-oa':'badge-co';
  var badgeLbl=model==='openai'?'OpenAI':'Cohere';
  var hasRecall=s.context_recall_avg!==undefined&&s.context_recall_avg!==null;
  var hasPrecision=s.context_precision_avg!==undefined&&s.context_precision_avg!==null;
  var numQ=s.total||0;
  var failedQ=(d.results||[]).filter(function(r){return r.error;}).length;
  var validQ=numQ-failedQ;

  var aggCards='<div class="ragas-grid">'
    +'<div class="ragas-card"><div class="ragas-val">'+(s.faithfulness_avg!=null?s.faithfulness_avg.toFixed(4):'N/A')+'</div><div class="ragas-lbl">Avg Faithfulness</div>'+(s.faithfulness_avg!=null?'<div class="ragas-bar"><div class="ragas-fill" style="width:'+Math.round(s.faithfulness_avg*100)+'%"></div></div>':'')+'</div>'
    +'<div class="ragas-card"><div class="ragas-val">'+(s.answer_relevancy_avg!=null?s.answer_relevancy_avg.toFixed(4):'N/A')+'</div><div class="ragas-lbl">Avg Ans Relevancy</div>'+(s.answer_relevancy_avg!=null?'<div class="ragas-bar"><div class="ragas-fill" style="width:'+Math.round(s.answer_relevancy_avg*100)+'%"></div></div>':'')+'</div>'
    +'<div class="ragas-card"><div class="ragas-val">'+(s.hallucination_avg!=null?s.hallucination_avg.toFixed(4):'N/A')+'</div><div class="ragas-lbl">Avg Hallucination</div>'+(s.hallucination_avg!=null?'<div class="ragas-bar"><div class="ragas-fill" style="width:'+Math.round(s.hallucination_avg*100)+'%"></div></div>':'')+'</div>'
    +(hasPrecision?'<div class="ragas-card"><div class="ragas-val">'+s.context_precision_avg.toFixed(4)+'</div><div class="ragas-lbl">Avg Ctx Precision</div><div class="ragas-bar"><div class="ragas-fill" style="width:'+Math.round(s.context_precision_avg*100)+'%"></div></div></div>':'<div class="ragas-card ragas-na"><div class="ragas-val">N/A</div><div class="ragas-lbl">Avg Ctx Precision</div></div>')
    +(hasRecall?'<div class="ragas-card"><div class="ragas-val">'+s.context_recall_avg.toFixed(4)+'</div><div class="ragas-lbl">Avg Ctx Recall</div><div class="ragas-bar"><div class="ragas-fill" style="width:'+Math.round(s.context_recall_avg*100)+'%"></div></div></div>':'<div class="ragas-card ragas-na"><div class="ragas-val">N/A</div><div class="ragas-lbl">Avg Ctx Recall</div></div>')
    +'</div>';

  var rows=(d.results||[]).map(function(r,i){
    if(r.error){
      return '<tr><td>'+(i+1)+'</td><td style="max-width:200px;word-break:break-word">'+esc(r.query)+'</td>'
        +'<td colspan="6" style="color:var(--err);font-size:11px">'+esc(r.error)+'</td></tr>';
    }
    return '<tr>'
      +'<td class="mono">'+(i+1)+'</td>'
      +'<td style="max-width:200px;word-break:break-word;font-size:12px">'+esc(r.query)+'</td>'
      +'<td class="mono" style="color:'+ragasColor(r.faithfulness)+'">'+r.faithfulness.toFixed(4)+'</td>'
      +'<td class="mono" style="color:'+ragasColor(r.answerrelevancy)+'">'+r.answerrelevancy.toFixed(4)+'</td>'
      +'<td class="mono">'+(r.hallucination!=null?'<span style="color:'+ragasColor(1-r.hallucination)+'">'+r.hallucination.toFixed(4)+'</span>':'<span style="color:var(--text-3)">—</span>')+'</td>'
      +'<td class="mono">'+(r.contextprecision!=null?'<span style="color:'+ragasColor(r.contextprecision)+'">'+r.contextprecision.toFixed(4)+'</span>':'<span style="color:var(--text-3)">—</span>')+'</td>'
      +'<td class="mono">'+(r.contextrecall!=null?'<span style="color:'+ragasColor(r.contextrecall)+'">'+r.contextrecall.toFixed(4)+'</span>':'<span style="color:var(--text-3)">—</span>')+'</td>'
      +'<td class="mono">'+(r.eval_latency_ms||0)+'ms</td>'
      +'</tr>';
  }).join('');

  return '<div style="display:flex;flex-direction:column;gap:14px">'
    +'<div class="card"><div class="card-hdr"><span class="card-title">Aggregate DeepEval Metrics</span>'
    +'<div style="display:flex;gap:6px"><span class="badge '+badgeCls+'">'+badgeLbl+'</span><span class="badge badge-ragas">DeepEval</span>'
    +'<span class="pill">K='+d.k+'</span><span class="pill">'+numQ+' queries</span>'
    +'<span class="pill">'+validQ+' valid / '+failedQ+' failed</span>'
    +'</div></div>'
    +'<div class="card-body">'+aggCards+'</div></div>'
    +'<div class="card"><div class="card-hdr"><span class="card-title">Per-Query Results</span></div>'
    +'<div style="overflow-x:auto"><table class="data-tbl"><thead><tr>'
    +'<th>#</th><th>Query</th>'
    +'<th style="color:var(--ragas)">Faithfulness</th>'
    +'<th style="color:var(--ragas)">Ans Relevancy</th>'
    +'<th style="color:var(--ragas)">Hallucination</th>'
    +'<th style="color:var(--ragas)">Ctx Precision</th>'
    +'<th style="color:var(--ragas)">Ctx Recall</th>'
    +'<th>Eval Latency</th>'
    +'</tr></thead><tbody>'+rows+'</tbody></table></div></div>'
    +'</div>';
}

function renderBenchCompare(data, k, numQ){
  var oa=data.openai, co=data.cohere;
  var metrics=[
    ['Faithfulness',     'faithfulness_avg'],
    ['Answer Relevancy', 'answer_relevancy_avg'],
    ['Hallucination',    'hallucination_avg'],
    ['Context Precision','context_precision_avg'],
    ['Context Recall',   'context_recall_avg'],
  ];

  var summaryRows=metrics.map(function(m){
    var ov=oa.summary[m[1]], cv=co.summary[m[1]];
    if((ov===undefined||ov===null)&&(cv===undefined||cv===null)){
      return '<tr><td style="font-weight:700">'+m[0]+'</td>'
        +'<td class="mono" style="color:var(--text-3)">N/A</td>'
        +'<td class="mono" style="color:var(--text-3)">N/A</td></tr>';
    }
    ov=ov||0; cv=cv||0;
    // For hallucination, lower wins
    var w = m[1]==='hallucination_avg'
      ? (ov<cv?'oa':cv<ov?'co':'')
      : (ov>cv?'oa':cv>ov?'co':'');
    return '<tr>'
      +'<td style="font-weight:700;font-size:13px">'+m[0]+'</td>'
      +'<td class="mono" style="color:'+(w==='oa'?'var(--oa)':'var(--text-2)')+'"><strong>'+ov.toFixed(4)+'</strong>'+(w==='oa'?' &#10003;':'')+'</td>'
      +'<td class="mono" style="color:'+(w==='co'?'var(--co)':'var(--text-2)')+'"><strong>'+cv.toFixed(4)+'</strong>'+(w==='co'?' &#10003;':'')+'</td>'
      +'</tr>';
  }).join('');

  var perQRows=(oa.results||[]).map(function(r,i){
    var cr=co.results&&co.results[i]?co.results[i]:{};
    var of_=r.faithfulness||0, cf=cr.faithfulness||0;
    var oar=r.answerrelevancy||0, car=cr.answerrelevancy||0;
    return '<tr>'
      +'<td class="mono">'+(i+1)+'</td>'
      +'<td style="max-width:180px;word-break:break-word;font-size:12px">'+esc(r.query)+'</td>'
      +'<td class="mono" style="color:'+(of_>=cf?'var(--oa)':'var(--text-2)')+'">'+of_.toFixed(4)+'</td>'
      +'<td class="mono" style="color:'+(cf>=of_?'var(--co)':'var(--text-2)')+'">'+cf.toFixed(4)+'</td>'
      +'<td class="mono" style="color:'+(oar>=car?'var(--oa)':'var(--text-2)')+'">'+oar.toFixed(4)+'</td>'
      +'<td class="mono" style="color:'+(car>=oar?'var(--co)':'var(--text-2)')+'">'+car.toFixed(4)+'</td>'
      +'</tr>';
  }).join('');

  return '<div style="display:flex;flex-direction:column;gap:14px">'
    +'<div class="card"><div class="card-hdr"><span class="card-title">OpenAI vs Cohere — DeepEval Summary</span>'
    +'<div style="display:flex;gap:6px"><span class="badge badge-ragas">DeepEval</span><span class="pill">K='+k+'</span><span class="pill">'+numQ+' queries</span></div>'
    +'</div><div class="card-body"><table class="data-tbl"><thead><tr>'
    +'<th>Metric</th>'
    +'<th style="color:var(--oa)">OpenAI</th>'
    +'<th style="color:var(--co)">Cohere</th>'
    +'</tr></thead><tbody>'+summaryRows+'</tbody></table></div></div>'
    +'<div class="card"><div class="card-hdr"><span class="card-title">Per-Query Comparison</span></div>'
    +'<div style="overflow-x:auto"><table class="data-tbl"><thead><tr>'
    +'<th>#</th><th>Query</th>'
    +'<th style="color:var(--oa)">OAI Faith.</th><th style="color:var(--co)">Coh. Faith.</th>'
    +'<th style="color:var(--oa)">OAI Relev.</th><th style="color:var(--co)">Coh. Relev.</th>'
    +'</tr></thead><tbody>'+perQRows+'</tbody></table></div></div>'
    +'</div>';
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────
checkStatus();
setInterval(checkStatus, 30000);
</script>
</body>
</html>"""