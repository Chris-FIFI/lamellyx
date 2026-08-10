"""A local dashboard for building membrane systems.

    python -m lamellyx.dashboard

Opens a page on 127.0.0.1 where you drop in a protein, set the parameters and
press Build. Standard library only -- no Flask, no CDN, no build step. It
binds to localhost and serves nothing outside its own workspace.

Everything the page does is also available from `build_membrane` directly;
the dashboard is a front end, not a second implementation.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import topology
from .library import available_lipids
from .membrane import (KNOWN_APL, MembraneConfig, build_membrane,
                       check_settings)

DEFAULT_WORKSPACE = os.path.join(os.path.expanduser("~"), "membrane_builds")

_jobs = {}
_jobs_lock = threading.Lock()
_build_lock = threading.Lock()

# A page you have open in another tab can send requests to a server on
# localhost, and a hostile site can point a domain it controls at 127.0.0.1 to
# get past a naive origin check. Four things stop that here: the server binds
# to the loopback interface only, it refuses any request whose Host header is
# not the address it is serving on, it refuses a cross-origin Origin or
# Referer, and every API call must carry a token that only the process that
# started it knows. The token requirement also means a cross-site form POST
# cannot reach the API, because setting a custom header forces a preflight and
# no CORS headers are ever sent.
MAX_JSON = 1 << 20                 # 1 MB is generous for a settings blob
MAX_UPLOAD = 64 << 20              # 64 MB; a big cryo-EM tetramer is ~10 MB
UPLOAD_SUFFIXES = (".pdb", ".ent", ".gro")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Resource-Policy": "same-origin",
}

# The page is one self-contained file with inline style and script and no
# external resources at all, so everything else can be denied outright.
CSP = ("default-src 'none'; script-src 'unsafe-inline'; "
       "style-src 'unsafe-inline'; connect-src 'self'; form-action 'none'; "
       "base-uri 'none'; frame-ancestors 'none'")


def safe_upload_name(name):
    """A filename that cannot escape a directory or surprise the filesystem."""
    name = _SAFE_NAME.sub("_", os.path.basename(name or "")).lstrip(".")
    if not name:
        name = "protein.pdb"
    stem, ext = os.path.splitext(name)
    if ext.lower() not in UPLOAD_SUFFIXES:
        raise ValueError("only %s files can be uploaded, not %r"
                         % (", ".join(UPLOAD_SUFFIXES), ext or name))
    return stem[:80] + ext.lower()


def looks_like_structure(data):
    """Cheap sniff so a renamed file is caught before the builder reads it."""
    head = data[:200_000]
    return any(k in head for k in (b"ATOM  ", b"HETATM", b"CRYST1")) or \
        (data[:2048].count(b"\n") > 1 and b"." in data[:2048])


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

class _Log:
    """Collects a build's output so the page can stream it."""

    def __init__(self, job):
        self.job = job

    def write(self, s):
        if s:
            with _jobs_lock:
                self.job["log"].append(s.rstrip("\n") if s.strip() else "")
        return len(s)

    def flush(self):
        pass


def _protein_config(d, out):
    """Map the page's settings onto BuildConfig.

    The two configs do not share units: the page and MembraneConfig speak
    nanometres and `salt_concentration`, BuildConfig speaks Angstrom and
    `concentration`. The box's x and y are not taken from the page in this
    mode -- they come from the reference system, because the protein was
    oriented in that frame and the bilayer has to match it.
    """
    from .builder import BuildConfig
    cfg = BuildConfig(
        protein_pdb=d["protein_pdb"],
        reference_dir=d["reference_dir"],
        output_dir=out,
        orient_from_pdb=d.get("orient_from_pdb", "") or "",
        lipid=d.get("lipid", "POPC"),
        water_thickness=float(d.get("water_thickness", 1.5)) * 10.0,
        concentration=float(d.get("salt_concentration", 0.15)),
        cation=d.get("cation", "POT"),
        anion=d.get("anion", "CLA"),
        water=d.get("water_model", "TIP3"),
        temperature=int(d.get("temperature", 310)),
        seed=int(d.get("seed", 0)),
        margin_x=float(d.get("margin_x", 2.0)) * 10.0,
        margin_y=float(d.get("margin_y", 2.0)) * 10.0,
        area_per_lipid=float(d.get("area_per_lipid", 0.15)) * 100.0,
    )
    mol = (d.get("protein_molecules") or "").split()
    ch = (d.get("chains") or "").split()
    if mol:
        cfg.protein_molecules = tuple(mol)
    if ch:
        cfg.chains = tuple(ch)
    if len(cfg.chains) != len(cfg.protein_molecules):
        raise ValueError("%d molecules but %d chains -- they must correspond"
                         % (len(cfg.protein_molecules), len(cfg.chains)))
    if d.get("n_upper"):
        cfg.n_upper = int(d["n_upper"])
    if d.get("n_lower"):
        cfg.n_lower = int(d["n_lower"])
    return cfg


def _run_build(job_id, cfg_dict, workspace):
    job = _jobs[job_id]
    with _build_lock:
        job["status"] = "running"
        out = os.path.join(workspace, job_id)
        with_protein = bool(cfg_dict.get("protein_pdb"))
        old = sys.stdout
        try:
            sys.stdout = _Log(job)
            if with_protein:
                if not cfg_dict.get("reference_dir"):
                    raise ValueError(
                        "embedding a protein needs a reference directory "
                        "containing its topology (PROA.itp and the rest). "
                        "Force-field parameters cannot be generated here -- "
                        "run the sequence through CHARMM-GUI or pdb2gmx once, "
                        "then any conformation of it can be built from here.")
                from .builder import build as build_protein
                pcfg = _protein_config(cfg_dict, out)
                res = build_protein(pcfg)
                counts, box_nm = res.counts, [round(v / 10.0, 3) for v in res.box]
                stats, warnings = res.stats, []
            else:
                mcfg = MembraneConfig(**{
                    k: v for k, v in cfg_dict.items()
                    if k in {f.name for f in fields(MembraneConfig)}})
                mcfg.output_dir = out
                res = build_membrane(mcfg)
                counts, box_nm = res.counts, [round(v, 3) for v in res.box_nm]
                stats, warnings = res.stats, res.warnings
            job["result"] = {
                "counts": counts,
                "box_nm": box_nm,
                "warnings": warnings,
                "stats": {k: v for k, v in stats.items() if k != "contacts"},
                "contacts": stats.get("contacts", {}),
                "output_dir": out,
                "files": sorted(os.listdir(out)),
                "bytes": topology.directory_size(out),
            }
            job["status"] = "done"
        except Exception as exc:                            # noqa: BLE001
            job["status"] = "failed"
            job["error"] = "%s: %s" % (type(exc).__name__, exc)
            job["traceback"] = traceback.format_exc()
            with _jobs_lock:
                job["log"].append("")
                job["log"].append("BUILD FAILED -- %s" % job["error"])
        finally:
            sys.stdout = old
            job["finished"] = time.time()


def _history(workspace):
    out = []
    if not os.path.isdir(workspace):
        return out
    for name in sorted(os.listdir(workspace), reverse=True):
        d = os.path.join(workspace, name)
        if not os.path.isdir(d) or name == "uploads":
            continue
        rep = os.path.join(d, "build_report.json")
        entry = {"id": name, "bytes": topology.directory_size(d),
                 "mtime": os.path.getmtime(d), "ok": os.path.exists(rep)}
        if entry["ok"]:
            try:
                with open(rep) as fh:
                    r = json.load(fh)
                entry["counts"] = r.get("counts", {})
                entry["box_nm"] = [round(v, 2) for v in r.get("box_nm", [])]
                entry["lipid"] = r.get("config", {}).get("lipid", "")
            except Exception:                               # noqa: BLE001
                entry["ok"] = False
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------

class Server(ThreadingHTTPServer):
    """A server that refuses to share its port.

    `http.server` sets SO_REUSEADDR, and on Windows that flag does not mean
    what it means on Unix: it lets a second process bind a port another
    process is already listening on. Both then accept connections and which
    one answers is arbitrary. Starting the dashboard a few times in a row
    leaves several of them alive, and requests are served by whichever the
    kernel picks -- so a page can silently be talking to a build of the code
    from an hour ago, which is confusing at best and a way to intercept
    requests at worst.

    SO_EXCLUSIVEADDRUSE is the Windows flag that actually prevents this.
    Elsewhere, turning SO_REUSEADDR off is enough.
    """

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        opt = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if opt is not None:
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, opt, 1)
            except OSError:
                pass
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    workspace = DEFAULT_WORKSPACE
    token = ""
    allowed_hosts = ()
    server_version = "lamellyx"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):        # quieter console
        pass

    def version_string(self):
        return "lamellyx"          # no Python version in the banner

    # -- helpers -----------------------------------------------------------

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self, limit=MAX_JSON):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > limit:
            raise _TooLarge("body is %d bytes, the limit is %d" % (n, limit))
        return self.rfile.read(n) if n else b""

    # -- access control ----------------------------------------------------

    def _reject(self):
        """None if the request may proceed, else (code, message).

        Checked before anything is parsed, so a request that fails here never
        reaches the builder or the filesystem.
        """
        host = (self.headers.get("Host") or "").strip()
        if host not in self.allowed_hosts:
            return 403, ("refused: Host header %r is not this server's "
                         "address. This blocks DNS rebinding." % host)
        allowed_origins = {"http://%s" % h for h in self.allowed_hosts}
        origin = self.headers.get("Origin")
        if origin and origin not in allowed_origins:
            return 403, "refused: cross-origin request from %s" % origin
        ref = self.headers.get("Referer")
        if ref and not any(ref.startswith(o + "/") or ref == o
                           for o in allowed_origins):
            return 403, "refused: cross-site referer"
        return None

    def _authorised(self, query_token=None):
        given = self.headers.get("X-Auth-Token") or query_token or ""
        if not self.token or not hmac.compare_digest(str(given), self.token):
            return False
        return True

    def _safe(self, *parts):
        """Join under the workspace, refusing anything that escapes it."""
        root = os.path.abspath(self.workspace)
        p = os.path.abspath(os.path.join(root, *parts))
        if p != root and not p.startswith(root + os.sep):
            raise ValueError("path outside the workspace")
        return p

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        u = urlparse(self.path)
        bad = self._reject()
        if bad:
            return self._send(bad[0], {"error": bad[1]})
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                if not self._authorised(q.get("t", [None])[0]):
                    return self._send(401, LOCKED,
                                      "text/html; charset=utf-8",
                                      {"Content-Security-Policy": CSP})
                return self._send(200, PAGE, "text/html; charset=utf-8",
                                  {"Content-Security-Policy": CSP})
            if not self._authorised():
                return self._send(401, {"error": "missing or bad auth token"})
            if u.path == "/api/state":
                return self._send(200, {
                    "defaults": asdict(MembraneConfig()),
                    "fields": [f.name for f in fields(MembraneConfig)],
                    "lipids": available_lipids(),
                    "known_apl": KNOWN_APL,
                    "workspace": self.workspace,
                    "workspace_bytes": topology.directory_size(self.workspace)
                    if os.path.isdir(self.workspace) else 0,
                    "history": _history(self.workspace),
                })
            if u.path.startswith("/api/job/"):
                jid = u.path.rsplit("/", 1)[-1]
                job = _jobs.get(jid)
                if not job:
                    return self._send(404, {"error": "no such job"})
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                with _jobs_lock:
                    lines = job["log"][since:]
                    total = len(job["log"])
                return self._send(200, {
                    "status": job["status"], "lines": lines, "next": total,
                    "result": job.get("result"), "error": job.get("error"),
                    "traceback": job.get("traceback"),
                })
            if u.path.startswith("/api/file/"):
                rel = unquote(u.path[len("/api/file/"):])
                p = self._safe(rel)
                if not os.path.isfile(p):
                    return self._send(404, {"error": "not found"})
                with open(p, "rb") as fh:
                    data = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition",
                                 'attachment; filename="%s"' % os.path.basename(p))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            return self._send(404, {"error": "not found"})
        except _TooLarge as exc:
            return self._send(413, {"error": str(exc)})
        except ValueError as exc:
            return self._send(403, {"error": str(exc)})
        except Exception as exc:                            # noqa: BLE001
            return self._send(500, {"error": str(exc)})

    def do_POST(self):
        u = urlparse(self.path)
        bad = self._reject()
        if bad:
            return self._send(bad[0], {"error": bad[1]})
        if not self._authorised():
            return self._send(401, {"error": "missing or bad auth token"})
        try:
            if u.path == "/api/check":
                # The page also carries protein settings, which belong to a
                # different config; keep only what MembraneConfig knows.
                known = {f.name for f in fields(MembraneConfig)}
                raw = json.loads(self._body())
                cfg = MembraneConfig(**{k: v for k, v in raw.items()
                                        if k in known})
                issues = check_settings(
                    cfg, protein=bool(raw.get("protein_pdb")),
                    xy_margin=raw.get("margin_x"))
                return self._send(200, {"issues": [
                    {"level": l, "message": m} for l, m in issues]})

            if u.path == "/api/upload":
                name = safe_upload_name(
                    parse_qs(u.query).get("name", ["protein.pdb"])[0])
                data = self._body(MAX_UPLOAD)
                if not data:
                    return self._send(400, {"error": "empty upload"})
                if not looks_like_structure(data):
                    return self._send(400, {
                        "error": "this does not look like a structure file "
                                 "-- no ATOM, HETATM or CRYST1 record found"})
                up = self._safe("uploads")
                os.makedirs(up, exist_ok=True)
                p = os.path.join(up, name)
                with open(p, "wb") as fh:
                    fh.write(data)
                return self._send(200, {"path": p, "name": name,
                                        "bytes": os.path.getsize(p)})

            if u.path == "/api/build":
                cfg = json.loads(self._body())
                cfg.pop("output_dir", None)
                jid = time.strftime("%Y%m%d-%H%M%S")
                if jid in _jobs:
                    jid += "-%d" % (len(_jobs) % 100)
                _jobs[jid] = {"status": "queued", "log": [],
                              "started": time.time()}
                threading.Thread(target=_run_build,
                                 args=(jid, cfg, self.workspace),
                                 daemon=True).start()
                return self._send(200, {"job": jid})

            if u.path == "/api/delete":
                jid = os.path.basename(json.loads(self._body())["id"])
                shutil.rmtree(self._safe(jid), ignore_errors=True)
                return self._send(200, {"ok": True})

            return self._send(404, {"error": "not found"})
        except _TooLarge as exc:
            return self._send(413, {"error": str(exc)})
        except TypeError as exc:
            return self._send(400, {"error": "bad setting: %s" % exc})
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:                            # noqa: BLE001
            return self._send(500, {"error": str(exc),
                                    "traceback": traceback.format_exc()})


class _TooLarge(Exception):
    pass


LOCKED = """<!doctype html><meta charset="utf-8"><title>Lamellyx</title>
<style>body{font:15px/1.6 system-ui,sans-serif;max-width:34em;margin:14vh auto;
padding:0 6vw;color:#1a1a18}code{background:#eee;padding:2px 6px;border-radius:4px}
@media(prefers-color-scheme:dark){body{background:#16171a;color:#e8e8e4}
code{background:#2c2e33}}</style>
<h1>Not authorised</h1>
<p>This dashboard needs the one-time link printed in the terminal that started
it. Open that link, or restart with <code>python -m lamellyx.dashboard</code>
and use the address it prints.</p>
<p>The token stops anything else on this machine &mdash; including a page open
in another tab &mdash; from driving the builder.</p>"""


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lamellyx</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e2e1dd;--card:#fff;
--acc:#1f6f5c;--warn:#8a6100;--err:#a32d22;--ok:#1f6f5c;--mono:ui-monospace,
"SF Mono",Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e8e8e4;--mut:#9a9a94;
--line:#2c2e33;--card:#1d1f23;--acc:#5fbfa3;--warn:#d9a441;--err:#e5796b;--ok:#5fbfa3}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,
-apple-system,Segoe UI,sans-serif}
header{padding:20px 24px 14px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:19px;letter-spacing:-.01em}
header p{margin:4px 0 0;color:var(--mut);font-size:13px}
.wrap{display:grid;grid-template-columns:minmax(320px,400px) 1fr;gap:20px;
padding:20px 24px;align-items:start}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin-bottom:16px}
.card h2{margin:0 0 12px;font-size:12px;text-transform:uppercase;
letter-spacing:.07em;color:var(--mut);font-weight:600}
label{display:block;font-size:12px;color:var(--mut);margin:10px 0 3px}
input,select{width:100%;padding:7px 9px;border:1px solid var(--line);
border-radius:6px;background:var(--bg);color:var(--fg);font:inherit;font-size:14px}
input:focus,select:focus{outline:2px solid var(--acc);outline-offset:-1px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
button{font:inherit;font-weight:600;padding:9px 16px;border-radius:7px;
border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
button.ghost{background:transparent;color:var(--fg);border-color:var(--line);
font-weight:500}
#drop{border:1.5px dashed var(--line);border-radius:8px;padding:18px;
text-align:center;color:var(--mut);font-size:13px;cursor:pointer}
#drop.over{border-color:var(--acc);color:var(--acc)}
.issue{border-left:3px solid;padding:7px 10px;margin:7px 0;font-size:12.5px;
border-radius:0 5px 5px 0;background:var(--bg)}
.issue.error{border-color:var(--err)}.issue.warning{border-color:var(--warn)}
.issue b{display:block;font-size:11px;text-transform:uppercase;
letter-spacing:.06em;margin-bottom:2px}
.issue.error b{color:var(--err)}.issue.warning b{color:var(--warn)}
pre#log{margin:0;font-family:var(--mono);font-size:12px;line-height:1.5;
max-height:46vh;overflow:auto;white-space:pre-wrap;word-break:break-word;
background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;
font-weight:600;letter-spacing:.03em}
.pill.done{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.pill.failed{background:color-mix(in srgb,var(--err) 18%,transparent);color:var(--err)}
.pill.running{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.hint{font-size:11.5px;color:var(--mut);margin-top:3px}
.mono{font-family:var(--mono);font-size:12px}
a{color:var(--acc)}
details summary{cursor:pointer;font-size:12px;color:var(--mut);
text-transform:uppercase;letter-spacing:.07em;font-weight:600}
</style></head><body>
<header>
  <h1>Lamellyx</h1>
  <p>Packs a bilayer, solvates it, adds salt, and writes a directory GROMACS can run.
     Optionally embeds a protein.</p>
</header>
<div class="wrap">
 <div>
  <div class="card">
    <h2>Protein <span style="text-transform:none;font-weight:400">(optional)</span></h2>
    <div id="drop">Drop a PDB here, or click to choose<br>
      <span class="hint">leave empty for a plain bilayer</span></div>
    <input type="file" id="file" accept=".pdb" hidden>
    <div id="fileinfo" class="hint"></div>
    <div id="protfields" style="display:none">
      <label>Reference directory <span class="hint">toppar/ + step5_input.gro</span></label>
      <input id="reference_dir" placeholder="…/charmm-gui-XXXX/gromacs">
      <label>Orientation reference <span class="hint">PDB the reference was built from</span></label>
      <input id="orient_from_pdb" placeholder="optional">
      <div class="row"><div><label>Molecules</label>
      <input id="protein_molecules" value="PROA PROB PROC PROD"></div>
      <div><label>Chains</label><input id="chains" value="A B C D"></div></div>
    </div>
  </div>

  <div class="card">
    <h2>Membrane</h2>
    <div class="row">
      <div><label>Lipid</label><select id="lipid"></select></div>
      <div><label>Arrangement</label><select id="leaflet">
        <option>bilayer</option><option>monolayer</option></select></div>
    </div>
    <label>Length of X and Y based on</label>
    <select id="size_mode">
      <option value="lipid_numbers">numbers of lipid components</option>
      <option value="box">an explicit box size</option>
      <option value="protein_margin">the protein, plus a margin each side</option>
    </select>

    <div id="numbersbox">
      <div class="row">
        <div><label>Upper leaflet</label>
          <input id="n_upper" type="number" step="1" min="1"></div>
        <div><label>Lower leaflet</label>
          <input id="n_lower" type="number" step="1" min="0"></div>
      </div>
      <div class="hint">Counts in, box out &mdash; the same way round as
        CHARMM-GUI.</div>
    </div>

    <div id="explicitbox" style="display:none">
      <div class="row">
        <div><label>X (nm)</label><input id="x" type="number" step="0.1"></div>
        <div><label>Y (nm)</label><input id="y" type="number" step="0.1"></div>
      </div>
      <div class="hint">The full box edge, not a half-width. Minimum 2.4 nm
        for CHARMM36's 1.2 nm cutoffs. Lipid counts follow from the area.</div>
    </div>

    <div id="marginbox" style="display:none">
      <div class="row">
        <div><label>X-dimension (nm)</label>
          <input id="margin_x" type="number" step="0.1" min="0"></div>
        <div><label>Y-dimension (nm)</label>
          <input id="margin_y" type="number" step="0.1" min="0"></div>
      </div>
      <div class="hint">Membrane between the protein and the box edge, left
        and right. CHARMM-GUI's default is 20 &Aring;, which is 2.0 nm. The
        width is measured across the protein's <em>transmembrane</em>
        section, so a large extracellular domain does not inflate the box.</div>
    </div>

    <details style="margin-top:12px">
      <summary>Area per lipid</summary>
      <label>nm² per lipid &mdash; 0 uses the measured value</label>
      <input id="area_per_lipid" type="number" step="0.001" min="0">
      <div id="aplhint" class="hint"></div>
      <div class="hint">CHARMM-GUI has no such field; it uses a measured area
        for each lipid type. Setting one overrides that.</div>
    </details>
    <div id="lipidcount" class="hint"></div>
    <div id="boxnote" class="hint"></div>
  </div>

  <div class="card">
    <h2>Solvent</h2>
    <div class="row3">
      <div><label>Water (nm)</label>
        <input id="water_thickness" type="number" step="0.1"></div>
      <div><label>Salt (M)</label>
        <input id="salt_concentration" type="number" step="0.05"></div>
      <div><label>Temp (K)</label><input id="temperature" type="number"></div>
    </div>
    <div class="row3">
      <div><label>Cation</label><input id="cation"></div>
      <div><label>Anion</label><input id="anion"></div>
      <div><label>Water model</label><input id="water_model"></div>
    </div>
  </div>

  <div class="card">
    <h2>Force field &amp; storage</h2>
    <div class="row">
      <div><label>Force field</label><input id="forcefield"></div>
      <div><label>Parameters</label><select id="toppar_mode">
        <option value="copy">copy into each build</option>
        <option value="reference">share one directory</option></select></div>
    </div>
    <div class="hint">Sharing keeps every build a few hundred kB smaller and
      is safe as long as the directory stays put.</div>
    <div class="row" style="margin-top:10px">
      <div><label>Seed</label><input id="seed" type="number"></div>
      <div><label>Ignore unrunnable settings</label>
        <select id="strict"><option value="true">no, stop and explain</option>
        <option value="false">yes, build anyway</option></select></div>
    </div>
  </div>

  <div id="issues"></div>
  <button id="go" style="width:100%">Build</button>
 </div>

 <div>
  <div class="card"><h2>Build log</h2><pre id="log">Ready.</pre></div>
  <div class="card" id="resultcard" style="display:none">
    <h2>Result</h2><div id="result"></div></div>
  <div class="card">
    <h2>Previous builds <span id="wsbytes" style="text-transform:none;
      font-weight:400;color:var(--mut)"></span></h2>
    <div id="history"></div>
    <div class="hint" id="wspath"></div>
  </div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);

// The token arrives in the URL, is kept in memory, and is scrubbed from the
// address bar straight away so it does not end up in history or a screenshot.
const TOKEN=new URLSearchParams(location.search).get("t")||"";
history.replaceState(null,"",location.pathname);

function api(path,opts){
 opts=opts||{}; opts.headers=Object.assign({"X-Auth-Token":TOKEN},opts.headers||{});
 opts.credentials="omit";
 return fetch(path,opts);
}
async function apiJSON(path,opts){
 const r=await api(path,opts);
 if(!r.ok){let m; try{m=(await r.json()).error}catch(e){m=r.statusText}
   throw new Error(m||("HTTP "+r.status))}
 return r.json();
}
// Anything that came from the server or the filesystem is escaped before it
// touches innerHTML -- a file called <img onerror=...> is a valid filename.
const esc=s=>String(s==null?"":s).replace(/[&<>"']/g,
  c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const NUM=["x","y","area_per_lipid","water_thickness","salt_concentration",
"temperature","seed","margin_x","margin_y","n_upper","n_lower"];
const TEXT=["lipid","leaflet","cation","anion","water_model","forcefield",
"toppar_mode","reference_dir","orient_from_pdb","protein_molecules",
"chains","size_mode"];
let STATE=null, PROTEIN=null, POLL=null;

function bytes(n){if(n==null)return"";const u=["B","kB","MB","GB"];let i=0;
 while(n>=1024&&i<u.length-1){n/=1024;i++}return n.toFixed(i?1:0)+" "+u[i]}

function cfg(){
 const c={};
 for(const k of NUM){const e=$(k); if(e) c[k]=parseFloat(e.value)||0}
 for(const k of TEXT){const e=$(k); if(e&&e.value!=="") c[k]=e.value}
 c.seed=Math.round(c.seed||0); c.temperature=Math.round(c.temperature||310);
 c.strict = $("strict").value==="true";
 if(PROTEIN) c.protein_pdb=PROTEIN;
 if(sizeMode()==="protein_margin" && !PROTEIN) c.size_mode="lipid_numbers";
 return c;
}

async function load(){
 try{ STATE=await apiJSON("/api/state") }
 catch(e){ $("log").textContent="Cannot reach the builder: "+e.message+
   "\n\nOpen the link printed in the terminal that started the dashboard.";
   $("go").disabled=true; return }
 const d=STATE.defaults;
 for(const k of NUM.concat(TEXT)){const e=$(k); if(e&&d[k]!==undefined)e.value=d[k]}
 const sel=$("lipid"); sel.innerHTML="";
 for(const l of (STATE.lipids.length?STATE.lipids:["POPC"])){
   const o=document.createElement("option");o.textContent=l;sel.appendChild(o)}
 sel.value=d.lipid;
 $("wspath").textContent="Workspace: "+STATE.workspace;
 $("wsbytes").textContent="— "+bytes(STATE.workspace_bytes)+" total";
 history(); check();
}

function history(){
 const h=STATE.history||[];
 if(!h.length){$("history").innerHTML='<div class="hint">Nothing built yet.</div>';return}
 let t='<table><tr><th>Build</th><th>System</th><th class="num">Size</th><th></th></tr>';
 for(const e of h){
  const c=e.counts||{};
  const desc=Object.entries(c).filter(([k])=>k!=="TOTAL_ATOMS")
    .map(([k,v])=>v+" "+k).join(", ")||"incomplete";
  t+=`<tr><td class="mono">${esc(e.id)}</td><td>${esc(desc)}<br>
    <span class="hint">${(e.box_nm||[]).join(" × ")} nm</span></td>
    <td class="num">${bytes(e.bytes)}</td>
    <td><button class="ghost" onclick="del('${esc(e.id)}')">Delete</button></td></tr>`;
 }
 $("history").innerHTML=t+"</table>";
}

async function del(id){
 if(!confirm("Delete build "+id+"? This removes the folder and its files."))return;
 await apiJSON("/api/delete",{method:"POST",body:JSON.stringify({id})});
 await load();
}

let ct=null;
function check(){clearTimeout(ct);ct=setTimeout(doCheck,220)}
async function doCheck(){
 const apl=parseFloat($("area_per_lipid").value), lip=$("lipid").value;
 const ref=STATE&&STATE.known_apl[lip];
 $("aplhint").textContent = ref ?
   `measured ${lip}: ${ref} nm² — you have ${apl?(100*apl/ref).toFixed(0):"—"}% of it` : "";
 const bx=parseFloat($("x").value), by=parseFloat($("y").value);
 const eff = apl>0 ? apl : (ref||0.643);
 const m = sizeMode();
 if(m==="protein_margin"){ $("lipidcount").textContent = PROTEIN
   ? "Lipid count is worked out from the free area once the protein has taken its share."
   : ""; }
 else if(m==="lipid_numbers"){
   const nu=parseInt($("n_upper").value)||0, nl=parseInt($("n_lower").value)||0;
   const side=Math.sqrt(Math.max(nu,nl,1)*eff);
   $("lipidcount").textContent =
     (nu+nl)+" lipids give a "+side.toFixed(2)+" x "+side.toFixed(2)+" nm box";
 }
 else if(bx>0&&by>0){
   const per=Math.max(1,Math.round(bx*by/eff));
   const both=$("leaflet").value==="bilayer";
   $("lipidcount").textContent =
     per+" lipids per leaflet, "+(both?per*2:per)+" in total";
 } else $("lipidcount").textContent="";
 let r;
 try{ r=await apiJSON("/api/check",{method:"POST",
      body:JSON.stringify(cfg())}) }catch(e){return}
 const box=$("issues"); box.innerHTML="";
 let err=false;
 for(const i of (r.issues||[])){
  if(i.level==="error")err=true;
  const d=document.createElement("div");
  d.className="issue "+i.level;
  d.innerHTML="<b>"+esc(i.level)+"</b>"+esc(i.message);
  box.appendChild(d);
 }
 $("go").textContent = err && $("strict").value==="true"
   ? "Build (fix the errors first)" : "Build";
}

$("drop").onclick=()=>$("file").click();
$("drop").ondragover=e=>{e.preventDefault();$("drop").classList.add("over")};
$("drop").ondragleave=()=>$("drop").classList.remove("over");
$("drop").ondrop=e=>{e.preventDefault();$("drop").classList.remove("over");
 if(e.dataTransfer.files[0])upload(e.dataTransfer.files[0])};
$("file").onchange=e=>{if(e.target.files[0])upload(e.target.files[0])};

async function upload(f){
 $("fileinfo").textContent="Uploading "+f.name+"…";
 let r;
 try{ r=await apiJSON("/api/upload?name="+encodeURIComponent(f.name),
   {method:"POST",body:await f.arrayBuffer()}) }
 catch(e){ $("fileinfo").textContent="Upload refused: "+e.message; return }
 PROTEIN=r.path;
 $("fileinfo").innerHTML="<b>"+esc(r.name)+"</b> — "+bytes(r.bytes)+
   ' · <a href="#" onclick="clearProtein();return false">remove</a>';
 $("protfields").style.display="block";
 $("drop").textContent="Replace protein";
 boxMode();
}
function sizeMode(){ return $("size_mode").value }
function boxMode(){
 // With a protein loaded the box is always sized from it. An explicit box
 // plus a protein is not a combination the builder supports, so the choice
 // is removed rather than offered and then ignored.
 const sel=$("size_mode");
 if(PROTEIN) sel.value="protein_margin";
 for(const o of sel.options) o.disabled = PROTEIN && o.value!=="protein_margin";
 const m=sizeMode();
 $("numbersbox").style.display  = m==="lipid_numbers"  ? "block" : "none";
 $("explicitbox").style.display = m==="box"            ? "block" : "none";
 $("marginbox").style.display   = m==="protein_margin" ? "block" : "none";
 if(m==="protein_margin" && !PROTEIN){
   $("boxnote").innerHTML='<b>Upload a protein above</b> — a margin has '+
     'nothing to measure from.';
 } else if(m==="protein_margin"){
   $("boxnote").textContent="Box X and Y = protein width + 2 x margin. "+
     "Lipid count follows from the area left over.";
 } else {
   $("boxnote").textContent="";
 }
 check();
}
function clearProtein(){PROTEIN=null;$("fileinfo").textContent="";
 $("protfields").style.display="none"; boxMode();
 $("drop").innerHTML='Drop a PDB here, or click to choose<br>'+
 '<span class="hint">leave empty for a plain bilayer</span>'}

for(const k of NUM.concat(TEXT,["strict"])){
 const e=$(k); if(e)e.addEventListener("input",check)}
$("size_mode").addEventListener("change",boxMode);
$("leaflet").addEventListener("change",check);

$("go").onclick=async()=>{
 if(PROTEIN && !$("reference_dir").value.trim()){
   alert("Embedding a protein needs a reference directory containing its "+
     "topology (PROA.itp and the rest). Force-field parameters cannot be "+
     "generated here — run the sequence through CHARMM-GUI or pdb2gmx once, "+
     "then any conformation of it can be built from this page.");
   return;
 }
 $("go").disabled=true; $("log").textContent="";
 $("resultcard").style.display="none";
 let r;
 try{ r=await apiJSON("/api/build",{method:"POST",body:JSON.stringify(cfg())}) }
 catch(e){ $("log").textContent="Error: "+e.message; $("go").disabled=false; return }
 if(r.error){$("log").textContent="Error: "+r.error;$("go").disabled=false;return}
 poll(r.job,0);
};

async function poll(job,since){
 const r=await apiJSON("/api/job/"+job+"?since="+since);
 if(r.lines&&r.lines.length){
  $("log").textContent += (since?"\n":"")+r.lines.join("\n");
  $("log").scrollTop=$("log").scrollHeight;
 }
 if(r.status==="done"||r.status==="failed"){
  $("go").disabled=false;
  if(r.status==="done")showResult(job,r.result); else showError(r);
  STATE=await apiJSON("/api/state"); history();
  $("wsbytes").textContent="— "+bytes(STATE.workspace_bytes)+" total";
  return;
 }
 setTimeout(()=>poll(job,r.next),450);
}

function showError(r){
 $("resultcard").style.display="block";
 $("result").innerHTML='<span class="pill failed">failed</span><p>'+
  esc(r.error)+"</p>"+
  (r.traceback?'<details><summary>Traceback</summary><pre class="mono">'+
   esc(r.traceback)+"</pre></details>":"");
}

function showResult(job,res){
 $("resultcard").style.display="block";
 const c=res.counts||{}, s=res.stats||{}, k=res.contacts||{};
 let rows="";
 for(const [n,v] of Object.entries(c))
   rows+=`<tr><td>${esc(n)}</td><td class="num">${Number(v).toLocaleString()}</td></tr>`;
 const files=(res.files||[]).filter(f=>f!=="toppar").map(f=>
   `<a href="#" onclick="dl('${esc(job)}','${esc(f)}');return false">${esc(f)}</a>`).join(" · ");
 $("result").innerHTML=`<span class="pill done">done</span>
  <table style="margin-top:10px">${rows}
  <tr><td>box</td><td class="num">${(res.box_nm||[]).join(" × ")} nm</td></tr>
  <tr><td>area per lipid</td><td class="num">${(s.area_per_lipid_nm2||0).toFixed(3)} nm²</td></tr>
  <tr><td>density</td><td class="num">${(s.density_g_cm3||0).toFixed(3)} g/cm³</td></tr>
  <tr><td>closest heavy-atom pair</td><td class="num">${(k.heavy_min||0).toFixed(2)} Å</td></tr>
  <tr><td>pairs under 2.4 Å</td><td class="num">${k["heavy_below_2.4"]??"—"}</td></tr>
  <tr><td>net charge</td><td class="num">${(s.net_charge||0).toFixed(3)}</td></tr>
  <tr><td>on disk</td><td class="num">${bytes(res.bytes)}</td></tr>
  <tr><td>build time</td><td class="num">${(s.seconds||0).toFixed(0)} s</td></tr>
  </table>
  <p class="hint" style="margin-top:10px">${esc(res.output_dir)}</p>
  <p style="font-size:12.5px">${files}</p>
  <p class="hint">Next: <span class="mono">gmx grompp -f step6.0_minimization.mdp
  -c step5_input.gro -r step5_input.gro -p topol.top -n index.ndx -o min.tpr</span></p>`;
}
async function dl(job,f){
 const r=await api("/api/file/"+encodeURIComponent(job)+"/"+encodeURIComponent(f));
 if(!r.ok){alert("Download refused ("+r.status+")");return}
 const b=await r.blob(), a=document.createElement("a");
 a.href=URL.createObjectURL(b); a.download=f; a.click();
 setTimeout(()=>URL.revokeObjectURL(a.href),5000);
}
boxMode();
load();
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8733,
                   help="0 picks a free port")
    p.add_argument("--workspace", default=DEFAULT_WORKSPACE,
                   help="where builds are written (default %s)" % DEFAULT_WORKSPACE)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--token", help="fix the access token instead of "
                                   "generating a fresh one each start")
    args = p.parse_args(argv)

    os.makedirs(args.workspace, exist_ok=True)
    Handler.workspace = os.path.abspath(args.workspace)
    Handler.token = args.token or secrets.token_urlsafe(32)

    try:
        srv = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        print("Could not listen on port %d: %s\n" % (args.port, exc))
        print("Something is already using it -- most likely a dashboard that "
              "is still running.\nStop it, or start this one on a free port:")
        print("    python -m lamellyx.dashboard --port 0")
        return 1
    port = srv.server_address[1]
    Handler.allowed_hosts = ("127.0.0.1:%d" % port, "localhost:%d" % port)
    url = "http://127.0.0.1:%d/?t=%s" % (port, Handler.token)

    print("Lamellyx dashboard")
    print("  open: %s" % url)
    print("  workspace: %s" % Handler.workspace)
    print("  bound to 127.0.0.1 only; the token in that link is required for "
          "every request")
    print("  Ctrl-C to stop")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
