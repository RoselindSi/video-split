"""Loopback-only blind reviewer for the 30-case graph-contract packet."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import io
import json
import math
import re
import secrets
import threading
from pathlib import Path
from urllib.parse import urlparse

from src.auditor.boundary.final_timeline_audit import (
    AuditServer, Handler, atomic_json, file_sha256, safe_name,
)
from src.auditor.boundary.interaction_graph_contract_review import ALLOWED, REVIEW_FIELDS


SCHEMA = "interaction_graph_contract_human_review_v1"
RID = re.compile(r"graph_review_[0-9]{4}")
BOUNDARIES = {"SAME_ACTION_NEW_INSTANCE", "NEW_ACTION", "BOUNDARY_TYPE_UNRESOLVED"}


def validate(value, review_id, duration, final):
    if not isinstance(value, dict) or value.get("review_id") != review_id:
        raise ValueError("Review ID does not match")
    if not isinstance(value.get("revision"), int) or isinstance(value["revision"], bool):
        raise ValueError("Missing revision")
    result = {"review_id": review_id}
    for field, allowed in ALLOWED.items():
        item = value.get(field, "")
        if not isinstance(item, str) or (item and item not in allowed):
            raise ValueError(f"Invalid {field}")
        result[field] = item
    result["observed_type_path"] = str(value.get("observed_type_path") or "").strip()[:200]
    result["observable_evidence"] = str(value.get("observable_evidence") or "").strip()[:2000]
    for field in ("boundary_start_s", "boundary_end_s"):
        raw = value.get(field, "")
        if raw in ("", None):
            result[field] = ""
            continue
        try:
            number = round(float(raw), 3)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {field}") from None
        if not math.isfinite(number) or not 0 <= number <= duration:
            raise ValueError(f"{field} is outside the clip")
        result[field] = number
    if result["boundary_start_s"] != "" and result["boundary_end_s"] != "" \
            and result["boundary_start_s"] > result["boundary_end_s"]:
        raise ValueError("Boundary start is after boundary end")
    result["human_confirmed"] = value.get("human_confirmed") is True
    if final:
        for field in ALLOWED:
            if not result[field]:
                raise ValueError(f"Choose {field.replace('_', ' ')}")
        if result["relation"] in BOUNDARIES | {"END_ONLY"} and result["boundary_start_s"] == "":
            raise ValueError("Boundary relations require a boundary time or interval")
        relation = result["relation"]
        expected_type = {"CONTINUE": "same", "SAME_ACTION_NEW_INSTANCE": "same",
                         "NEW_ACTION": "different", "BOUNDARY_TYPE_UNRESOLVED": "unknown"}
        if relation in expected_type and result["type_equivalence"] != expected_type[relation]:
            raise ValueError("Relation conflicts with type equivalence")
        expected_link = {"CONTINUE": "same_instance",
                         "SAME_ACTION_NEW_INSTANCE": "new_instance"}
        if relation in expected_link and result["instance_link"] != expected_link[relation]:
            raise ValueError("Relation conflicts with instance link")
        if result["return_to_prior_type"] == "yes" and not result["observed_type_path"]:
            raise ValueError("Return to prior type requires an observed type path")
        if not result["observable_evidence"]:
            raise ValueError("Add observable evidence")
        if not result["human_confirmed"]:
            raise ValueError("Confirm that you reviewed the full clip")
    return result


class Store:
    def __init__(self, manifest_path, out, reviewer):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != "interaction_graph_contract_review_v1" \
                or manifest.get("n_cases") != 30:
            raise ValueError("Not a 30-case graph-contract manifest")
        rows = manifest.get("cases") or []
        self.cases = {}
        for row in rows:
            review_id = row.get("review_id", "")
            video = Path(row.get("video", ""))
            duration = float(row.get("duration_s", 0))
            candidate = float(row.get("candidate_offset_s", -1))
            if not RID.fullmatch(review_id) or not video.is_file() or not 0 <= candidate <= duration:
                raise ValueError(f"Invalid case {review_id}")
            if row.get("video_sha256") and file_sha256(video) != row["video_sha256"]:
                raise ValueError(f"Clip changed after packet preparation: {review_id}")
            self.cases[review_id] = {**row, "duration_s": duration,
                                     "candidate_offset_s": candidate, "video": str(video)}
        if len(self.cases) != 30:
            raise ValueError("Expected 30 unique cases")
        self.ids = sorted(self.cases)
        self.reviewer = safe_name(reviewer)
        self.out = Path(out) / self.reviewer
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest_hash = file_sha256(self.manifest_path)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()

    def state(self, review_id):
        path = self.out / f"{review_id}.latest.json"
        if not path.exists():
            return {"review_id": review_id, "revision": 0, "status": "not_started"}
        state = json.loads(path.read_text())
        if state.get("manifest_sha256") != self.manifest_hash:
            raise ValueError("Saved review belongs to another manifest")
        return state

    def progress(self):
        states = {rid: self.state(rid)["status"] for rid in self.ids}
        return {"schema_version": SCHEMA, "total": 30,
                "final": sum(value == "final" for value in states.values()), "states": states}

    def save(self, review_id, raw, final):
        clean = validate(raw, review_id, self.cases[review_id]["duration_s"], final)
        with self.lock:
            current = self.state(review_id)
            if raw["revision"] != current["revision"]:
                raise RuntimeError("This case changed in another tab; reload")
            clean.update(schema_version=SCHEMA, revision=current["revision"] + 1,
                         status="final" if final else "draft", reviewer_id=self.reviewer,
                         manifest_sha256=self.manifest_hash,
                         saved_at=dt.datetime.now(dt.timezone.utc).isoformat())
            if final:
                atomic_json(self.out / f"{review_id}.final.{clean['revision']:05d}.json", clean)
            atomic_json(self.out / f"{review_id}.latest.json", clean)
            atomic_json(self.out / "progress.json", self.progress())
        return clean

    def export(self):
        rows = [self.state(rid) for rid in self.ids if self.state(rid)["status"] == "final"]
        stream = io.StringIO(newline="")
        fields = REVIEW_FIELDS + ["schema_version", "revision", "status", "manifest_sha256", "saved_at"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()


class ReviewHandler(Handler):
    store: Store

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            progress = self.store.progress()
            status_labels = {"not_started": "未开始", "draft": "草稿", "final": "已提交"}
            rows = "".join(
                f'<tr data-state="{progress["states"][rid]}"><td>{index:02d}</td>'
                f'<td><a href="/r/{rid}">{rid}</a></td>'
                f'<td>{status_labels[progress["states"][rid]]}</td></tr>'
                for index, rid in enumerate(self.store.ids, 1))
            page = INDEX.replace("__ROWS__", rows).replace("__FINAL__", str(progress["final"]))
            return self._html(page.encode())
        if path == "/api/progress":
            return self._json(200, self.store.progress())
        if path == "/api/export":
            body = self.store.export()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="graph_contract_review.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        match = re.fullmatch(r"/(r|clip|api/state)/(graph_review_[0-9]{4})", path)
        if not match or match[2] not in self.store.cases:
            return self.send_error(404)
        review_id = match[2]
        if match[1] == "clip":
            return self._video(Path(self.store.cases[review_id]["video"]))
        state = self.store.state(review_id)
        if match[1] == "api/state":
            return self._json(200, state)
        index = self.store.ids.index(review_id)
        case = self.store.cases[review_id]
        payload = {"review_id": review_id, "index": index + 1, "total": 30,
                   "previous": self.store.ids[index - 1] if index else None,
                   "next": self.store.ids[index + 1] if index < 29 else None,
                   "candidate_offset_s": case["candidate_offset_s"],
                   "duration_s": case["duration_s"], "state": state,
                   "token": self.store.token}
        source = PAGE.replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))
        return self._html(source.encode())

    def do_POST(self):
        match = re.fullmatch(r"/api/(draft|final)/(graph_review_[0-9]{4})",
                             urlparse(self.path).path)
        if not match or match[2] not in self.store.cases:
            return self._json(404, {"error": "Unknown case"})
        if self.headers.get("X-Review-Token") != self.store.token:
            return self._json(403, {"error": "Invalid session token; reload"})
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._json(403, {"error": "Cross-origin submission rejected"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError("Invalid request size")
            raw = json.loads(self.rfile.read(length))
            value = self.store.save(match[2], raw, match[1] == "final")
            return self._json(200, {"revision": value["revision"], "status": value["status"]})
        except RuntimeError as exc:
            return self._json(409, {"error": str(exc)})
        except (ValueError, UnicodeError) as exc:
            return self._json(400, {"error": str(exc)})


INDEX = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>动作关系复核</title><style>body{margin:0;background:#151717;color:#eee;font:16px/1.5 system-ui}header,main{max-width:850px;margin:auto;padding:20px}a{color:#82d4bd;text-decoration:none}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #3f4442;text-align:left}</style>
<header><b>动作关系复核</b><span style="float:right">已提交 __FINAL__ / 30</span></header><main><a href="/api/export">导出已提交结果</a><table><thead><tr><th>序号</th><th>片段</th><th>状态</th></tr></thead><tbody>__ROWS__</tbody></table></main></html>"""


PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>动作关系复核</title><style>
:root{color-scheme:dark;--bg:#151717;--panel:#202322;--line:#4a504e;--ink:#eef1ef;--muted:#b9c4bf;--accent:#7acdb3}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui;letter-spacing:0}a{color:var(--accent);text-decoration:none}header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center}#status{margin-left:auto;color:var(--muted)}main{max-width:1320px;margin:auto;padding:18px}.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(380px,1fr);gap:24px}video{width:100%;aspect-ratio:4/3;object-fit:contain;background:#000;max-height:68vh}.tools,.footer,.nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding-top:10px}button,select,input,textarea{font:inherit;color:var(--ink);background:#282d2b;border:1px solid #626a67;border-radius:4px;padding:9px}button{cursor:pointer;min-height:44px}button:hover{background:#363d3a}button:focus-visible,select:focus-visible,input:focus-visible,textarea:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:2px}.rules{border-left:3px solid #e5b76a;padding:2px 0 2px 12px;margin-bottom:20px}.rules strong{display:block}.rules ul{margin:6px 0 0;padding-left:20px;color:var(--muted);font-size:15px}.step{border:0;border-top:1px solid var(--line);padding:15px 0;margin:0}.step legend{font-size:17px;font-weight:700;padding:0 0 8px}.step>label{display:block;margin-bottom:6px}.relations{display:grid;grid-template-columns:1fr;gap:7px}.relation{padding:9px 11px;border:1px solid var(--line);border-radius:4px;min-height:54px;display:flex;gap:11px;align-items:flex-start;cursor:pointer}.relation input{margin:5px 0 0;flex:none}.relation span{display:block;min-width:0}.relation b,.relation small{display:block}.relation small{font-size:14px;color:var(--muted);line-height:1.4}.relation:has(input:checked){background:#25473b;border-color:var(--accent)}.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field{display:flex;flex-direction:column;gap:5px;min-width:0}.field label{font-size:15px;font-weight:600}.wide{grid-column:1/-1}.hidden{display:none!important}.rare,.optional{margin-top:10px;color:var(--muted)}.rare .relations,.optional .fields{margin-top:9px}.hint{font-size:14px;color:var(--muted);margin:0}.required:after{content:' · 必填';color:#ffb5a8;font-size:12px;font-weight:500}.evidence-grid{display:grid;gap:10px}.evidence-grid label{display:grid;gap:5px;font-weight:600}.evidence-grid input{width:100%}.sentence-preview{margin:12px 0 0;padding:10px 12px;border-left:3px solid var(--accent);background:#202624;color:var(--muted)}.sentence-preview strong{color:var(--ink)}.time{display:grid;grid-template-columns:1fr auto 1fr auto;gap:5px;align-items:end}.time input{min-width:0}.footer{border-top:1px solid var(--line);margin-top:12px}.footer #submit{margin-left:auto;background:#28634b}.nav{justify-content:space-between;border-top:1px solid var(--line);margin-top:16px}.mono{font-family:ui-monospace,monospace;color:var(--muted)}#alert{color:#ffaea3;min-height:24px}#alert:not(:empty){padding:8px 0}.optional summary,.rare summary{cursor:pointer}@media(max-width:850px){.grid{grid-template-columns:1fr}video{max-height:48vh}}@media(max-width:480px){header{flex-wrap:wrap}.fields{grid-template-columns:1fr}.wide{grid-column:auto}.time{grid-template-columns:1fr auto}video{max-height:42vh}.footer #submit{width:100%;margin-left:0}}
</style><header><a href="/">全部片段</a><b>动作关系复核</b><span id="status" aria-live="polite"></span></header><main><div class="grid"><section><video id="video" controls preload="metadata" aria-label="待复核视频"></video><div class="tools"><button type="button" id="candidate">跳到标记处</button><button type="button" data-speed="1">1x</button><button type="button" data-speed="2">2x</button><span class="mono" id="clock"></span></div></section><form id="form"><div class="rules" aria-labelledby="rules_title"><strong id="rules_title">判断时记住三条硬规则</strong><ul><li>只看眼前要完成的事，不靠动作快慢或物品名称猜。</li><li>松手、换手、停顿、方向改变，本身不算换了一件事。</li><li>说“重新做一次”前，必须看到上一次已完成、放弃或重置。</li></ul></div><fieldset class="step"><legend class="required">1. 标记的位置发生了什么？</legend><div class="relations">
<label class="relation"><input type="radio" name="relation" value="CONTINUE"><span><b>继续同一件事</b><small>目标没变，也没有看到上一次结束。</small></span></label>
<label class="relation"><input type="radio" name="relation" value="SAME_ACTION_NEW_INSTANCE"><span><b>同一种事，重新做一次</b><small>上一次已完成、放弃或重置，又开始同类目标。</small></span></label>
<label class="relation"><input type="radio" name="relation" value="NEW_ACTION"><span><b>换了另一种事</b><small>眼前要达成的结果变了，例如拧紧变成拧松。</small></span></label>
<label class="relation"><input type="radio" name="relation" value="BOUNDARY_TYPE_UNRESOLVED"><span><b>确定开始新一段，但看不清是不是同一种事</b><small>能看到新的目标开始，却看不清与上一段是否一样。</small></span></label>
<label class="relation"><input type="radio" name="relation" value="UNOBSERVABLE"><span><b>看不清有没有换</b><small>连是否出现新的一段都无法判断。</small></span></label>
</div><details class="rare"><summary>特殊情况：只看到结束，或视频从中途开始/结束</summary><div class="relations"><label class="relation"><input type="radio" name="relation" value="END_ONLY"><span><b>只看到结束</b><small>上一件事结束，之后暂时没有新目标。</small></span></label><label class="relation"><input type="radio" name="relation" value="INITIAL"><span><b>视频开始时已在做</b><small>缺少开始之前的画面。</small></span></label><label class="relation"><input type="radio" name="relation" value="TERMINAL"><span><b>视频结束时仍在做</b><small>缺少之后的画面。</small></span></label></div></details></fieldset>
<fieldset class="step"><legend>2. 必须确认</legend><p class="hint">下面只显示与你刚才选择有关的问题。</p><div class="fields">
<div class="field wide hidden" id="prior_field"><label class="required" for="prior_episode_status">前一件事后来怎样了？</label><select id="prior_episode_status"></select><p class="hint">“同时进行”表示前一件事还在继续，不能当成已经结束。</p></div>
<div class="field wide hidden" id="return_field"><label class="required" for="return_to_prior_type">这是回到更早做过的那种事吗？</label><select id="return_to_prior_type"><option value="">请选择</option><option value="no">不是</option><option value="yes">是</option><option value="unknown">看不清</option></select></div>
<div class="field wide hidden" id="instance_field"><label class="required" for="instance_link">回来的这一次，是接着做，还是重新开始？</label><select id="instance_link"><option value="">请选择</option><option value="resumed_instance">接着做之前没做完的那一次</option><option value="new_instance">开始全新的一次</option><option value="unknown">看不清</option></select></div>
<div class="field wide hidden" id="boundary_field"><label class="required" for="boundary_start_s">变化发生在视频的第几秒？</label><div class="time"><input id="boundary_start_s" type="number" min="0" step="0.1" placeholder="开始秒数" aria-label="变化开始秒数"><button type="button" data-set="boundary_start_s">取当前时间</button><input id="boundary_end_s" type="number" min="0" step="0.1" placeholder="结束秒数" aria-label="变化结束秒数，可不填"><button type="button" data-set="boundary_end_s">取当前时间</button></div><p class="hint">很明确就只填开始；看不准就填一个起止范围。</p></div>
<div class="field wide hidden" id="transition_field"><label class="required" for="transition_shape">这个变化是怎样发生的？</label><select id="transition_shape"><option value="">请选择</option><option value="sharp">一下子变了</option><option value="gradual">慢慢变了</option><option value="uncertain">看不清</option></select></div>
<div class="field wide"><label class="required" for="visibility">关键画面看得清吗？</label><select id="visibility"><option value="">请选择</option><option value="visible">看得清</option><option value="partial">只能看清一部分</option><option value="not_visible">关键画面没拍到</option></select></div>
</div><details class="optional"><summary>可选线索：手有没有一直碰着、有没有松开</summary><div class="fields"><div class="field"><label for="continuous_contact">一直有接触？</label><select id="continuous_contact"><option value="">未记录</option><option value="yes">是</option><option value="no">否</option><option value="uncertain">看不清</option></select></div><div class="field"><label for="release_observed">看到松手？</label><select id="release_observed"><option value="">未记录</option><option value="yes">是</option><option value="no">否</option><option value="uncertain">看不清</option></select></div></div></details></fieldset>
<fieldset class="step"><legend>3. 填完四个空</legend><div class="evidence-grid">
<label for="evidence_before"><span class="required">1. 之前在做什么？</span><input id="evidence_before" maxlength="250" placeholder="例如：拧开盖子"></label>
<label for="evidence_after"><span class="required">2. 之后在做什么？</span><input id="evidence_after" maxlength="250" placeholder="例如：拧紧盖子"></label>
<label for="evidence_cue"><span class="required">3. 你看到了什么关键画面？</span><input id="evidence_cue" maxlength="500" placeholder="例如：盖子已经打开，手开始反向旋转"></label>
<label for="evidence_judgment"><span class="required">4. 所以你判断什么？</span><input id="evidence_judgment" maxlength="250" list="judgment_options" placeholder="例如：换了另一种事"></label>
</div><datalist id="judgment_options"><option value="继续同一件事"><option value="同一种事，重新做一次"><option value="换了另一种事"><option value="确定开始新一段，但看不清是不是同一种事"><option value="看不清有没有换"></datalist>
<input type="hidden" id="observable_evidence"><p class="sentence-preview"><strong>整句话：</strong><span id="evidence_preview">填写后会自动合成。</span></p></fieldset>
<div class="footer"><label class="required"><input type="checkbox" id="human_confirmed"> 我已看完整段视频</label><button type="button" id="draft">保存草稿</button><button id="submit">提交本条</button></div><div id="alert" role="alert"></div></form></div><div class="nav"><button id="previous">上一条</button><span id="count"></span><button id="next">下一条</button></div></main><script>
const D=__PAYLOAD__, $=s=>document.querySelector(s), v=$('#video'), form=$('#form');
const fields=['prior_episode_status','return_to_prior_type','instance_link',
  'boundary_start_s','boundary_end_s','transition_shape','continuous_contact',
  'release_observed','visibility','observable_evidence'];
const evidenceIds=['evidence_before','evidence_after','evidence_cue','evidence_judgment'];
let revision=D.state.revision, dirty=false, pending=Promise.resolve(), timer=null;
v.src='/clip/'+D.review_id;
$('#count').textContent=`${D.index} / ${D.total}`;
for(const k of fields) if(D.state[k]!==undefined) $('#'+k).value=D.state[k];
if(D.state.relation){
  const x=document.querySelector(`[name=relation][value="${D.state.relation}"]`);
  if(x) x.checked=true;
  if(['END_ONLY','INITIAL','TERMINAL'].includes(D.state.relation))$('.rare').open=true;
}
$('#human_confirmed').checked=D.state.human_confirmed===true;
function status(value){$('#status').textContent=value}
status({not_started:'未开始',draft:'草稿已保存',final:'已提交'}[D.state.status]||D.state.status);
const priorChoices={
  SAME_ACTION_NEW_INSTANCE:[['','请选择'],['completed','已经做完'],['terminated','已放弃或重新开始']],
  NEW_ACTION:[['','请选择'],['completed','已经做完'],['terminated','已停止或放弃'],['suspended','暂时停下，可能接着做'],['coactive','两件事同时在做'],['unknown','看不清']],
  BOUNDARY_TYPE_UNRESOLVED:[['','请选择'],['completed','已经做完'],['terminated','已停止或放弃'],['suspended','暂时停下，可能接着做'],['coactive','两件事同时在做'],['unknown','看不清']],
  END_ONLY:[['','请选择'],['completed','已经做完'],['terminated','已停止或放弃'],['unknown','看不清']]
};
function setChoices(select, choices){
  const current=select.value;
  select.replaceChildren(...choices.map(([value,label])=>{
    const option=document.createElement('option');option.value=value;option.textContent=label;return option;
  }));
  if(choices.some(([value])=>value===current))select.value=current;
}
function selectedRelation(){const x=document.querySelector('[name=relation]:checked');return x?x.value:''}
function renderQuestions(){
  const relation=selectedRelation(), hasPrior=Object.prototype.hasOwnProperty.call(priorChoices,relation);
  $('#prior_field').classList.toggle('hidden',!hasPrior);
  if(hasPrior)setChoices($('#prior_episode_status'),priorChoices[relation]);
  const asksReturn=relation==='NEW_ACTION';
  $('#return_field').classList.toggle('hidden',!asksReturn);
  $('#instance_field').classList.toggle('hidden',!(asksReturn&&$('#return_to_prior_type').value==='yes'));
  const hasBoundary=['SAME_ACTION_NEW_INSTANCE','NEW_ACTION','BOUNDARY_TYPE_UNRESOLVED','END_ONLY'].includes(relation);
  $('#boundary_field').classList.toggle('hidden',!hasBoundary);
  const overlap=['NEW_ACTION','BOUNDARY_TYPE_UNRESOLVED'].includes(relation)&&$('#prior_episode_status').value==='coactive';
  $('#transition_field').classList.toggle('hidden',!hasBoundary||overlap);
}
renderQuestions();
if(D.state.prior_episode_status)$('#prior_episode_status').value=D.state.prior_episode_status;
renderQuestions();
function evidenceText(){
  const values=evidenceIds.map(id=>$('#'+id).value.trim());
  if(!values.some(Boolean))return '';
  return `之前在${values[0]}；之后在${values[1]}；因为看到${values[2]}，所以我判断${values[3]}。`;
}
function renderEvidence(){
  const text=evidenceText();
  $('#observable_evidence').value=text;
  $('#evidence_preview').textContent=text||'填写后会自动合成。';
}
function restoreEvidence(text){
  if(!text)return;
  const match=text.match(/^之前在(.*?)；之后在(.*?)；因为看到(.*?)，所以我判断(.*?)。?$/);
  if(match)evidenceIds.forEach((id,index)=>$('#'+id).value=match[index+1]);
  else $('#evidence_cue').value=text;
  renderEvidence();
}
restoreEvidence(D.state.observable_evidence||'');
function body(){
  const relation=selectedRelation(), prior=$('#prior_episode_status').value;
  const result={review_id:D.review_id,relation,type_equivalence:'',prior_episode_status:'',
    next_episode_status:'',return_to_prior_type:'',instance_link:'',observed_type_path:'',
    boundary_start_s:'',boundary_end_s:'',transition_shape:'',overlap_present:'',
    continuous_contact:$('#continuous_contact').value||'uncertain',
    release_observed:$('#release_observed').value||'uncertain',visibility:$('#visibility').value,
    observable_evidence:evidenceText(),human_confirmed:$('#human_confirmed').checked};
  if(relation==='CONTINUE')Object.assign(result,{type_equivalence:'same',prior_episode_status:'ongoing',next_episode_status:'ongoing',return_to_prior_type:'not_applicable',instance_link:'same_instance',observed_type_path:'A',transition_shape:'no_transition',overlap_present:'no'});
  if(relation==='SAME_ACTION_NEW_INSTANCE')Object.assign(result,{type_equivalence:'same',prior_episode_status:prior,next_episode_status:'new',return_to_prior_type:'not_applicable',instance_link:'new_instance',observed_type_path:'A>A',transition_shape:$('#transition_shape').value,overlap_present:'no'});
  if(relation==='NEW_ACTION'){
    const returning=$('#return_to_prior_type').value, link=returning==='yes'?$('#instance_link').value:(returning==='no'?'new_instance':'unknown');
    Object.assign(result,{type_equivalence:'different',prior_episode_status:prior,return_to_prior_type:returning,instance_link:link,
      next_episode_status:link==='resumed_instance'?'resumed':(link==='unknown'?'unknown':'new'),
      observed_type_path:returning==='yes'?'A>B>A':(returning==='no'?'A>B':''),
      transition_shape:prior==='coactive'?'overlap':$('#transition_shape').value,
      overlap_present:prior==='coactive'?'yes':(prior==='unknown'?'uncertain':'no')});
  }
  if(relation==='BOUNDARY_TYPE_UNRESOLVED')Object.assign(result,{type_equivalence:'unknown',prior_episode_status:prior,next_episode_status:'new',return_to_prior_type:'unknown',instance_link:'unknown',
    transition_shape:prior==='coactive'?'overlap':$('#transition_shape').value,
    overlap_present:prior==='coactive'?'yes':(prior==='unknown'?'uncertain':'no')});
  if(relation==='UNOBSERVABLE')Object.assign(result,{type_equivalence:'unknown',prior_episode_status:'unknown',next_episode_status:'unknown',return_to_prior_type:'unknown',instance_link:'unknown',transition_shape:'uncertain',overlap_present:'uncertain'});
  if(relation==='END_ONLY')Object.assign(result,{type_equivalence:'not_applicable',prior_episode_status:prior,next_episode_status:'not_applicable',return_to_prior_type:'not_applicable',instance_link:'not_applicable',transition_shape:$('#transition_shape').value,overlap_present:'no'});
  if(relation==='INITIAL'||relation==='TERMINAL')Object.assign(result,{type_equivalence:'not_applicable',prior_episode_status:'not_applicable',next_episode_status:'not_applicable',return_to_prior_type:'not_applicable',instance_link:'not_applicable',transition_shape:'no_transition',overlap_present:'no'});
  if(['SAME_ACTION_NEW_INSTANCE','NEW_ACTION','BOUNDARY_TYPE_UNRESOLVED','END_ONLY'].includes(relation)){
    result.boundary_start_s=$('#boundary_start_s').value;result.boundary_end_s=$('#boundary_end_s').value;
  }
  return result;
}
function validateBeforeFinal(){
  const relation=selectedRelation();
  function missing(target,message){
    $('#alert').textContent=message;$(target).focus();return false;
  }
  if(!relation)return missing('[name=relation]','请先选择标记的位置发生了什么。');
  if(!$('#prior_field').classList.contains('hidden')&&!$('#prior_episode_status').value)
    return missing('#prior_episode_status','请选择前一件事后来怎样了。');
  if(!$('#return_field').classList.contains('hidden')&&!$('#return_to_prior_type').value)
    return missing('#return_to_prior_type','请确认是否回到更早做过的那种事。');
  if(!$('#instance_field').classList.contains('hidden')&&!$('#instance_link').value)
    return missing('#instance_link','请确认回来后是接着做，还是重新开始。');
  if(!$('#boundary_field').classList.contains('hidden')){
    const start=$('#boundary_start_s').value,end=$('#boundary_end_s').value;
    if(!start)return missing('#boundary_start_s','请标出变化开始的秒数。');
    if(+start>D.duration_s||(end&&(+end>D.duration_s||+end<+start)))
      return missing('#boundary_start_s','请检查起止秒数，必须在视频范围内且先后正确。');
  }
  if(!$('#transition_field').classList.contains('hidden')&&!$('#transition_shape').value)
    return missing('#transition_shape','请选择变化是一下子发生、慢慢发生，还是看不清。');
  if(!$('#visibility').value)return missing('#visibility','请选择关键画面是否看得清。');
  const evidencePrompts=['请填写“之前在做什么”。','请填写“之后在做什么”。',
    '请填写你看到的关键画面。','请填写你的最终判断。'];
  for(let index=0;index<evidenceIds.length;index++){
    const target='#'+evidenceIds[index];
    if(!$(target).value.trim())return missing(target,evidencePrompts[index]);
  }
  if(!$('#human_confirmed').checked)return missing('#human_confirmed','请看完整段视频后勾选确认。');
  $('#alert').textContent='';return true;
}
function save(final=false){
  const snapshot=body();
  status('保存中…');
  const operation=pending.catch(()=>{}).then(async()=>{
    const response=await fetch(`/api/${final?'final':'draft'}/${D.review_id}`,{
      method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':D.token},
      body:JSON.stringify({...snapshot,revision})});
    const result=await response.json();
    if(!response.ok) throw Error(result.error||'Save failed');
    revision=result.revision;
    if(JSON.stringify(body())===JSON.stringify(snapshot)) dirty=false;
    status(result.status==='final'?'已提交':'草稿已保存');
    $('#alert').textContent='';
  });
  pending=operation;
  operation.catch(error=>{$('#alert').textContent=error.message;status('未保存')});
  return operation;
}
form.oninput=event=>{
  renderQuestions();
  if(evidenceIds.includes(event.target.id))renderEvidence();
  dirty=true;
  if(event.target.id!=='human_confirmed') $('#human_confirmed').checked=false;
  status('有未保存修改');
  clearTimeout(timer);
  timer=setTimeout(()=>save(false),700);
};
$('#draft').onclick=()=>{clearTimeout(timer);save(false)};
form.onsubmit=async event=>{
  event.preventDefault();clearTimeout(timer);
  if(!validateBeforeFinal())return;
  $('#submit').disabled=true;
  try{await save(true)}catch(error){}finally{$('#submit').disabled=false}
};
$('#clock').textContent=`标记处 ${D.candidate_offset_s.toFixed(1)} 秒`;
v.ontimeupdate=()=>{$('#clock').textContent=`${v.currentTime.toFixed(1)} 秒 / 标记处 ${D.candidate_offset_s.toFixed(1)} 秒`};
$('#candidate').onclick=()=>{v.currentTime=D.candidate_offset_s;v.pause()};
document.querySelectorAll('[data-speed]').forEach(x=>x.onclick=()=>v.playbackRate=+x.dataset.speed);
document.querySelectorAll('[data-set]').forEach(x=>x.onclick=()=>{
  $('#'+x.dataset.set).value=v.currentTime.toFixed(1);
  form.dispatchEvent(new Event('input'));
});
async function nav(id){
  if(!id)return;
  clearTimeout(timer);
  try{if(dirty)await save(false);else await pending;location.href='/r/'+id}
  catch(error){$('#alert').textContent=error.message}
}
$('#previous').disabled=!D.previous;$('#next').disabled=!D.next;
$('#previous').onclick=()=>nav(D.previous);$('#next').onclick=()=>nav(D.next);
window.onbeforeunload=event=>{if(dirty){event.preventDefault();event.returnValue=''}};
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    store = Store(args.manifest, args.out, args.reviewer)
    handler = type("BoundGraphContractHandler", (ReviewHandler,), {"store": store})
    server = AuditServer(("127.0.0.1", args.port), handler)
    print(f"graph_contract_review_ready http://127.0.0.1:{args.port}/ reviewer={store.reviewer}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
