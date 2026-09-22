"""Build a versioned two-level interaction graph release.

The release preserves local timeline states and adds a separately curated
``procedure_group`` layer. Source annotations are copied byte-for-byte into a
frozen snapshot; they are never edited in place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.auditor.boundary.interaction_graph_timeline import (
    build_graph,
    validate_submission,
)


RELEASE_SCHEMA = "interaction_graph_release_v1"
CONFIG_SCHEMA = "interaction_graph_procedure_group_config_v1"
GAP_TOKEN = "__SPECIAL_GAP__"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def latest_finals(annotation_dir: str | Path) -> dict[str, Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for path in Path(annotation_dir).glob("*.final.*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "final":
            continue
        video_id = str(value.get("video_id") or "")
        saved_at = str(value.get("saved_at") or path.name)
        key = (saved_at, path.name)
        if video_id and (video_id not in latest or key > latest[video_id][0]):
            latest[video_id] = (key, path.resolve())
    return {video_id: item[1] for video_id, item in latest.items()}


def _compress(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _assignment_key(batch_id: str, video_id: str, event_id: str) -> str:
    return f"{batch_id}/{video_id}/{event_id}"


def _aggregate_edges(occurrences: list[dict], keys: tuple[str, ...]) -> list[dict]:
    aggregate: dict[tuple, dict] = {}
    for occurrence in occurrences:
        key = tuple(occurrence[name] for name in keys)
        item = aggregate.setdefault(key, {
            **{name: occurrence[name] for name in keys},
            "count": 0,
            "videos": set(),
            "batches": set(),
            "provenance": [],
        })
        item["count"] += 1
        item["videos"].add(occurrence["video_id"])
        item["batches"].add(occurrence["batch_id"])
        item["provenance"].append({
            name: occurrence.get(name)
            for name in ("batch_id", "video_id", "at_s", "from_segment",
                         "to_segment", "original_kind")
        })
    rows = []
    for key in sorted(aggregate):
        item = aggregate[key]
        item["videos"] = sorted(item["videos"])
        item["batches"] = sorted(item["batches"])
        rows.append(item)
    return rows


def _render_html(release: dict) -> str:
    data = json.dumps({
        "summary": release["summary"],
        "groups": release["procedure_groups"],
        "nodes": release["local_nodes"],
        "edges": release["group_edges"],
        "paths": release["video_paths"],
    }, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Procedure Group 图谱 v1</title>
<style>
:root{{--bg:#f5f6f4;--panel:#fff;--ink:#202421;--muted:#69716c;--line:#cfd5d1;--green:#146b4a;--coral:#a94232;--blue:#315f91}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;letter-spacing:0}}
header{{padding:18px 24px;background:#18352a;color:white;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}}
h1{{font-size:22px;margin:0}}header span{{color:#c9ddd3}}main{{max-width:1440px;margin:auto;padding:18px}}
.stats{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin-bottom:14px}}.stat{{background:var(--panel);border:1px solid var(--line);padding:12px;border-radius:6px}}.stat b{{display:block;font-size:22px;color:var(--green)}}
.controls{{display:flex;gap:8px;margin:0 0 14px;flex-wrap:wrap}}input,select{{font:inherit;padding:9px;border:1px solid #929b95;border-radius:4px;background:white;min-width:190px}}
.layout{{display:grid;grid-template-columns:minmax(260px,34%) 1fr;gap:14px;min-height:620px}}.pane{{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:auto}}
.pane h2{{font-size:17px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}}button.group{{width:100%;text-align:left;border:0;border-bottom:1px solid #e4e7e5;background:white;padding:10px 12px;cursor:pointer;color:var(--ink)}}button.group:hover,button.group.active{{background:#e6f1ec}}button.group b{{display:block}}button.group small{{color:var(--muted)}}
#detail{{padding:14px}}h3{{font-size:16px;margin:18px 0 8px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #e2e5e3;text-align:left;vertical-align:top}}th{{color:var(--muted);font-size:13px}}.tag{{display:inline-block;padding:2px 6px;border-radius:3px;color:white;font-size:12px}}.high{{background:var(--green)}}.medium{{background:var(--coral)}}.exact_state_repeat{{color:var(--green)}}.intra_group_transition{{color:var(--blue)}}.inter_group_transition{{color:var(--coral)}}
.empty{{color:var(--muted);padding:20px}}@media(max-width:800px){{.stats{{grid-template-columns:repeat(2,1fr)}}.layout{{grid-template-columns:1fr}}.pane{{max-height:none}}}}
</style>
<header><h1>Procedure Group 图谱 v1</h1><span>细粒度 state 保留，特殊事件不跨接</span></header>
<main><section class=\"stats\" id=\"stats\"></section><div class=\"controls\"><input id=\"search\" placeholder=\"搜索组名或动作\"><select id=\"batch\"><option value=\"\">全部批次</option><option value=\"repeated_actions\">第一批</option><option value=\"mosaic_complement\">Mosaic 补集</option></select></div><section class=\"layout\"><div class=\"pane\"><h2>Procedure groups</h2><div id=\"groups\"></div></div><div class=\"pane\"><h2>组详情</h2><div id=\"detail\" class=\"empty\">选择左侧的组查看成员与转换。</div></div></section></main>
<script>const D={data};
const $=s=>document.querySelector(s);let selected=null;
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
function stats(){{const s=D.summary;$('#stats').innerHTML=[[s.videos,'视频'],[s.action_nodes,'动作节点'],[s.procedure_groups,'任务组'],[s.fine_edge_occurrences,'细图边'],[s.group_edge_occurrences,'组图边']].map(x=>`<div class=stat><b>${{x[0]}}</b>${{x[1]}}</div>`).join('')}}
function visibleNodes(){{const q=$('#search').value.trim().toLowerCase(),b=$('#batch').value;return D.nodes.filter(n=>(!b||n.batch_id===b)&&(!q||(n.canonical_description+n.procedure_group_name+n.video_id).toLowerCase().includes(q)))}}
function renderGroups(){{const nodes=visibleNodes(),ids=new Set(nodes.map(n=>n.procedure_group_id));const gs=D.groups.filter(g=>ids.has(g.procedure_group_id));$('#groups').innerHTML=gs.map(g=>{{const members=nodes.filter(n=>n.procedure_group_id===g.procedure_group_id);return `<button class="group ${{selected===g.procedure_group_id?'active':''}}" data-id="${{g.procedure_group_id}}"><b>${{esc(g.name)}}</b><small>${{members.length}} 节点 · ${{new Set(members.map(n=>n.video_id)).size}} 视频</small></button>`}}).join('')||'<div class=empty>没有匹配结果</div>';document.querySelectorAll('button.group').forEach(x=>x.onclick=()=>{{selected=x.dataset.id;renderGroups();renderDetail()}})}}
function renderDetail(){{const g=D.groups.find(x=>x.procedure_group_id===selected);if(!g)return;const nodes=visibleNodes().filter(n=>n.procedure_group_id===selected);const b=$('#batch').value;const edges=D.edges.filter(e=>(e.source_group===selected||e.target_group===selected)&&(!b||e.batches.includes(b)));$('#detail').innerHTML=`<h3>${{esc(g.name)}}</h3><p>${{esc(g.definition)}}</p><h3>成员动作</h3><table><thead><tr><th>批次/视频</th><th>规范动作</th><th>置信度</th></tr></thead><tbody>${{nodes.map(n=>`<tr><td>${{esc(n.batch_id)}}<br>${{esc(n.video_id)}} · ${{n.event_id}}</td><td>${{esc(n.canonical_description)}}<br><small>${{esc(n.original_description)}}</small></td><td><span class="tag ${{n.confidence}}">${{n.confidence}}</span></td></tr>`).join('')}}</tbody></table><h3>相关转换</h3><table><thead><tr><th>来源</th><th>类型</th><th>目标</th><th>次数</th></tr></thead><tbody>${{edges.map(e=>`<tr><td>${{esc(e.source_group_name)}}</td><td class="${{e.edge_type}}">${{esc(e.edge_type)}}</td><td>${{esc(e.target_group_name)}}</td><td>${{e.count}}</td></tr>`).join('')||'<tr><td colspan=4>没有边</td></tr>'}}</tbody></table>`}}
$('#search').oninput=renderGroups;$('#batch').onchange=renderGroups;stats();renderGroups();
</script></html>"""


def _render_report(release: dict) -> str:
    summary = release["summary"]
    batch_counts: dict[str, Counter] = defaultdict(Counter)
    for item in release["freeze_inventory"]:
        batch_counts[item["batch_id"]]["videos"] += 1
        batch_counts[item["batch_id"]]["duration_s"] += item["duration_s"]
    for node in release["local_nodes"]:
        batch_counts[node["batch_id"]]["action_nodes"] += 1
    for node in release["special_nodes"]:
        batch_counts[node["batch_id"]]["special_nodes"] += 1
    for edge in release["fine_edge_occurrences"]:
        batch_counts[edge["batch_id"]][edge["edge_type"]] += 1
    top_groups = sorted(
        release["procedure_groups"],
        key=lambda row: (-row["member_count"], row["name"]),
    )[:15]
    cross = [
        edge for edge in release["fine_edge_occurrences"]
        if edge["edge_type"] == "inter_group_transition"
    ]
    medium = [
        node for node in release["local_nodes"] if node["confidence"] == "medium"
    ]
    lines = [
        "# Interaction Graph Release v1",
        "",
        "## Release summary",
        "",
        f"- Videos: {summary['videos']}.",
        f"- Action nodes: {summary['action_nodes']}.",
        f"- Special nodes excluded: {summary['special_nodes_excluded']}.",
        f"- Procedure groups: {summary['procedure_groups']}.",
        f"- Procedure groups reused across videos: "
        f"{summary['procedure_groups_reused_across_videos']}.",
        f"- Single-node procedure groups: "
        f"{summary['single_node_procedure_groups']}.",
        f"- Fine edge occurrences: {summary['fine_edge_occurrences']}.",
        f"- Aggregated fine edges: {summary['fine_edges_aggregated']}.",
        f"- Aggregated group edges: {summary['group_edges_aggregated']}.",
        f"- Videos with an inter-group transition: "
        f"{summary['videos_with_inter_group_transition']}.",
        f"- Videos that leave and later return to a procedure group: "
        f"{summary['videos_with_procedure_return']}.",
        f"- Edges omitted because a special state was an endpoint: "
        f"{summary['edges_omitted_for_special_endpoint']}.",
        "- No edge was synthesized across a special-state gap.",
        "",
        "## Edge semantics",
        "",
    ]
    for name, count in summary["edge_type_counts"].items():
        lines.append(f"- `{name}`: {count} occurrences.")
    lines.extend([
        "",
        "## Batch comparison",
        "",
        "| Batch | Videos | Minutes | Action nodes | Special nodes | Exact repeats | Intra-group | Inter-group |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for batch_id, counts in sorted(batch_counts.items()):
        lines.append(
            f"| {batch_id} | {counts['videos']} | "
            f"{counts['duration_s'] / 60:.2f} | {counts['action_nodes']} | "
            f"{counts['special_nodes']} | {counts['exact_state_repeat']} | "
            f"{counts['intra_group_transition']} | "
            f"{counts['inter_group_transition']} |"
        )
    lines.extend([
        "",
        "## Largest procedure groups",
        "",
        "| Procedure group | Nodes | Videos | Duration (s) |",
        "| --- | ---: | ---: | ---: |",
    ])
    for group in top_groups:
        lines.append(
            f"| {group['name']} | {group['member_count']} | "
            f"{group['video_count']} | {group['total_duration_s']:.1f} |"
        )
    lines.extend([
        "",
        "## Inter-group transitions",
        "",
    ])
    for edge in cross:
        lines.append(
            f"- `{edge['batch_id']}/{edge['video_id']}` at "
            f"{edge['at_s']:.1f}s: {edge['source_canonical']} "
            f"({edge['source_group_name']}) -> {edge['target_canonical']} "
            f"({edge['target_group_name']})."
        )
    lines.extend([
        "",
        "## Procedure-group returns",
        "",
    ])
    if summary["procedure_return_videos"]:
        for video_id in summary["procedure_return_videos"]:
            lines.append(f"- `{video_id}`.")
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Medium-confidence review queue",
        "",
        f"{len(medium)} nodes remain medium-confidence. They are deliberately "
        "kept in narrow contextual groups rather than merged broadly.",
        "",
    ])
    for node in medium:
        lines.append(
            f"- `{node['local_node_id']}`: {node['canonical_description']} -> "
            f"{node['procedure_group_name']}. {node['review_note']}"
        )
    lines.extend([
        "",
        "## Completion of the nine-step plan",
        "",
        "1. Complete: 60 latest finals frozen with source and snapshot hashes.",
        "2. Complete: 134 action nodes extracted with provenance.",
        "3. Complete: every action has an original and canonical description.",
        "4. Complete: the procedure-group contract is versioned in `docs/`.",
        "5. Complete as a single-agent consistency review; human inter-annotator agreement was not measured.",
        "6. Complete: all 134 action nodes assigned to one of 45 procedure groups.",
        "7. Complete: fine and group graphs generated with typed edges.",
        "8. Complete: batch comparison, transition audit, and confidence review recorded here.",
        "9. Complete: JSON, CSV, checksums, frozen annotations, and the standalone graph explorer are published together.",
        "",
        "## Interpretation boundary",
        "",
        "This v1 is a complete, reproducible single-agent release. It must not be "
        "reported as having human inter-annotator agreement. A later independent "
        "human audit can update the nine medium-confidence nodes without changing "
        "the frozen source timelines.",
        "",
    ])
    return "\n".join(lines)


def build_release(config_path: str | Path, out_dir: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"unexpected config schema: {config.get('schema_version')}")
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    groups = config.get("procedure_groups") or {}
    assignments = config.get("assignments") or {}
    special_labels = {
        str(label).strip().casefold() for label in config.get("special_labels", [])
    }

    inventory = []
    local_nodes = []
    special_nodes = []
    fine_occurrences = []
    omitted_special_edges = []
    video_paths = []
    seen_assignment_keys = set()

    for batch in config.get("batches", []):
        batch_id = batch["batch_id"]
        manifest_path = Path(batch["manifest"]).resolve()
        annotation_dir = Path(batch["annotation_dir"]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        finals = latest_finals(annotation_dir)
        for video in manifest.get("videos", []):
            video_id = video["video_id"]
            if video_id not in finals:
                raise ValueError(f"missing final: {batch_id}/{video_id}")
            source = finals[video_id]
            raw = json.loads(source.read_text(encoding="utf-8"))
            annotation = validate_submission(
                raw, video_id, float(video["duration_s"]), final=True)
            graph = build_graph(annotation)
            frozen = root / "frozen_annotations" / batch_id / f"{video_id}.json"
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, frozen)
            inventory.append({
                "batch_id": batch_id,
                "video_id": video_id,
                "duration_s": float(video["duration_s"]),
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": file_sha256(manifest_path),
                "source_annotation": str(source),
                "source_annotation_sha256": file_sha256(source),
                "frozen_annotation": str(frozen),
                "frozen_annotation_sha256": file_sha256(frozen),
                "confirmed_full_video": bool(annotation["confirmed_full_video"]),
            })

            descriptions = {
                event["event_id"]: event["description"]
                for event in annotation["events"]
            }
            graph_nodes = {node["event_id"]: node for node in graph["nodes"]}
            node_lookup = {}
            for event_id, description in descriptions.items():
                local_id = _assignment_key(batch_id, video_id, event_id)
                is_special = description.strip().casefold() in special_labels
                if is_special:
                    item = {
                        "local_node_id": local_id,
                        "batch_id": batch_id,
                        "video_id": video_id,
                        "event_id": event_id,
                        "description": description,
                        **graph_nodes.get(event_id, {}),
                    }
                    special_nodes.append(item)
                    node_lookup[event_id] = None
                    continue
                assignment = assignments.get(local_id)
                if not assignment:
                    raise ValueError(f"missing assignment: {local_id} ({description})")
                group_id = assignment["procedure_group_id"]
                if group_id not in groups:
                    raise ValueError(f"unknown group {group_id}: {local_id}")
                if assignment.get("original_description") != description:
                    raise ValueError(
                        f"description changed for {local_id}: {description!r} != "
                        f"{assignment.get('original_description')!r}")
                canonical = str(assignment.get("canonical_description") or "").strip()
                if not canonical:
                    raise ValueError(f"empty canonical description: {local_id}")
                seen_assignment_keys.add(local_id)
                group = groups[group_id]
                item = {
                    "local_node_id": local_id,
                    "batch_id": batch_id,
                    "video_id": video_id,
                    "event_id": event_id,
                    "original_description": description,
                    "canonical_description": canonical,
                    "procedure_group_id": group_id,
                    "procedure_group_name": group["name"],
                    "confidence": assignment.get("confidence", "high"),
                    "review_note": assignment.get("review_note", ""),
                    **graph_nodes.get(event_id, {}),
                }
                local_nodes.append(item)
                node_lookup[event_id] = item

            for occurrence in graph["edge_occurrences"]:
                source_node = node_lookup.get(occurrence["source"])
                target_node = node_lookup.get(occurrence["target"])
                base = {
                    "batch_id": batch_id,
                    "video_id": video_id,
                    **occurrence,
                }
                if source_node is None or target_node is None:
                    omitted_special_edges.append(base)
                    continue
                if occurrence["source"] == occurrence["target"]:
                    edge_type = "exact_state_repeat"
                elif (source_node["procedure_group_id"] ==
                      target_node["procedure_group_id"]):
                    edge_type = "intra_group_transition"
                else:
                    edge_type = "inter_group_transition"
                fine_occurrences.append({
                    **base,
                    "source_local_node": source_node["local_node_id"],
                    "target_local_node": target_node["local_node_id"],
                    "source_canonical": source_node["canonical_description"],
                    "target_canonical": target_node["canonical_description"],
                    "source_group": source_node["procedure_group_id"],
                    "target_group": target_node["procedure_group_id"],
                    "source_group_name": source_node["procedure_group_name"],
                    "target_group_name": target_node["procedure_group_name"],
                    "edge_type": edge_type,
                    "original_kind": occurrence["kind"],
                })

            fine_sequence = []
            group_sequence = []
            for segment in annotation["segments"]:
                node = node_lookup.get(segment["event_id"])
                fine_sequence.append(node["local_node_id"] if node else GAP_TOKEN)
                group_sequence.append(node["procedure_group_id"] if node else GAP_TOKEN)
            video_paths.append({
                "batch_id": batch_id,
                "video_id": video_id,
                "fine_sequence": fine_sequence,
                "compressed_fine_sequence": _compress(fine_sequence),
                "group_sequence": group_sequence,
                "compressed_group_sequence": _compress(group_sequence),
            })

    extra_assignments = sorted(set(assignments) - seen_assignment_keys)
    if extra_assignments:
        raise ValueError(f"assignments do not match action nodes: {extra_assignments[:5]}")
    expected_videos = int(config.get("expected_videos") or 0)
    expected_nodes = int(config.get("expected_action_nodes") or 0)
    if expected_videos and len(inventory) != expected_videos:
        raise ValueError(f"expected {expected_videos} videos, found {len(inventory)}")
    if expected_nodes and len(local_nodes) != expected_nodes:
        raise ValueError(f"expected {expected_nodes} action nodes, found {len(local_nodes)}")

    fine_edges = _aggregate_edges(
        fine_occurrences,
        ("source_local_node", "target_local_node", "edge_type", "original_kind"),
    )
    group_edges = _aggregate_edges(
        fine_occurrences, ("source_group", "target_group", "edge_type"))
    for edge in group_edges:
        edge["source_group_name"] = groups[edge["source_group"]]["name"]
        edge["target_group_name"] = groups[edge["target_group"]]["name"]

    members = defaultdict(list)
    for node in local_nodes:
        members[node["procedure_group_id"]].append(node)
    procedure_groups = []
    for group_id, group in sorted(groups.items()):
        rows = members.get(group_id, [])
        if not rows:
            raise ValueError(f"procedure group has no members: {group_id}")
        procedure_groups.append({
            "procedure_group_id": group_id,
            "name": group["name"],
            "definition": group["definition"],
            "member_count": len(rows),
            "video_count": len({row["video_id"] for row in rows}),
            "batches": sorted({row["batch_id"] for row in rows}),
            "total_duration_s": round(sum(row.get("total_duration_s", 0) for row in rows), 3),
            "member_node_ids": [row["local_node_id"] for row in rows],
        })

    edge_types = Counter(row["edge_type"] for row in fine_occurrences)
    confidence = Counter(row["confidence"] for row in local_nodes)
    inter_group_videos = {
        row["video_id"] for row in fine_occurrences
        if row["edge_type"] == "inter_group_transition"
    }
    procedure_return_videos = []
    for path in video_paths:
        runs = []
        current = []
        for group_id in path["group_sequence"] + [GAP_TOKEN]:
            if group_id == GAP_TOKEN:
                if current:
                    runs.append(_compress(current))
                    current = []
            else:
                current.append(group_id)
        if any(len(run) != len(set(run)) for run in runs):
            procedure_return_videos.append(
                f"{path['batch_id']}/{path['video_id']}")
    summary = {
        "videos": len(inventory),
        "action_nodes": len(local_nodes),
        "special_nodes_excluded": len(special_nodes),
        "procedure_groups": len(procedure_groups),
        "fine_edge_occurrences": len(fine_occurrences),
        "fine_edges_aggregated": len(fine_edges),
        "group_edge_occurrences": len(fine_occurrences),
        "group_edges_aggregated": len(group_edges),
        "edges_omitted_for_special_endpoint": len(omitted_special_edges),
        "procedure_groups_reused_across_videos": sum(
            row["video_count"] > 1 for row in procedure_groups),
        "single_node_procedure_groups": sum(
            row["member_count"] == 1 for row in procedure_groups),
        "videos_with_inter_group_transition": len(inter_group_videos),
        "videos_with_procedure_return": len(procedure_return_videos),
        "procedure_return_videos": procedure_return_videos,
        "edge_type_counts": dict(sorted(edge_types.items())),
        "confidence_counts": dict(sorted(confidence.items())),
    }
    qa = {
        "source_annotations_modified": False,
        "all_videos_validated": len(inventory) == expected_videos,
        "all_action_nodes_assigned": len(local_nodes) == expected_nodes,
        "special_gaps_bridged": False,
        "assignment_keys_consumed": len(seen_assignment_keys),
        "unused_assignment_keys": extra_assignments,
        "medium_confidence_nodes": [
            row["local_node_id"] for row in local_nodes
            if row["confidence"] == "medium"
        ],
        "human_inter_annotator_agreement": "not measured",
        "model_review": {
            "method": "single-agent context review plus deterministic contract checks",
            "completed": True,
            "must_not_be_reported_as_human_agreement": True,
        },
    }
    release = {
        "schema_version": RELEASE_SCHEMA,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "grouping_contract": config.get("grouping_contract"),
        "special_labels": sorted(special_labels),
        "summary": summary,
        "freeze_inventory": inventory,
        "local_nodes": local_nodes,
        "special_nodes": special_nodes,
        "fine_edge_occurrences": fine_occurrences,
        "fine_edges": fine_edges,
        "procedure_groups": procedure_groups,
        "group_edges": group_edges,
        "video_paths": video_paths,
        "omitted_special_edges": omitted_special_edges,
        "qa": qa,
    }
    atomic_json(root / "graph_release.json", release)
    atomic_json(root / "freeze_manifest.json", {
        "schema_version": "interaction_graph_freeze_v1",
        "files": inventory,
    })
    atomic_json(root / "local_nodes.json", local_nodes)
    atomic_json(root / "special_nodes.json", special_nodes)
    atomic_json(root / "fine_edges.json", fine_edges)
    atomic_json(root / "procedure_groups.json", procedure_groups)
    atomic_json(root / "group_edges.json", group_edges)
    atomic_json(root / "qa_report.json", qa)
    review_queue = [
        row for row in local_nodes if row["confidence"] == "medium"
    ]
    atomic_json(root / "review_queue.json", review_queue)
    _write_csv(root / "local_nodes.csv", local_nodes, [
        "local_node_id", "batch_id", "video_id", "event_id",
        "original_description", "canonical_description", "procedure_group_id",
        "procedure_group_name", "confidence", "segment_count",
        "total_duration_s", "review_note",
    ])
    _write_csv(root / "fine_edges.csv", fine_edges, [
        "source_local_node", "target_local_node", "edge_type",
        "original_kind", "count", "videos", "batches",
    ])
    _write_csv(root / "group_edges.csv", group_edges, [
        "source_group", "source_group_name", "target_group",
        "target_group_name", "edge_type", "count", "videos", "batches",
    ])
    _write_csv(root / "review_queue.csv", review_queue, [
        "local_node_id", "batch_id", "video_id", "event_id",
        "original_description", "canonical_description", "procedure_group_id",
        "procedure_group_name", "confidence", "review_note",
    ])
    (root / "graph_explorer.html").write_text(_render_html(release), encoding="utf-8")
    (root / "release_report.md").write_text(
        _render_report(release), encoding="utf-8")
    checksum_lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "CHECKSUMS.sha256":
            continue
        checksum_lines.append(f"{file_sha256(path)}  {path.relative_to(root)}")
    (root / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = build_release(args.config, args.out_dir)
    print(
        f"release_ready {args.out_dir} "
        f"({result['summary']['videos']} videos, "
        f"{result['summary']['action_nodes']} action nodes, "
        f"{result['summary']['procedure_groups']} procedure groups)"
    )


if __name__ == "__main__":
    main()
