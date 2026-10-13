# Procedure Group Contract v1

## Purpose

`procedure_group` is a parent layer over local action states. It supports
cross-video analysis without deleting local state IDs, changing time
boundaries, or treating different steps in one procedure as the same action.

## Unit definitions

- **Segment instance:** one annotated time interval.
- **Local state:** a reusable concrete action inside one video, such as
  `旋松并取下风扇头部`.
- **Procedure group:** an immediate goal-directed procedure containing one or
  more local states, such as `拆卸风扇`.

## Grouping rule

Two local states belong to the same procedure group only when their context is
compatible on all three axes:

1. **Goal:** they advance the same immediate task outcome.
2. **Object family:** they act on the same object or workflow object family.
3. **Result direction:** their outcomes do not contradict each other.

Different steps may share a procedure group. Their edge is an
`intra_group_transition`, not an exact self-loop. Installation and removal are
kept in separate groups when direction is central to the procedure.

## Edge types

- `exact_state_repeat`: the same local state repeats, whether represented by
  adjacent same-state segments or the within-segment repeat checkbox.
- `intra_group_transition`: two different local states belong to one procedure.
- `inter_group_transition`: the procedure group changes.
- A special-state endpoint removes the observed edge from the action graph.
  No replacement edge is synthesized across that gap.

## Special states

The exact labels `特殊事件`, `看不到内容`, `看不到动作`, `无内容`, `未开始`, and
`idle` are excluded from action nodes and procedure groups. Source timelines
remain frozen and retain those intervals for provenance.

## Conservative decisions

- Ambiguous actions are assigned to a narrow, context-specific group rather
  than merged into a broad group.
- Original and canonical descriptions are both retained.
- Medium-confidence nodes remain usable but are surfaced in QA.
- The same-agent review in v1 is an internal consistency check and must not be
  reported as human inter-annotator agreement.

## Examples

- `旋松并取下风扇头部` and `逐节拆下风扇支撑杆` belong to `拆卸风扇`.
- `前往另一房间取凳子并返回` belongs to `取用辅助物品`, not `拆卸风扇`.
- `用纸巾擦拭桌面` and `用抹布擦拭桌面` belong to `清洁表面`.
- `取下手机壳` and `装回手机壳` stay as different local states but share the
  broader procedure `装卸手机壳`.
- `炒菜` and `扫地` would be different procedure groups even if they occur in
  one continuous recording.
