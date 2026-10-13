# Semhand track-cache architecture

## Data flow

```text
source video
  -> detect hands/faces
  -> associate tracker ids
  -> frame owner model + OwnHold
  -> versioned track cache + base CSV
       -> conservative raw tid -> canonical tid map
       -> offline hand/not-hand, ownership, and side decisions
       -> cache replay -> decision-applied CSV
            -> offline temporal owner gate, aggregated by canonical tid
            -> cache replay -> final CSV/video
```

Detection, tracking, and the frame ownership model run only in the collector.
Both later renders decode the source video and rebuild masks, but obtain boxes,
track ids, smoothed owner scores, coasted boxes, and face proposals from the
same cache.

## Cache boundary

Each frame is stored immediately after `OwnHold` and before track-level
handness decisions, the temporal owner gate, and the final owner cap. A frame
record contains:

- raw detections and tracked detections;
- tracker ids, provenance, and track age;
- smoothed pre-semantic owner flags and cap diagnostics;
- foreign coasting boxes used across detector dropouts;
- face proposals after the hand-overlap veto;
- duplicate and no-id diagnostic counts.

This boundary lets semantic decisions change without changing track identity.
Face hold, hand masks, suppression, annotations, CSV measurement, and video
encoding remain replay-time operations.

## Validity rules

The JSONL cache is written atomically and contains a schema version, a SHA-256
manifest fingerprint, source file identities, upstream model provenance, and
all upstream configuration values. Replay fails closed when the source,
requested model configuration, frame range, camera, or tracking configuration
does not match. A completion record prevents a partial collector run from
being mistaken for a valid cache.

Replay does not require detector or classifier checkpoint files to remain
installed. Their concrete identities are retained as provenance in the cache;
their requested arguments are validated during replay.

## Offline identity and semantic policy

Raw tracker ids are immutable. `rig.track_fusion` writes a separate canonical
mapping and an audit table for every candidate link. It accepts only a short,
spatially continuous, unambiguous handoff with no strong side conflict. Votes
for hand/not-hand, ownership, and handedness use the canonical id, then expand
back to raw ids for deterministic replay.

The temporal owner gate also aggregates raw fragments by canonical id. A
context consensus admits a track. Stable whole-track owner evidence can rescue
an insufficient context sample or a context rejection; the output records both
evidence states and the exact decision reason. `max_owner=2` remains the final
anatomical cap after semantic eligibility.
