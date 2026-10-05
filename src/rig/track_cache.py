"""Versioned cache for the expensive detection/tracking half of a render.

The cache boundary is deliberately before offline semantic decisions.  A
collector writes the detector boxes, stable tracker ids, smoothed ownership
scores and face proposals once.  Replays can then change track-level
handness/identity decisions and rendering without loading or rerunning the
detector, tracker, or frame ownership model.
"""
from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import tempfile


SCHEMA = "semhand-track-cache"
SCHEMA_VERSION = 1


def source_manifest(paths):
    """Return stable, cheap source identities suitable for cache validation."""
    out = {}
    for name, path in sorted(paths.items()):
        absolute = os.path.abspath(path)
        item = {"path": absolute}
        try:
            stat = os.stat(absolute)
        except OSError:
            item["missing"] = True
        else:
            item.update({"size": int(stat.st_size),
                         "mtime_ns": int(stat.st_mtime_ns)})
        out[str(name)] = item
    return out


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot store {type(value).__name__} in a track cache")


def _fingerprint(value):
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mismatch(actual, expected, prefix="manifest"):
    """Return the first recursive manifest mismatch, or ``None``."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{prefix}: expected an object"
        for key, value in expected.items():
            if key not in actual:
                return f"{prefix}.{key}: missing"
            found = _mismatch(actual[key], value, f"{prefix}.{key}")
            if found:
                return found
        return None
    if actual != expected:
        return f"{prefix}: cache has {actual!r}, run needs {expected!r}"
    return None


class Writer:
    """Atomically write one manifest, frame records, then a completion row."""

    def __init__(self, path, manifest):
        self.path = os.path.abspath(path)
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, self.tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(self.path) + ".", suffix=".tmp",
            dir=parent, text=True)
        self._fh = os.fdopen(fd, "w", encoding="utf-8")
        self.count = 0
        self.closed = False
        contract = dict(manifest)
        self.manifest = {
            "type": "manifest",
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "created_utc": _datetime.datetime.now(
                _datetime.timezone.utc).isoformat(),
            "contract_sha256": _fingerprint(contract),
            **contract,
        }
        self._write(self.manifest)

    def _write(self, row):
        self._fh.write(json.dumps(
            row, ensure_ascii=True, separators=(",", ":"),
            default=_json_default) + "\n")

    def write_frame(self, row):
        if self.closed:
            raise RuntimeError("track cache writer is closed")
        frame = dict(row)
        frame["type"] = "frame"
        self._write(frame)
        self.count += 1

    def close(self):
        if self.closed:
            return
        self._write({"type": "end", "frame_count": self.count})
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        os.replace(self.tmp_path, self.path)
        self.closed = True

    def abort(self):
        if self.closed:
            return
        self._fh.close()
        try:
            os.unlink(self.tmp_path)
        except FileNotFoundError:
            pass
        self.closed = True


class Reader:
    """Stream a completed cache and fail loudly on stale or partial data."""

    def __init__(self, path, expected=None):
        self.path = os.path.abspath(path)
        self._fh = open(self.path, encoding="utf-8")
        self.count = 0
        self.finished = False
        try:
            self.manifest = self._read_json("manifest")
            if self.manifest.get("type") != "manifest":
                raise ValueError(f"{self.path}: first row is not a manifest")
            if self.manifest.get("schema") != SCHEMA:
                raise ValueError(f"{self.path}: unsupported cache schema "
                                 f"{self.manifest.get('schema')!r}")
            if self.manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"{self.path}: cache schema version "
                    f"{self.manifest.get('schema_version')!r}; expected "
                    f"{SCHEMA_VERSION}")
            contract = {
                key: value for key, value in self.manifest.items()
                if key not in {"type", "schema", "schema_version",
                               "created_utc", "contract_sha256"}
            }
            if self.manifest.get("contract_sha256") != _fingerprint(contract):
                raise ValueError(f"{self.path}: manifest fingerprint mismatch")
            mismatch = _mismatch(self.manifest, expected or {})
            if mismatch:
                raise ValueError(
                    f"{self.path}: stale/incompatible track cache: "
                    f"{mismatch}")
        except Exception:
            self._fh.close()
            raise

    def _read_json(self, label):
        line = self._fh.readline()
        if not line:
            raise ValueError(f"{self.path}: incomplete cache; missing {label}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{self.path}: invalid JSON in {label}: "
                             f"{error}") from error

    def read_frame(self, expected_frame):
        if self.finished:
            raise ValueError(f"{self.path}: requested frame {expected_frame} "
                             "after the cache ended")
        row = self._read_json(f"frame {expected_frame}")
        if row.get("type") == "end":
            raise ValueError(f"{self.path}: cache ended before frame "
                             f"{expected_frame}")
        if row.get("type") != "frame":
            raise ValueError(f"{self.path}: expected a frame row, got "
                             f"{row.get('type')!r}")
        if int(row.get("frame", -1)) != int(expected_frame):
            raise ValueError(f"{self.path}: expected frame {expected_frame}, "
                             f"cache has {row.get('frame')!r}")
        self.count += 1
        return row

    def finish(self):
        if self.finished:
            return
        row = self._read_json("completion row")
        if row.get("type") != "end":
            raise ValueError(f"{self.path}: replay stopped before all cached "
                             "frames were consumed")
        if int(row.get("frame_count", -1)) != self.count:
            raise ValueError(f"{self.path}: completion row says "
                             f"{row.get('frame_count')} frames, read "
                             f"{self.count}")
        if self._fh.readline():
            raise ValueError(f"{self.path}: data follows the completion row")
        self._fh.close()
        self.finished = True

    def close(self):
        if not self._fh.closed:
            self._fh.close()
