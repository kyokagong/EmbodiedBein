# ADR 0002: 本地 CAS 与 TSDB 布局

- Status: accepted
- Date: 2026-08-21

## CAS

`EB_DATA_ROOT`（默认 `/var/lib/eb`）：

```
cas/<ab>/<sha256>     # 内容寻址；ab 为 digest 前两字符
runs/<run_id>/        # console.log, logs/, outputs/, configs/
worktrees/<run_id>/   # 源仓快照（只读挂进任务容器）
tsdb/<run_id>/<tag>.jsonl
```

`eb-artifact::put` 写入 `cas/ab/abcd...`。媒体 Range 由 ebd `/media/{sha256}` 提供。

## TSDB

采用 **JSONL**（非 length-prefixed）：每行 `{"step":0,"value":1.2}`。
tag 中的非字母数字替换为 `_` 作为文件名。控制面 `GET /runs/{id}/metrics` 只转发 ebd，不在 Python 热路径解析 tfevents。
