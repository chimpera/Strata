# Shared conversation snapshots: convergence proposal

Status: proposed interface and development work for issues/PRs #41, #52 and #57.
This is not an agreed integration-owner assignment, a merged upstream feature,
or a claim that the disk tier already uses this core. Development starts at
upstream `b38c183` (0.1.18); the deployed/tested 0.1.15 branch remains separate.

## Boundaries

1. **Snapshot core:** own the in-memory representation, checked sizing,
   capture, complete non-mutating validation, and restore into existing session
   allocations. Validate all layers, running state and checkpoints before the
   first destination write. No cache policy, filesystem I/O or client markers.
2. **RAM policy:** longest compatible exact prefix, active wins ties, bounded
   payload bytes and slots, LRU eviction. Incoming/outgoing exchange payloads
   count together. Off by default. A failed admission is a normal cache miss.
3. **Disk adapter (separate contribution):** use the same state representation,
   but do not serialize C++ structs, pointers or native vector layouts. Define a
   versioned envelope and bounded streaming/staging explicitly before integration.

The serve loop selects a resume point and publishes new token/image/checkpoint
metadata only after successful restore. It must not infer that a failed transfer
left the old session usable.

## State checklist

| State | Representation or reconstruction |
|---|---|
| Token/image identity and steering | Exact IDs, image identity/grid hashes, immutable engine steering configuration plus per-request enabled state |
| GDN | Recurrences and convolution history, in live state and every retained checkpoint |
| PLE | Normalized history; previous token window reconstructed from the checkpoint IDs |
| QSA KV | Used, page-rounded identity-layout cells, all codes/scales for FP16/INT8/Q4; authoritative host pools for streamed layers |
| Indexer | Completed pooled rows, raw tail, per-sequence spare key (`idx_dead`), and block-position metadata; rebuild the spare pooled row at the selected resume point |
| MTP | Used authoritative draft KV; refill its resident ring at the selected resume point |
| Residency | Invalidate replaceable streamed maps; keep graph-bound addresses stable; do not treat outgoing VRAM slots as authoritative |
| Compatibility | Runtime geometry, layout and buffer extents; persistent identity additionally binds exact weights/tokenizer/steering/schema |

The first #57 implementation omitted `idx_dead` and block-position metadata;
fixtures with identical opening tokens can mask the spare-key omission. Include
distinct spare keys in the component regression and different opening prefixes
in future model gates. Do not describe earlier state hashes as coverage of these
previously unmeasured fields. Scratch recomputed before use (attention scores,
step/position uploads, graph intermediates) is not part of the persisted state.

## Failure contract

- Invalid snapshot: reject before CUDA synchronization/copies or destination
  writes; discard that entry and retain a valid active prefix or perform normal
  clean prefill. Do not discard the outgoing state before validating the incoming
  candidate.
- Allocation/admission failure: skip parking, preserving active inference.
- Transfer/synchronization failure: return a distinct fatal outcome. The caller
  reports an error and exits so the server can restart on the next request. A
  future clean-reset recovery needs its own proof that the CUDA context remains
  usable; validation does **not** make GPU failures transactional.
- Resume metadata is committed only after successful state application.

## Memory admission

Keep logical byte/slot capacity separate from physical-memory headroom. Proposed
flag: `--conversation-cache-min-free-mib` (default 2560, matching #41's starting
floor). Before allocating, require known available physical memory sufficient
for the estimated new payload plus that floor. Check again after capture; if the
floor is no longer met, discard the temporary image. Telemetry already includes
incoming images and parked allocations: do not subtract them a second time.

Use `GlobalMemoryStatusEx` on Windows and `MemAvailable` on Linux. Unknown or
malformed telemetry declines parking. These are instantaneous admission samples,
not a reservation against other processes. Host telemetry alone does not enforce
cgroup/job limits; container-aware limits and Windows runtime verification need
review. Disabled caching must not sample telemetry or allocate snapshots.

## Disk interface questions for review

- A spill consumes an evicted immutable snapshot. If persistence fails, should
  eviction still drop it, or may policy retain it within the existing bound?
- A promoted snapshot must reserve its staging bytes before reading. Either use
  an explicitly bounded whole-image reservation or a validated streaming format;
  a digest pass alone is not safe if the file can change before the apply pass.
- Bind model/weights, tokenizer, all geometry/KV layout, control vectors and
  scales, image identity, and schema. The current RAM-only steering boolean is
  not a persistent model identity.
- Use atomic publication and reject truncated/corrupt/foreign snapshots. Specify
  local permissions, retention/deletion and privacy: disk contains conversation
  tokens and model state, unlike the current RAM-only feature.
- Eviction-only spilling does not promise that the latest active turn survives
  a crash. Decide whether shutdown/periodic persistence is a separate opt-in.

## Acceptance and remaining work

Use #57's existing component/parity harnesses as a starting point, not a complete
acceptance contract. Add CPU tests for unknown/low memory, arithmetic boundaries,
late-layer and checkpoint rejection before mutation, PLE mismatch, zero-QSA and
256/512-expert geometries; GPU round trips with distinct indexer spare keys;
transfer-failure tests; and a dry-run-by-default known-answer 30-cycle soak.

Full-model testing must cover histories beyond resident KV at several lengths,
including near the configured limit, and compare exact answers plus the restored
state. Add real-encoder image smoke separately from synthetic embeddings.
Persistence/restart/corruption/staging tests belong with the disk adapter.
A controlled real Pi/Hermes task comparison still needs reproducible task inputs,
independent success validation, timings and peak memory. No existing synthetic
benchmark substitutes for that evidence.

No unified PR should claim the whole contract until these gates and the
collaborators' interface/ownership review are complete.
