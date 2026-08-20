# Stage 7: Immich integration

## Responsibility boundary

Immich owns uploads, originals, albums, timeline state and the user-facing
gallery. Memora owns derived AI data and algorithms. The integration uses the
public REST API instead of reading the Immich database or storage layout.

```text
Immich image asset
  -> POST /search/metadata (asset metadata and UUID)
  -> GET /assets/{id}/thumbnail?size=preview
  -> local private thumbnail cache
  -> OpenCLIP / pHash / quality scoring
  -> Memora index (PhotoRecord.id == Immich asset UUID)
  -> search, people, events, similar groups, best shot
  -> Immich UUIDs returned to the UI or exported to an Immich album
```

Videos are deliberately excluded from this first integration because the
current Memora encoders operate on still images.

## Configuration

Create a user-scoped API key in Immich. The complete workflow needs these
permissions:

- `asset.read` for metadata discovery;
- `asset.view` for preview thumbnails;
- `album.create` when exporting an AI selection as a new album.

Using an all-permissions key is convenient during local development, but a
scoped key is preferred for deployment.

```text
MEMORA_IMMICH_URL=http://immich-server:2283
MEMORA_IMMICH_API_KEY=<secret>
MEMORA_IMMICH_CACHE_PATH=data/immich-cache
MEMORA_IMMICH_TIMEOUT_SECONDS=30
```

The URL may be either the server root or end in `/api`; the client normalizes
both forms. The API key is sent only in the `x-api-key` request header and is
never returned by Memora endpoints.

## Initial and incremental sync

```powershell
memora immich-status
memora immich-sync --encoder open_clip --index-path data/index.json
```

Equivalent API calls:

```powershell
Invoke-RestMethod http://localhost:8000/immich/status
Invoke-RestMethod http://localhost:8000/immich/sync `
  -Method Post -ContentType application/json `
  -Body '{"page_size":250,"force":false,"prune_missing":false}'
```

The sync response reports `remote_count`, `indexed_count`, `reused_count`,
`removed_count`, `failed_count` and per-asset sanitized errors. A failed
refresh retains the previous record. Normal sync also retains assets no longer
returned by Immich; opt into index pruning with `prune_missing`. Pruning affects
only the Memora JSON index and does not delete the cached file or any Immich
asset.

Set `force` after changing `MEMORA_ENCODER`, the OpenCLIP model or pretrained
weights so every embedding is rebuilt with one consistent vector space.

## Frontend contract

For a synced result, `POST /search` includes:

```json
{
  "photo_id": "an-immich-asset-uuid",
  "source": "immich",
  "immich_asset_id": "an-immich-asset-uuid",
  "thumbnail_url": "/immich/assets/an-immich-asset-uuid/thumbnail",
  "score": 0.82
}
```

The frontend should use `thumbnail_url` rather than embedding the Immich API
key in a browser URL. Events, people and similar-shot groups already contain
the same `photo_id` values, so they can use the same thumbnail route.

To publish a Memora selection into the normal Immich UI:

```http
POST /immich/albums
Content-Type: application/json

{
  "album_name": "Memora - Best of the beach trip",
  "photo_ids": ["uuid-1", "uuid-2"],
  "description": "Best shots selected by Memora AI"
}
```

Memora rejects IDs that are not currently known as synced Immich assets.

## Operational notes

- `data/immich-cache/` is private derived data and is ignored by Git.
- Keep `data/` on a persistent Docker volume.
- Do not expose the Memora API publicly without adding application-level
  authentication; the thumbnail and album routes act with the configured
  Immich user's authority.
- Sync is explicit rather than automatic. A scheduler can call
  `POST /immich/sync`; repeated calls are incremental and idempotent.
