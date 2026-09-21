# Dataset validation report

- Loaded 50 rows from data/raw/metadata.csv
- OK: all required columns present
- OK: every row's image_path exists on disk

### Images per (location, floor)

| location | floor | count |
|---|---|---|
| xerox | 2 | 1 ⚠️ low count |
| botany garden | 0 | 1 ⚠️ low count |
| xaviers hall gate | 0 | 1 ⚠️ low count |
| department of biotechnology | 0 | 1 ⚠️ low count |
| girls washroom | 0 | 1 ⚠️ low count |
| washroom | 0 | 1 ⚠️ low count |
| hostel building entrance | 0 | 1 ⚠️ low count |
| volleyball court | 0 | 1 ⚠️ low count |
| staircase 2 | 0 | 1 ⚠️ low count |
| staircase | 0 | 1 ⚠️ low count |
| xaviers hostel wing 2 | 0 | 2 ⚠️ low count |
| woods | 0 | 2 ⚠️ low count |
| basketball | 0 | 2 ⚠️ low count |
| library | 2 | 2 ⚠️ low count |
| hostel building | 0 | 2 ⚠️ low count |
| exit gate | 0 | 2 ⚠️ low count |
| xaviers institute of communications | 0 | 3 ⚠️ low count |
| main gate 1 | 0 | 3 ⚠️ low count |
| chapel | 0 | 4 ⚠️ low count |
| office building | 0 | 4 ⚠️ low count |
| canteen | 0 | 6 ⚠️ low count |
| xaviers hall | 0 | 8 |

- **WARNING**: 21 (location, floor) pairs have fewer than 8 images. These are unlikely to generalize well.

### Sessions per location

| location | distinct sessions |
|---|---|
| basketball | 1 |
| xaviers hostel wing 2 | 1 |
| xaviers hall gate | 1 |
| woods | 1 |
| washroom | 1 |
| volleyball court | 1 |
| staircase 2 | 1 |
| staircase | 1 |
| office building | 1 |
| xaviers institute of communications | 1 |
| main gate 1 | 1 |
| hostel building entrance | 1 |
| hostel building | 1 |
| girls washroom | 1 |
| exit gate | 1 |
| department of biotechnology | 1 |
| chapel | 1 |
| canteen | 1 |
| botany garden | 1 |
| library | 1 |
| xerox | 1 |
| xaviers hall | 2 |

- **WARNING**: these locations have only 1 capture session, so a clean session-based test split is not possible for them: ['basketball', 'xaviers hostel wing 2', 'xaviers hall gate', 'woods', 'washroom', 'volleyball court', 'staircase 2', 'staircase', 'office building', 'xaviers institute of communications', 'main gate 1', 'hostel building entrance', 'hostel building', 'girls washroom', 'exit gate', 'department of biotechnology', 'chapel', 'canteen', 'botany garden', 'library', 'xerox']. They will fall back to a stratified random split in Task 3, which is weaker evidence of generalization — flag this in the final report.

### Near-duplicate detection (perceptual hash)

- OK: no near-duplicate images detected above threshold