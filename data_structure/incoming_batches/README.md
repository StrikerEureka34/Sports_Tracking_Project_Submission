# Incoming Batches (Full Dataset - 40,010 frames)
This directory contains batched frames organized for parallel processing.

## Structure
```
incoming_batches/
├── batch_0000/    (500 frames each)
│   ├── frame_00000.jpg
│   ├── frame_00001.jpg
│   └── ...
├── batch_0001/
├── ...
└── batch_0081/    (82 batches total)
```

## Stats
- Total frames: 40,010
- Batch size: 500 frames
- Total batches: 82
- Frame dimensions: 1280x720

## Generation
Batches are created using `local_preprocessing/split_batches.py`

**Note:** Actual batch files not included due to size constraints (~15GB).
