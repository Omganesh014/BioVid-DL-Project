# CRITICAL: USE THE ACTUAL BIOVID DATA FROM MY LOCAL DRIVE

The BioVid dataset is already available on my Linux system at:

`/run/media/acernitro/Ventoy/DB`

**You MUST work from the actual files at this location.**

Do NOT create a hypothetical BioVid dataset.

Do NOT invent filenames.

Do NOT assume the directory structure.

Do NOT create fake metadata.

Do NOT use a random example dataset.

Do NOT download another copy of BioVid unless explicitly instructed.

The first step is to inspect the actual contents of:

`/run/media/acernitro/Ventoy/DB`

and understand exactly how my downloaded BioVid data is organized.

---

# STEP 1 — INSPECT THE ACTUAL DIRECTORY

Start by running filesystem inspection commands equivalent to:

```bash
ls -lah /run/media/acernitro/Ventoy/DB
```

Then recursively inspect the structure:

```bash
find /run/media/acernitro/Ventoy/DB -maxdepth 2 -type d
```

and:

```bash
find /run/media/acernitro/Ventoy/DB -maxdepth 3 -type f | head -200
```

Also determine:

- total size
- number of files
- number of directories
- file extensions
- video files
- CSV files
- metadata files
- README/documentation files
- MATLAB files
- signal files
- any ZIP/RAR archives
- any labels/annotation files

Use appropriate commands such as:

```bash
du -sh /run/media/acernitro/Ventoy/DB
```

and:

```bash
find /run/media/acernitro/Ventoy/DB -type f | sed 's/.*\.//' | sort | uniq -c
```

The purpose is to discover the REAL dataset structure.

---

# STEP 2 — DO NOT ASSUME WHETHER THIS IS PART A, B, C, D OR E

Determine exactly which BioVid part is present.

The official BioVid documentation describes several parts.

Part A contains:

- frontal video
- GSR/EDA
- ECG
- EMG from trapezius
- approximately 8,700 samples
- 87 subjects
- five classes
- 20 samples per class per subject

The official documentation also describes Part B, C, D and E separately.

Therefore:

**Do not simply assume that `/DB` is Part A.**

Inspect the files and determine what is actually present.

If the directory contains multiple BioVid parts, identify them separately.

---

# STEP 3 — FIND THE VIDEO DATA

Locate all video files.

For example, search for extensions such as:

```bash
find /run/media/acernitro/Ventoy/DB -type f \
\( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \)
```

Determine:

- how many videos exist
- how they are organized
- which directory corresponds to each subject
- how filenames identify trials/samples
- video codec
- FPS
- resolution
- duration
- number of frames

Inspect representative videos using Python/OpenCV or another reliable method.

For example, determine:

```text
FPS
frame_count
width
height
duration
codec
```

Do this for multiple videos, not just one.

---

# STEP 4 — FIND THE METADATA

Search for files such as:

```text
Samples.csv
metadata.csv
labels.csv
*.csv
*.mat
*.xlsx
*.txt
*.json
```

Do not assume that `Samples.csv` exists.

If it exists, inspect its actual columns.

Determine whether it contains fields such as:

```text
subject_id
subject_name
sample_name
class_id
temperature
trial
```

or equivalent fields.

Print:

- column names
- number of rows
- first several rows
- unique subjects
- unique labels
- number of samples per subject
- number of samples per label

Do NOT modify the original metadata.

---

# STEP 5 — VERIFY THE BIOVID LABELS FROM THE ACTUAL FILES

This is extremely important.

Do not assume:

```text
0 = baseline
1 = pain level 1
2 = pain level 2
3 = pain level 3
4 = pain level 4
```

until the actual metadata confirms it.

If the metadata confirms this mapping, explicitly document it.

If the metadata uses another mapping, use the actual mapping.

Create an explicit mapping such as:

```python
LABEL_MAP = {
    ...
}
```

and print it before training.

The code must stop if an unexpected label is encountered.

---

# STEP 6 — DETERMINE HOW A VIDEO MAPS TO A LABEL

For at least 10 randomly selected samples, trace:

```text
subject
↓
trial/sample ID
↓
video filename
↓
metadata row
↓
label
```

Verify that the relationship is correct.

For example:

```text
Subject X
Trial Y
Video X_Y.mp4
Metadata row X_Y
Class = Z
```

Do not assume the filename itself is the label.

The mapping must be verified from the actual dataset.

---

# STEP 7 — DETERMINE THE ACTUAL VIDEO FORMAT

For several videos, inspect:

```text
resolution
FPS
frame count
duration
color channels
codec
```

The original BioVid video recordings are described as being recorded at 25 Hz, and Part A samples are short time windows. However, **the actual files in `/run/media/acernitro/Ventoy/DB` take precedence over general documentation**.

Therefore compare:

DOCUMENTED DATA
vs
ACTUAL DATA

and report any differences.

---

# STEP 8 — UNDERSTAND THE TEMPORAL STRUCTURE

Do NOT immediately extract one frame per video.

Our objective is a **ViViT video model**.

ViViT needs temporal information.

Therefore determine:

```text
How many frames does each BioVid video contain?
```

If a video is approximately 5.5 seconds and approximately 25 FPS, we may expect roughly 137–138 frames, but this MUST be measured from the actual files.

Then determine an appropriate temporal sampling strategy.

For example:

```text
Full video
↓
sample N frames
↓
temporal clip
↓
ViViT
```

Do not blindly select N = 16.

Compare feasible choices such as:

- 8 frames
- 16 frames
- 32 frames

considering:

- temporal information
- ViViT architecture
- RTX 4050 6GB VRAM
- training speed
- memory usage

Then select a configuration and explain why.

---

# STEP 9 — DETERMINE WHETHER FACE CROPPING IS REQUIRED

Inspect the actual video frames.

Determine whether the videos contain:

- full face
- upper body
- background
- multiple people
- significant head movement

Do not automatically crop the face unless there is a reason.

If face detection/cropping is used, explicitly document:

```text
original frame
↓
face detector
↓
face bounding box
↓
crop
↓
resize
↓
ViViT
```

Also consider what happens if face detection fails.

The code must NOT silently produce a black/empty/incorrect frame.

---

# STEP 10 — BUILD THE BIOVID DATASET MANIFEST

After inspection, generate a manifest containing at minimum:

```text
subject_id
trial_id
video_path
label
fps
frame_count
duration
```

Additional metadata may be included.

Save it as something like:

```text
biovid_manifest.csv
```

The manifest must be generated from the actual files.

Do not manually type the manifest.

---

# STEP 11 — SUBJECT-LEVEL SPLIT

Use the manifest to create:

```text
train
validation
test
```

splits at the SUBJECT level.

Never split individual frames randomly.

Never split clips randomly.

Never allow clips from the same subject to appear across train and test.

Verify:

```python
assert len(train_subjects & val_subjects) == 0
assert len(train_subjects & test_subjects) == 0
assert len(val_subjects & test_subjects) == 0
```

Also check duplicate trial IDs.

The program must stop if leakage is detected.

---

# STEP 12 — INSPECT THE DATA VISUALLY

Before training ViViT, select representative samples from every class.

Create a visualization showing sampled frames from several BioVid videos.

For example:

```text
Class 0
Video
Frame 1 | Frame 2 | Frame 3 | ... | Frame N

Class 1
Video
Frame 1 | Frame 2 | Frame 3 | ... | Frame N

...
```

This allows us to verify that:

- videos are loaded correctly
- frames are in the correct order
- labels correspond to the correct videos
- face information is present
- preprocessing is not destroying useful information

---

# STEP 13 — ONLY AFTER DATA INSPECTION DESIGN ViViT INPUT

Now determine the actual ViViT input.

For example:

```text
[B, T, C, H, W]
```

where:

B = batch size
T = temporal frames
C = 3 RGB channels
H = image height
W = image width

A possible configuration may be:

```text
B = small batch
T = 16
C = 3
H = 224
W = 224
```

BUT this is only an example.

The actual final values must be selected after:

1. inspecting the videos
2. inspecting the selected ViViT implementation
3. checking GPU memory

---

# STEP 14 — IMPLEMENT ViViT

After the dataset has been understood, implement the selected ViViT architecture.

The implementation must explicitly define:

```text
number of frames
image size
patch size
tubelet size
embedding dimension
number of transformer layers
attention heads
MLP dimension
dropout
number of classes
```

The exact architecture must match the actual implementation being used.

Do not mix implementations.

---

# STEP 15 — TRACE ONE VIDEO THROUGH ViViT

Before training, take ONE actual BioVid video from:

`/run/media/acernitro/Ventoy/DB`

and trace it through the entire pipeline.

Show:

```text
Original video
↓
frame count
↓
sampled frame indices
↓
sampled frames
↓
resize
↓
normalization
↓
tensor shape
↓
tubelet/patch embedding
↓
token count
↓
Transformer
↓
representation
↓
classification head
↓
output
```

Print the shape at every major step.

This is mandatory.

---

# STEP 16 — VERIFY THE MODEL WITH ONE REAL SAMPLE

Before full training:

1. Load one actual BioVid video.
2. Process it.
3. Pass it through ViViT.
4. Verify the output shape.
5. Calculate the loss.
6. Run backward propagation.
7. Verify gradients.
8. Check for NaN/Inf.
9. Verify that the label is valid.

Only then proceed to batch training.

---

# STEP 17 — MEMORY TEST ON RTX 4050

Run a GPU memory test.

Determine whether the selected configuration fits in approximately 6 GB VRAM.

If it does not:

1. reduce batch size
2. enable mixed precision
3. use gradient accumulation
4. reduce temporal frames
5. use a smaller model
6. freeze some pretrained layers

Do not silently modify the configuration.

Every change must be printed and documented.

---

# STEP 18 — TRAINING

After all validation passes, train the model using the actual BioVid data.

Save:

```text
best_model.pt
last_model.pt
config.yaml
label_mapping.json
training_history.csv
```

Also record:

```text
GPU
PyTorch version
CUDA version
model architecture
dataset size
train subjects
validation subjects
test subjects
number of frames
image resolution
batch size
learning rate
optimizer
scheduler
```

---

# STEP 19 — EVALUATION

Evaluate ONLY on previously unseen subjects.

Report:

- accuracy
- precision
- recall
- F1
- macro F1
- weighted F1
- confusion matrix
- per-class metrics

Also preserve:

```text
subject_id
trial_id
true_label
predicted_label
confidence
```

for error analysis.

---

# STEP 20 — MOST IMPORTANT: NEVER HIDE A DATASET PROBLEM

If you discover that:

- the dataset is incomplete
- videos are missing
- metadata is missing
- labels don't match expectations
- filenames don't match metadata
- videos have unexpected dimensions
- subjects cannot be identified
- the mounted path does not contain BioVid
- files are corrupted
- multiple dataset versions are mixed together

STOP.

Do not "fix" it silently.

Explain exactly what is wrong and what information is required.

---

# FINAL OBJECTIVE

The final system should become:

```text
/run/media/acernitro/Ventoy/DB
             │
             ▼
     ACTUAL BIOVID DATA
             │
             ▼
      DATASET INSPECTION
             │
             ▼
       METADATA MANIFEST
             │
             ▼
    SUBJECT-LEVEL SPLITTING
             │
             ▼
       VIDEO LOADING
             │
             ▼
      TEMPORAL SAMPLING
             │
             ▼
      FRAME PREPROCESSING
             │
             ▼
     [B,T,C,H,W] TENSOR
             │
             ▼
           ViViT
             │
             ▼
   TRANSFORMER REPRESENTATION
             │
             ▼
       CLASSIFICATION HEAD
             │
             ▼
       PAIN PREDICTION
             │
             ▼
          EVALUATION
```

The implementation must be based on the **actual files located at `/run/media/acernitro/Ventoy/DB`**, not a hypothetical BioVid directory.

The priority order is:

**1. Correct dataset understanding**

**2. Correct label mapping**

**3. No data leakage**

**4. Correct temporal processing**

**5. Correct ViViT tensor dimensions**

**6. Correct model implementation**

**7. GPU/memory safety**

**8. Reproducibility**

**9. Complete evaluation**

**10. No silent mistakes**

Do not proceed from one stage to the next until the previous stage has been verified.