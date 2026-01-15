# Simple Image Boundary Area Layer (SIBAL)

SIBAL is a learning-free, rule-based region proposal layer for object detection, implemented in Rust and wgpu.  
The project is currently in an experimental stage.

SIBAL is **not a full object detector**. It focuses on generating bounding box proposals from image boundary signals using deterministic GPU compute pipelines.

---

## What it is

SIBAL explores how far a learning-free, rule-based vision pipeline can go on modern GPUs.

Given an input image, SIBAL produces a set of bounding box proposals based on edge-derived activity, without any training, model weights, or learned representations.

- No classification
- No training
- Deterministic, rule-based processing
- Designed as a **region proposal layer**, not a detector

---

## Key Properties

- **Learning-free**  
  No training, datasets, or learned parameters are involved.

- **Rule-based**  
  All decisions are made using explicit, hand-designed rules.

- **GPU-native (wgpu)**  
  The pipeline is designed from the ground up as a GPU compute workflow using wgpu.  
  It runs as a sequence of GPU compute passes rather than CPU-driven preprocessing.

- **Experimental**  
  APIs, parameters, and behaviors may change as the project evolves.

---

## Pipeline Overview

SIBAL processes images through multiple stages, each operating on increasingly structured representations.

### L0 – Directional Edge Extraction
- Computes pixel differences independently for R, G, and B channels
- Extracts directional boundary signals per pixel

### L1 – Cell-level Activity Aggregation
- Aggregates L0 directional activations
- Groups **2×2 pixels into a single cell**
- Computes per-cell statistics such as:
  - total pixel count
  - active pixel count

### L2 – Spatial Pooling
- Applies max pooling over L1 cells
- Produces a coarse activity map
- Reduces spatial resolution and downstream computation cost

### L3 – Initial Seed Generation
- Generates initial region seeds from pooled activity
- Forms the core set of region proposal candidates

### L4 – Bounding Box Expansion (Experimental)
- Expands initial seeds into bounding boxes
- Merging and expansion policies are under active experimentation

> Currently, **L0–L3 are considered the core pipeline**, while **L4 is experimental**.

---

## Running

```bash
cd Simple_image_boundary_area_layer
cargo build --release
./target/release/Simple_image_boundary_area_layer
