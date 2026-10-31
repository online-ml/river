# Track

A track evaluate a model's performance.

The following metrics are recorded:

- Time, which should be interpreted with wisdom. Indeed time can depend on the architecture

  and local resource situations. Comparison via FLOPS should be preferred. - The model's memory footprint.
- The model's predictive performance on the track's dataset.

## Parameters

- **name**

  *Type* → `str`

  The name of the track.
- **datasets**

  The datasets that compose the track.
- **metric**

  The metric(s) used to track performance.

## Methods

> [!NOTE]
>
> **run**
