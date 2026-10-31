# Project Documentation

These documents support collaborative reproduction of baseline methods from published papers.

- [`research_protocol.md`](research_protocol.md) - shared reproduction principles and workflow.
- [`dataset.md`](dataset.md) - dataset sources, splits, labels, and preparation notes.
- [`methods.md`](methods.md) - papers, baselines, implementations, and reproduction status.
- [`experiment_log.md`](experiment_log.md) - concise record of runs, results, and issues.

Keep project-wide guidance here. Put implementation-specific instructions in the relevant method directory.

## Updating method documentation

When a baseline reproduction begins, use [`methods.md`](methods.md) to add the method’s shared research summary and status. Follow the agreed method-entry template, update the status as work progresses, and keep detailed implementation notes in the method directory.

## Method-entry template

Copy this template into `methods.md` when the team begins documenting a method:

```markdown
## <Method name>

### Overview

- **Full name:**
- **Paper:**
- **Source:**
- **Year:**

### Mechanism

Briefly explain how the method works and its underlying idea.

### Implementation

- Backbone:
- Loss function:
- Training procedure:
- Special requirements:

### Experimental Settings

- Dataset:
- Batch size:
- Learning rate:
- Number of epochs:
- Optimizer:
- Random seeds:

### Evaluation

- Average accuracy:
- Worst-group accuracy:
- Other relevant metrics:

### Implementation Status

- Not started
- In progress
- Implemented
- Reproduced
- Verified

### Notes

Important implementation details, deviations from the original paper, and reproduction observations.
```
