"""Dataset-agnostic port of the strictly nested DPE post-hoc study (protocol strict-nested-postmean-v2).

Every function here mirrors one in experiments/dpe_nested_corrected/code (corrected_nested.py,
nested_selection.py) or src/methods/dpe_independent / dpe_head_reweighting, with the four
Waterbirds groups and two classes replaced by the groups and classes found in the data.
For Waterbirds inputs the arithmetic is unchanged; tests/test_equivalence.py checks this
bit for bit against the original code.
"""
