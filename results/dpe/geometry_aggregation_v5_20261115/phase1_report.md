# Phase1 — frozen exploratory screening

25 total configs including uniform and diagnostic Top1; 75 candidate/seed evaluations. All losing results retained. G2 scalers fit actual fold-specific meta_train heads only.

Locked before subgroup evaluation: [{"id": "G1_g50", "method": "G1", "gamma": 50.0, "alpha": 0.25}, {"id": "S2_t0.025", "method": "S2", "tau": 0.025, "T_class": 1.0, "alpha": 1.0}]

The proxy is mean worst-class NLL under strict per-seed AA floor; it is not WGA. Class-wise S methods use separate class-specific prototype contributions, not a common expert mixture. No extra class-temperature tuning, neural model or historical test rows are used. The two-family shortlist rule was preregistered. OOF slot transfer is not used in nesting.
