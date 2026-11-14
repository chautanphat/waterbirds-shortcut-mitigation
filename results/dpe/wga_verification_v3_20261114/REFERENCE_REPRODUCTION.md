# Gate A independent reproduction

{
  "status": "FAIL",
  "reasons": [
    "Required original analysis/reproduction source unavailable: DPE_WGA_Deep_Analysis.md, DPE_WGA_Audit_Repro.zip; original script and original SHA manifest cannot be independently executed/compared."
  ],
  "mean_delta_AA_pp": 0.44481512371420395,
  "mean_delta_WGA_pp": 1.7898824314003876,
  "flips": {
    "flips": 48,
    "corrected": 32,
    "broken": 16
  },
  "source_comparison_completed": false
}

[
  {
    "seed": 0,
    "uniform_AA": 94.74562135112595,
    "candidate_AA": 95.07923269391159,
    "uniform_WGA": 91.72932330827066,
    "candidate_WGA": 93.98496240601504,
    "max_discrepancy_pp": 3.759398495617461e-05
  },
  {
    "seed": 1,
    "uniform_AA": 94.82902418682235,
    "candidate_AA": 95.24603836530441,
    "uniform_WGA": 92.27467811158799,
    "candidate_WGA": 93.13304721030042,
    "max_discrepancy_pp": 4.7210300422761975e-05
  },
  {
    "seed": 2,
    "uniform_AA": 95.16263552960801,
    "candidate_AA": 95.7464553794829,
    "uniform_WGA": 92.4812030075188,
    "candidate_WGA": 94.73684210526315,
    "max_discrepancy_pp": 4.4620517101634505e-05
  }
]

Legacy metrics: [
  {
    "seed": 0,
    "uniform": {
      "accuracy": 0.9541666666666667,
      "balanced_accuracy": 0.9503077388759964,
      "nll": 0.6370971357630488,
      "brier": 0.4440523378130737,
      "ece": 0.42478044256567954,
      "per_class": {
        "0": 0.9572192513368984,
        "1": 0.9433962264150944
      },
      "count": 240,
      "per_group": {
        "0": {
          "accuracy": 0.9693877551020408,
          "count": 98
        },
        "1": {
          "accuracy": 0.9438202247191011,
          "count": 89
        },
        "2": {
          "accuracy": 0.9166666666666666,
          "count": 24
        },
        "3": {
          "accuracy": 0.9655172413793104,
          "count": 29
        }
      },
      "wga": 0.9166666666666666,
      "worst_class_accuracy": 0.9433962264150944
    },
    "candidate": {
      "accuracy": 0.9625,
      "balanced_accuracy": 0.9556553324588841,
      "nll": 0.621495235326907,
      "brier": 0.4285717944705265,
      "ece": 0.4245402280685928,
      "per_class": {
        "0": 0.9679144385026738,
        "1": 0.9433962264150944
      },
      "count": 240,
      "per_group": {
        "0": {
          "accuracy": 0.9591836734693877,
          "count": 98
        },
        "1": {
          "accuracy": 0.9775280898876404,
          "count": 89
        },
        "2": {
          "accuracy": 0.9166666666666666,
          "count": 24
        },
        "3": {
          "accuracy": 0.9655172413793104,
          "count": 29
        }
      },
      "wga": 0.9166666666666666,
      "worst_class_accuracy": 0.9433962264150944
    },
    "solver": {
      "seed": 0,
      "solver_success": true,
      "direct_agreement": true,
      "objective": 0.6285327398429216,
      "weight_min": 0.0,
      "weight_max": 0.25,
      "effective_members": 4.58560492016133,
      "top_members": [
        0,
        2,
        1,
        3,
        11
      ]
    }
  },
  {
    "seed": 1,
    "uniform": {
      "accuracy": 0.9375,
      "balanced_accuracy": 0.9328523862375139,
      "nll": 0.6419395589433258,
      "brier": 0.4488781342987261,
      "ece": 0.41002670526504514,
      "per_class": {
        "0": 0.9411764705882353,
        "1": 0.9245283018867925
      },
      "count": 240,
      "per_group": {
        "0": {
          "accuracy": 0.9787234042553191,
          "count": 94
        },
        "1": {
          "accuracy": 0.9032258064516129,
          "count": 93
        },
        "2": {
          "accuracy": 0.8846153846153846,
          "count": 26
        },
        "3": {
          "accuracy": 0.9629629629629629,
          "count": 27
        }
      },
      "wga": 0.8846153846153846,
      "worst_class_accuracy": 0.9245283018867925
    },
    "candidate": {
      "accuracy": 0.9541666666666667,
      "balanced_accuracy": 0.9503077388759964,
      "nll": 0.6255001568263824,
      "brier": 0.43254784765000776,
      "ece": 0.4178141643012021,
      "per_class": {
        "0": 0.9572192513368984,
        "1": 0.9433962264150944
      },
      "count": 240,
      "per_group": {
        "0": {
          "accuracy": 0.9787234042553191,
          "count": 94
        },
        "1": {
          "accuracy": 0.9354838709677419,
          "count": 93
        },
        "2": {
          "accuracy": 0.9230769230769231,
          "count": 26
        },
        "3": {
          "accuracy": 0.9629629629629629,
          "count": 27
        }
      },
      "wga": 0.9230769230769231,
      "worst_class_accuracy": 0.9433962264150944
    },
    "solver": {
      "seed": 1,
      "solver_success": true,
      "direct_agreement": true,
      "objective": 0.6264880191080039,
      "weight_min": 0.0,
      "weight_max": 0.25,
      "effective_members": 4.609643735642836,
      "top_members": [
        0,
        1,
        2,
        4,
        3
      ]
    }
  },
  {
    "seed": 2,
    "uniform": {
      "accuracy": 0.9583333333333334,
      "balanced_accuracy": 0.9665018666128544,
      "nll": 0.6382333440391551,
      "brier": 0.44518299746830464,
      "ece": 0.4292484750350317,
      "per_class": {
        "0": 0.9518716577540107,
        "1": 0.9811320754716981
      },
      "count": 240,
      "per_group": {
        "0": {
          "accuracy": 0.9797979797979798,
          "count": 99
        },
        "1": {
          "accuracy": 0.9204545454545454,
          "count": 88
        },
        "2": {
          "accuracy": 0.9565217391304348,
          "count": 23
        },
        "3": {
          "accuracy": 1.0,
          "count": 30
        }
      },
      "wga": 0.9204545454545454,
      "worst_class_accuracy": 0.9518716577540107
    },
    "candidate": {
      "accuracy": 0.9625,
      "balanced_accuracy": 0.9691756634042983,
      "nll": 0.6196752336512141,
      "brier": 0.42676001375101186,
      "ece": 0.4233028640735447,
      "per_class": {
        "0": 0.9572192513368984,
        "1": 0.9811320754716981
      },
      "count": 240,
      "per_group": {
        "0": {
          "accuracy": 0.9696969696969697,
          "count": 99
        },
        "1": {
          "accuracy": 0.9431818181818182,
          "count": 88
        },
        "2": {
          "accuracy": 0.9565217391304348,
          "count": 23
        },
        "3": {
          "accuracy": 1.0,
          "count": 30
        }
      },
      "wga": 0.9431818181818182,
      "worst_class_accuracy": 0.9572192513368984
    },
    "solver": {
      "seed": 2,
      "solver_success": true,
      "direct_agreement": true,
      "objective": 0.6270914387254819,
      "weight_min": 0.0,
      "weight_max": 0.25,
      "effective_members": 4.236610311722964,
      "top_members": [
        1,
        2,
        0,
        3,
        5
      ]
    }
  }
]

This reproduces a non-nested sensitivity analysis of existing 3-fold OOF predictions. Head slots differ across original folds; static slot weights are not assumed compatible with final refits. Prior method nomination used development group metrics; no untouched confirmation is claimed. No historical test arrays were opened. Original source scripts were not available, and no replacement script is represented as the original.
