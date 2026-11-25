# PCM conductance drift replication

This package reproduces the numerical results of the revised manuscript. It uses the version 1 Figure 2D source workbooks from Cai et al. and the Digits dataset packaged with scikit-learn. The script verifies the exact source archive by SHA-256. [Cai et al. data](https://zenodo.org/records/19162997) · [Digits data loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

## Run in PyCharm or a terminal

Open this folder as a PyCharm project and select a Python 3.12 environment with the pinned dependencies. Run `pcm_drift_accuracy_experiment.py` with a local copy of `Data.zip` and a new output directory. From a terminal in this folder, the equivalent commands are

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python pcm_drift_accuracy_experiment.py --data /path/to/Data.zip --out repeat
python plot_results.py repeat
```

Omitting `--data` downloads the versioned archive from Zenodo over HTTPS. If the Python interpreter cannot validate the HTTPS connection, download `Data.zip` from the record in your browser and use the local argument. The original raw archive is available from Zenodo. The supplied outputs were generated on macOS arm64 with Python 3.12.6. Numerical libraries or hardware can affect retraining across environments.

## Output files

| Path | Contents |
| --- | --- |
| `run/device_summary.csv` | Workbook paths, state assignments, counts, means, ratio percentiles and fraction of ratios above one |
| `run/trial_scores.csv` | Floating-point, quantized and state-mean rows for every fitted model, plus one empirical row per trial and positive λ value |
| `run/seed_summary.csv` | Accuracy, mapping and endpoint losses, device-assignment SD and changed predictions per fitted model |
| `run/legacy_mlp_trial_scores.csv` | Replay of the draft and original PyCharm MLP RNG streams at training seed 42 |
| `run/split.npz` | Outer training and test row indices into scikit-learn Digits |
| `run/mlp_*.npz`, `run/cnn_*.npz` | Trained weight and bias tensors, stored in layer order |
| `run/training.csv` | Iterations or epochs per fit and CNN checkpoint epoch |
| `run/manifest.json` | Data and code hashes, package versions, training seeds and interpolation grid |
| `run/figures/` | PNG and editable vector PDF exports of the manuscript figures |

There are 20 fitted models and 6,060 rows in the main score file. The quantized baseline is deterministic per fit and stored once. Each of 30 simulated device assignments is fixed across positive λ values within a fit and tested on all 450 images. Values above zero denote progression along a log-conductance interpolation. Only λ = 1 uses the complete empirical endpoint ratios. The parameter has no time calibration.

The sample SD in `endpoint_trial_sd_pp` describes simulated assignments within a fit. The manuscript's across-fit SD describes variation over the ten training seeds after averaging over assignments. All models share the same outer test images. The CNN is a newly implemented model using the architecture described in the first draft.
