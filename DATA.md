# Data and licensing

This code release does not include training, validation, held-out, or external-screening datasets.

The materials records used in the manuscript were derived from the Topological Materials Database and related topological-material resources. Those data are not redistributed here because they are governed by the terms of the original providers.

To reproduce the experiments:

1. Obtain the source materials records from the original Topological Materials Database / Topological Quantum Chemistry resources.
2. Check and follow the license and citation requirements of those resources.
3. Convert the records into the local JSON schema expected by the scripts, or adapt the scripts to your schema.
4. Create a fixed held-out split and a subtraining/validation split manifest.

Expected local files used by many scripts:

```text
label_noise_analysis/clean_training_data.json
label_noise_analysis/clean_test_data.json
label_noise_analysis/shared_split_manifest.json
```

The code assumes that each material record contains enough information to build composition, space-group, electron-count, orbital, bonding, element-category, topogivity, and structured-text features.
