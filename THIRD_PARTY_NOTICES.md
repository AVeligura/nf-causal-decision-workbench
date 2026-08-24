# Third-party notices

NF-Causal Decision Workbench is distributed under the MIT License. Its runtime
distribution includes third-party components governed by their own licenses.
The authoritative license texts are included with the installed Python wheels
and remain controlling.

| Component | License family |
|---|---|
| PySide6 / Qt for Python | LGPL-3.0 / GPL-3.0 / commercial options |
| NumPy, pandas, SciPy, scikit-learn, statsmodels | BSD-style licenses |
| NetworkX, seaborn, psutil, joblib, PyTorch | BSD-style licenses |
| Matplotlib | Matplotlib and PSF-based licenses |
| PyArrow | Apache License 2.0 |
| openpyxl, pydantic, pydantic-settings, PyYAML | MIT License |
| ReportLab | BSD-style license |
| Nuitka, used only to produce the Windows build | Apache License 2.0 |

The Windows package is produced from the exact dependency set recorded in
`requirements-windows.lock`. Users who redistribute the binary should retain
the license and notice files supplied by the individual components.

## Dataset

The UCI Polish Companies Bankruptcy dataset is created by Sebastian Tomczak and
is distributed under the Creative Commons Attribution 4.0 International
license (CC BY 4.0). UCI identifier 365; DOI:
<https://doi.org/10.24432/C5F600>. Full attribution and the accepted file hash
are recorded in `data/raw/uci_polish/ATTRIBUTION.md`.

