# NF-Causal Decision Workbench V3.1.1 - Windows testing build

This prerelease combines the accepted V3.1 computational core with GUI revision V3.1.1. Scientific formulas, algorithms, parameters, factor plans, and metrics are unchanged.

## Assets

- `NF_Causal_Workbench_V3_1_1_Windows_Tester.zip` - portable Windows 10/11 x64 application; Python is not required;
- `NF_Causal_Workbench_V3_1_1_Test_Data.zip` - accepted UCI source pool, reference configuration, and a deterministic sample import file;
- matching `_SHA256.txt` files - SHA-256 checksums for both release archives.

## Verification completed

- the accepted UCI source file matches SHA-256 `afbfbed015d20f8421c32c62db37367c018eb6e92b00ea62a23354af8f84c44e`;
- the deterministic 1500-row release sample matches SHA-256 `0d0a2cef3999beeb2cd8ec717124084b15c917c2697f99a12c72c04a7f458f59`;
- the Windows x64 application was built locally with Python 3.12 and Nuitka 4.1.1;
- autonomous package smoke-check passed, including main-window initialization and all 5 workspaces;
- distribution verification passed: AMD64 PE, 10,559 files, and no `.py`, `.pyc`, `.pyo`, `.ipynb`, source, test, or script directories in the portable package;
- the source regression suite passed: 82 tests passed, with one non-blocking third-party warning;
- the final ZIP was extracted to a clean directory and the packaged EXE launched successfully.

## Test scope and known limitations

- custom data import and manual evidence editing were not part of the final acceptance scope;
- G4 does not produce CATE profiles by design in the sequential g-formula path;
- the EXE is not digitally signed and may trigger Windows SmartScreen;
- the first launch may be slower while scientific libraries initialize;
- the full 5000/5000 experiment corpus is not included in this testing release.

