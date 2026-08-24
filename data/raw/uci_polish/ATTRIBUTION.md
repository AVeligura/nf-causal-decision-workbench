# UCI Polish Companies Bankruptcy — attribution

Local source pool for the semi-synthetic generator.

- Dataset: *Polish Companies Bankruptcy*
- Creator: Sebastian Tomczak
- UCI identifier: 365
- DOI: https://doi.org/10.24432/C5F600
- UCI publication date: 10 April 2016
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Source URL: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
- Downloaded file: `data.csv`
- SHA-256: `afbfbed015d20f8421c32c62db37367c018eb6e92b00ea62a23354af8f84c44e`

The application uses A1, A2, A4, A5, A21, A27, A29 and A44 as an empirical
covariate pool. Missing values are median-imputed, values are winsorised at the
1st and 99th percentiles, and robust standardisation uses the median and IQR.
No UCI bankruptcy label is used as a causal outcome.

Reference paper: M. Zięba, S. Tomczak, J. M. Tomczak (2016), “Ensemble
boosted trees with synthetic features generation in application to bankruptcy
prediction”, *Expert Systems with Applications*, DOI 10.1016/j.eswa.2016.04.001.

