# PSC-scDRS notes (sample)

This is a small example corpus file. Replace it with your real project docs.

## scDRS overview
- scDRS is used to score disease relevance at the single-cell level using GWAS summary statistics.
- Typical inputs: GWAS summary statistics; single-cell expression matrix; gene set (optional).
- Typical outputs: per-cell scores, p-values, and cell-type level summary statistics.

## PSC context
Primary sclerosing cholangitis (PSC) is a chronic cholestatic liver disease.

## Practical notes
- Keep thresholds (e.g., Monte Carlo p-values) in configuration.
- Store provenance: which GWAS, which scRNA dataset, which parameters.
