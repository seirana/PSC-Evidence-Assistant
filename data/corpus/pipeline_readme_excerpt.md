# Pipeline excerpt (sample)

Steps:
1. Prepare GWAS summary statistics (harmonize alleles, liftover if needed).
2. Run scDRS on scRNA-seq data.
3. Summarize scores by cell type.

Common parameters:
- top_n_genes: number of genes used for gene set scoring.
- n_ctrl: number of control gene sets.
