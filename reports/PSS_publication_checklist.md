# Planetary and Space Science Submission Prep

Local files used:

- `pubs/els-cas-templates.zip`
- `pubs/Guide for authors - Planetary and Space Science - ISSN 0032-0633 _ ScienceDirect.com by Elsevier.pdf`
- `reports/processing_report.tex`

Current preparation status:

- Uses the Elsevier CAS single-column class `cas-sc` copied from the provided template archive.
- Uses `natbib`/BibTeX with the CAS `cas-model2-names` bibliography style.
- Builds successfully from `reports` with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
- Includes abstract, highlights, keywords, author block, affiliations, main article body, declarations, data availability, acknowledgements, and bibliography.
- Main text is extracted from the processing report through `Future Work`; diagnostic appendices remain in the report and can be converted to supplementary material later.

Open items before submission:

- Verify author order, affiliations, email address, and CRediT roles.
- Replace NASA social-media secondary-source wording with a primary technical source, or keep it clearly caveated.
- Decide which diagnostic appendices should become supplementary material.
- Add funding acknowledgements and facility/data-use statements.
- Complete the Elsevier generative-AI declaration according to the authors' intended disclosure.
- Archive the analysis code and processed products, then update the data availability statement.
- Consider reducing the number of figures/tables for the main article if PSS page flow becomes too long.
