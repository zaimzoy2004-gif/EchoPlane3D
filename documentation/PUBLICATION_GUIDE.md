# Publication guide

Checked on 2026-11-18 against the journals' own pages where they could be fetched. Rules change — re-check the
linked pages before submitting.

## 0. Before any submission

1. Push the repository to a public host (GitHub/GitLab) with the MIT licence.
2. Replace the placeholder author entries in `CITATION.cff`, `paper/paper.md` and
   `paper/softwarex_manuscript.md` with the real authors, affiliations and ORCIDs.
3. Re-run the evidence on the release commit, in this order:
   ```bash
   python scripts/run_paper_benchmarks.py      # ≈ 70 min on the reference laptop (CAMUS downloads included)
   python scripts/record_test_summary.py       # test counts quoted in the README
   python scripts/build_paper_numbers.py       # renders README.md, paper/paper.md, paper/softwarex_manuscript.md
   pytest                                      # includes tests/test_paper_sync.py
   ```
   Edit prose in `paper/templates/`, never the rendered files; numbers are `{{placeholders}}`.
4. Decide what to do with `CLAUDE.md` and `UPGRADE_SPEC.md` (the AI build specifications). Keeping them in
   the repository is the most transparent record of AI-assisted development.

## 1. JOSS (Journal of Open Source Software)

Source: <https://joss.readthedocs.io/en/latest/submitting.html> (fetched 2026-11-18).

* **Public development history:** the software needs *at least six months of public history* before
  submission, with development spread over that period (not concentrated in a few days or weeks), and evidence
  of research use (at minimum by the developers).
  **EchoPlane3D currently has no public history**, so it cannot be submitted to JOSS until about six months
  after the repository is made public and actively developed. This is the main blocker.
* **AI usage disclosure (mandatory):** state the tools and versions, where they were used (code, docs, paper),
  the nature and scope of assistance, and confirm that humans reviewed and validated the outputs and made the
  core design decisions. Incomplete or inaccurate disclosure can be treated as an ethical breach.
  AI must not be used in author–editor/reviewer conversations (except translation).
* **Paper:** short `paper.md` + `paper.bib` in the repository, with title, summary, authors, affiliations and
  key references. The page states no word limit; JOSS papers are conventionally short.
* **Licence:** OSI-approved (MIT is).
* **On acceptance:** tag a release, archive it (Zenodo/figshare), report the version and archive DOI.

Draft: `paper/paper.md`, `paper/paper.bib`.

## 2. SoftwareX (Elsevier) — Original Software Publication

The official Guide for Authors (<https://www.sciencedirect.com/journal/softwarex/publish/guide-for-authors>)
refused automated access; the points below come from search results quoting it and must be confirmed there.

* **Template is mandatory** (Word or LaTeX "Original software publication" template) — move
  `paper/softwarex_manuscript.md` into it.
* **Figures:** maximum six.
* **Word limit:** sources disagree (3,000 vs 4,000 words; abstract, text, captions and footnotes count;
  title, authors, references and metadata tables do not). The draft targets ≤ 3,000.
* **Code metadata table** (C1, C2, …): current version, permanent link to the code, licence, versioning system,
  languages/tools, dependencies, documentation link, support contact — copy the exact rows from the template.
* **Generative-AI declaration** (Elsevier policy,
  <https://www.elsevier.com/about/policies-and-standards/the-use-of-generative-ai-and-ai-assisted-technologies-in-writing-for-elsevier>):
  a section titled *Declaration of generative AI and AI-assisted technologies in the writing process*, placed
  at the end of the manuscript above the references; AI use in the research itself is described in the methods.
  AI tools cannot be authors.

Draft: `paper/softwarex_manuscript.md`.

## 3. arXiv / medRxiv preprint

* **arXiv:** suitable categories are `eess.IV` (image and video processing) with cross-lists such as `cs.CV`
  and `physics.med-ph`. First-time submitters may need an endorsement. Upload a PDF built from the SoftwareX
  or JOSS text with the figures in `results/figures/`.
* **medRxiv** screens for health-research manuscripts; a software paper without a clinical study may be
  declined there, so arXiv is the safer primary venue. Check medRxiv's scope page before submitting.
* Posting a preprint is compatible with both JOSS and SoftwareX as far as is known; confirm in each journal's
  preprint policy.

## 4. Zenodo DOI

1. Log in to Zenodo with the GitHub account, enable the repository under *GitHub* settings.
2. Create a GitHub release (e.g. `v1.1.0`); Zenodo archives it and mints a version DOI plus a concept DOI.
3. Put the DOI in `CITATION.cff`, `README.md` and the manuscripts' code metadata.
4. Archive **code and results only** — never CAMUS files (see `documentation/DATASETS.md`).

## 5. Wording rules for every venue

* Educational and research simulation prototype; **not** a medical device, not FDA/CE cleared, not for diagnosis.
* "Reference" means the CAMUS expert masks processed by our own Simpson code; it is not clinical ground truth.
* Report the PCA long-axis limitation, the schematic (non-patient) 3-D model, the tuned-on demo cohort, and
  that the Ollama path was tested only with mocked HTTP.
