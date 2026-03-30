# Repository instructions

This repository is a public materials repository for an academic study on human discrimination of real versus AI-generated face portraits.

## Institutional context
This repository is associated with research conducted at the Department of Bio-Healthcare, Hwasung Medi-Science University (HSMU), Hwaseong, Korea.

For questions regarding this repository or the associated study, use:
- Sunwhi Kim
- gdrpaul3@gmail.com

## Purpose
- Share de-identified behavioral and survey data
- Share AI-generated portrait stimuli
- Share metadata and licensing information for FFHQ-derived real portraits
- Support manuscript Data availability and Supplementary Table S1 statements

## Repository constraints
- This is a public repository.
- Do not add raw logs, private/admin files, or unnecessary operational assets.
- Keep documentation concise, academic, and practical.
- Do not overstate legal conclusions.
- FFHQ-derived real portraits retain their original per-image licenses.
- Use `data/stimuli_metadata/Supplementary_Table_S1_FFHQ_credits.csv` as the source of attribution/license information.
- Some FFHQ-derived images may be under CC BY-NC 2.0, so documentation must clearly state that per-image licenses apply and that some files are non-commercial only.
- Real portraits in `stimuli/real/` are FFHQ-derived.
- AI-generated portraits are in `stimuli/ai_generated/`.
- Public de-identified data are in `data/exp_data/`.

## Documentation goals
Create or improve:
- `README.md`
- `data/exp_data/data_dictionary.md`
- `stimuli/real/README.md`

## README requirements
The README should:
- Explain the purpose of the repository
- Describe the folder structure
- Mention the public web app: https://hsmu-real-vs-ai-test.web.app/
- Explain that real portraits were sampled from FFHQ (Karras et al., 2019)
- Explain that per-image license and attribution information are provided in `data/stimuli_metadata/Supplementary_Table_S1_FFHQ_credits.csv`
- Explain that de-identified trial-level and survey data are included
- Include institutional affiliation and contact information