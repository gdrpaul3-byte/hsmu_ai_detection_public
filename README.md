# Real vs. AI-Generated Face Portrait Detection

Public stimuli, metadata, and de-identified behavioral data for a study on human ability to discriminate real from AI-generated face portraits. This repository is associated with research conducted at the Department of Bio-Healthcare, Hwasung Medi-Science University (HSMU), Hwaseong, Korea.

**Web app:** https://hsmu-real-vs-ai-test.web.app/

## Repository structure

```
├── stimuli/
│   ├── real/                 # FFHQ-derived real portrait stimuli (per-image licenses apply)
│   └── ai_generated/        # AI-generated portrait stimuli (ChatGPT, Gemini)
├── data/
│   ├── exp_data/
│   │   ├── responses_export_deidentified.csv   # Trial-level behavioral data
│   │   ├── surveys_export_deidentified.csv     # Post-task survey data
│   │   └── data_dictionary.md                  # Column descriptions
│   └── stimuli_metadata/
│       ├── Supplementary_Table_S1_FFHQ_credits.csv  # Per-image attribution and license info
│       └── Supplementary_Table_S1_FFHQ_credits.md   # Formatted version of the above
```

## Real portraits and licensing

Real portrait stimuli in `stimuli/real/` were sampled from the Flickr-Faces-HQ (FFHQ) dataset (Karras et al., 2019). Each image retains its original per-image license. Some images are licensed under CC BY-NC 2.0 and may not be used for commercial purposes.

Per-image attribution, license type, source URL, creator, and modification details are provided in:

> `data/stimuli_metadata/Supplementary_Table_S1_FFHQ_credits.csv`

Users must comply with the individual license terms listed in that file.

## De-identified data

Trial-level responses and post-task survey responses are provided in `data/exp_data/`. All data are de-identified. See `data/exp_data/data_dictionary.md` for column descriptions.

Additional raw logs are not included in this public repository but may be available from the corresponding author upon reasonable request.

## Contact

Department of Bio-Healthcare, Hwasung Medi-Science University (HSMU), Hwaseong, Korea
Sunwhi Kim — gdrpaul3@gmail.com
