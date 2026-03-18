# Data Dictionary

Column descriptions for the de-identified data files in this directory.

## responses_export_deidentified.csv

Trial-level behavioral data. Each row is one trial for one participant.

| Column | Description |
|---|---|
| `imageType` | Stimulus category: `real`, `ai_chatgpt`, or `ai_gemini` |
| `appVersion` | Version of the web app used for the session |
| `response` | Participant's response: `REAL` or `AI` |
| `isCorrect` | Whether the response was correct (`True` / `False`) |
| `participantId` | De-identified participant identifier |
| `deviceType` | Device used: `mobile` or `web` |
| `trial` | Trial number within the session (1-indexed) |
| `rt` | Response time in milliseconds |
| `imageId` | Stimulus identifier (corresponds to filenames in `stimuli/`) |

## surveys_export_deidentified.csv

Post-task survey data. Each row is one participant session.

| Column | Description |
|---|---|
| `resultTitle_en` | Result title shown to participant (English) |
| `resultTitle_ko` | Result title shown to participant (Korean) |
| `appVersion` | Version of the web app used for the session |
| `participantId` | De-identified participant identifier |
| `timestamp` | Session timestamp (UTC) |
| `language` | Interface language selected by participant |
| `consent` | Whether the participant consented (`TRUE` / `FALSE`) |
| `gender` | Self-reported gender |
| `age` | Self-reported age |
| `nationality` | Self-reported nationality |
| `occupation` | Self-reported occupation |
| `mbti` | Self-reported MBTI type (optional) |
| `score` | Number of correct responses |
| `totalTrials` | Total number of trials in the session |
| `overallAccuracy` | Overall accuracy (percentage) |
| `realAccuracy` | Accuracy on real image trials (percentage) |
| `aiAccuracy` | Accuracy on AI-generated image trials (percentage) |
| `avgRT` | Mean response time in milliseconds |
| `repeatCount` | Number of times the participant has completed the task |
| `firstTime` | Whether this was the participant's first attempt (`yes` / `no`) |
| `deviceType` | Inferred from `participantId` suffix |
| `aiAttitude` | Self-reported attitude toward AI (e.g., `positive`, `neutral`, `negative`) |
| `aiConfidence` | Self-reported confidence in ability to detect AI images |
| `aiExposureFrequency` | How often the participant encounters AI-generated content |
| `aiExposureSources` | Comma-separated list of AI content exposure sources |
| `otherExposureSource` | Free-text field for unlisted exposure sources |
| `usedAiTools` | AI tools the participant has used (comma-separated) |
| `otherAiTool` | Free-text field for unlisted AI tools |
| `aiBenefits` | Perceived benefits of AI (comma-separated) |
| `otherBenefit` | Free-text field for unlisted benefits |
| `aiConcerns` | Perceived concerns about AI (comma-separated) |
| `otherConcern` | Free-text field for unlisted concerns |
| `strategy` | Self-reported strategies used during the task (comma-separated) |
| `otherStrategy` | Free-text field for unlisted strategies |
| `suggestions` | Open-ended participant suggestions |
| `responses` | JSON array of all trial-level responses for the session |

*Note:* Column descriptions were inferred from column names and observed values. Some survey items (e.g., `aiConfidence`, `aiExposureFrequency`) may use categorical scales whose exact wording is defined in the web app.
