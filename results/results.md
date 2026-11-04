
|   | Train Its.    | Dataset | Model   | From Scratch/Pre-trained     | REPA? | REPA Target Repr. | Depth | By who? | FID                | sFID               | Inception Score    | Precision | Recall              |
|---|---------------|---------|---------|------------------------------|-------|-------------------|-------|---------|--------------------|--------------------|--------------------|-----------|---------------------|
|   | **Baselines** |         |         |                              |       |                   |       |         |                    |                    |                    |           |                     |
| 1 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | No    | N/A               | N/A   | Mark    | 80.27303933042617  | 62.258300652943376 | 2.617237091064453  | 0.60225   | 0.15037037037037038 |
| 1 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | No    | N/A               | N/A   | Mark    | 51.64963551101266  | 55.72212571673333  | 2.863781690597534  | 0.655     | 0.22864197530864197 |
| 2 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | No    | N/A               | N/A   | Mark    | 19.27869544933168  | 49.33616251007152  | 3.569061517715454  | 0.7395    | 0.5022222222222222  |
| 3 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-b          | 8     | Mark    | 79.65747509200366  | 60.89592521520444  | 2.5499279499053955 | 0.62225   | 0.15777777777777777 |
| 3 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-b          | 8     | Mark    | 54.04050976074956  | 55.82398739859457  | 2.8306543827056885 | 0.67075   | 0.21777777777777776 |
| 4 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-L          | 8     | Mark    | 80.39328605923117  | 61.7360577277899   | 2.5780441761016846 | 0.606     | 0.17308641975308642 |
| 4 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-L          | 8     | Mark    | 53.77622273080169  | 56.53031059528735  | 2.7971341609954834 | 0.6725    | 0.21160493827160493 |
| 5 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-g          | 8     | Mark    | 82.43904877538353  | 62.1949197495785   | 2.527996778488159  | 0.62175   | 0.17185185185185184 |
| 5 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-g          | 8     | Mark    | 52.54135039122525  | 56.252716062052286 | 2.915581703186035  | 0.66825   | 0.23308641975308642 |
| 6 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-b          | 8     | Mark    | 18.728884935492715 | 49.26786589754215  | 3.677661895751953  | 0.73325   | 0.5269135802469136  |
| 7 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-L          | 8     | Mark    | 18.808025739850507 | 49.341572885164396 | 3.6531643867492676 | 0.72975   | 0.5217283950617284  |
| 8 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-g          | 8     | Mark    | 18.884579815931033 | 49.36871052880076  | 3.7132115364074707 | 0.7225    | 0.5214814814814814  |
|   | **Base Models** |       |         |                              |       |                   |       |         |                    |                    |                    |           |                     |
| 9 | 60590        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | SatDINO-b         | 8     | Natasha | 18.832926920680677 | 49.333538254733526 | 3.618562936782837   | 0.725     | 0.5130864197530864  |
| 9 | 51830        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | SatDINO-b         | 8     | Natasha | 18.683377087011195 | 49.323988383189885 | 3.6550614833831787  | 0.708     | 0.5145679012345679  |
| 10 | 60590        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | SatDINO-s         | 8     | Natasha | 19.101463468296913 | 49.262847479803554 | 3.5430285930633545  | 0.716     | 0.5061728395061729  |
| 10 | 52560        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | SatDINO-s         | 8     | Natasha | 18.797160340873802 | 49.272870074258435 | 3.562255382537842   | 0.71225   | 0.5130864197530864  |
| 11 | 60590        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-L + SatDINO-b | 8  | Natasha | 18.824982514980547 | 49.25133713051969  | 3.6046314239501953  | 0.72      | 0.5214814814814814  |
| 11 | 50370        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-L + SatDINO-b | 8  | Natasha | 18.54439405343328  | 49.21815290751101  | 3.6572911739349365  | 0.705     | 0.5269135802469136  |
| 12 | 60590        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | SatDINO-b         | 8     | Natasha | 70.35061153170875  | 71.59246996203831  | 3.0716867446899414 | 0.528     | 0.1362962962962963  |
| 12 | 30660        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | SatDINO-b         | 8     | Natasha | 107.11349886252123 | 97.71336316800813  | 2.766411781311035  | 0.38725   | 0.03925925925925926 |
|   | **LoRa finetuning** |   |         |                              |       |                   |       |         |                    |                    |                    |           |                     |
| 13 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L (LoRA) | Yes   | SatDINO-b         | 8     | Natasha | 18.984920965749325 | 49.46311911436476  | 3.6067888736724854  | 0.7015    | 0.5306172839506172  |
| 13 | 70080         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L (LoRA) | Yes   | SatDINO-b         | 8     | Natasha | 19.2322864290162   | 49.529578426702756 | 3.575568675994873   | 0.71275   | 0.5160493827160494  |
| 14 | 60590        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L (LoRA) | Yes   | SatDINO-s         | 8     | Natasha | 18.966548800812802 | 49.4533929082188   | 3.6146390438079834  | 0.7055    | 0.5209876543209877  |
| 14 | 64240        | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L (LoRA) | Yes   | SatDINO-s         | 8     | Natasha | 19.091244683107504 | 49.494668038064106 | 3.616442918777466   | 0.702     | 0.5202469135802469  |

## Notes

- NN distance is calculated based on per-pixel MSE metric and relative to training dataset, tests whether model is memorizing
- Models trained from scratch are not trained for long enough, evaluate at later checkpoint
- Best performing by far is model pre-trained with REPA and then fine-tuned without REPA, surprising. Maybe explained by fact that DINOv2 has poor semantic representation of domain specific images.
