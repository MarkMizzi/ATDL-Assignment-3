
|   | Train Its.    | Dataset | Model   | From Scratch/Pre-trained     | REPA? | REPA Target Repr. | Depth | By who? | FID                | sFID               | Inception Score    | Precision | Recall              |
|---|---------------|---------|---------|------------------------------|-------|-------------------|-------|---------|--------------------|--------------------|--------------------|-----------|---------------------|
|   | **Baselines** |         |         |                              |       |                   |       |         |                    |                    |                    |           |                     |
| 1 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | No    | N/A               | N/A   | Mark    | 80.27303933042617  | 62.258300652943376 | 2.617237091064453  | 0.60225   | 0.15037037037037038 |
| 1 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | No    | N/A               | N/A   | Mark    | 51.64963551101266  | 55.72212571673333  | 2.863781690597534  | 0.655     | 0.22864197530864197 |
| 2 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | No    | N/A               | N/A   | Mark    | 3.569061517715454  | 19.27869544933168  | 49.33616251007152  | 0.7395    | 0.5022222222222222  |
| 3 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-b          | 8     | Mark    | 79.65747509200366  | 60.89592521520444  | 2.5499279499053955 | 0.62225   | 0.15777777777777777 |
| 3 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-b          | 8     | Mark    | 54.04050976074956  | 55.82398739859457  | 2.8306543827056885 | 0.67075   | 0.21777777777777776 |
| 4 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-L          | 8     | Mark    | 80.39328605923117  | 61.7360577277899   | 2.5780441761016846 | 0.606     | 0.17308641975308642 |
| 4 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-L          | 8     | Mark    | 53.77622273080169  | 56.53031059528735  | 2.7971341609954834 | 0.6725    | 0.21160493827160493 |
| 5 | 60590         | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-g          | 8     | Mark    | 82.43904877538353  | 62.1949197495785   | 2.527996778488159  | 0.62175   | 0.17185185185185184 |
| 5 | 100010        | EuroSAT | SiT-L/2 | From Scratch                 | Yes   | DINOv2-g          | 8     | Mark    | 52.54135039122525  | 56.252716062052286 | 2.915581703186035  | 0.66825   | 0.23308641975308642 |
| 6 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-b          | 8     | Mark    | 18.728884935492715 | 49.26786589754215  | 3.677661895751953  | 0.73325   | 0.5269135802469136  |
| 7 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-L          | 8     | Mark    | 18.808025739850507 | 49.341572885164396 | 3.6531643867492676 | 0.72975   | 0.5217283950617284  |
| 8 | 60590         | EuroSAT | SiT-L/2 | Pre-trained w/ REPA DINOv2-L | Yes   | DINOv2-g          | 8     | Mark    | 18.884579815931033 | 49.36871052880076  | 3.7132115364074707 | 0.7225    | 0.5214814814814814  |

## Notes

- NN distance is calculated based on per-pixel MSE metric and relative to training dataset, tests whether model is memorizing
- Models trained from scratch are not trained for long enough, evaluate at later checkpoint
- Best performing by far is model pre-trained with REPA and then fine-tuned without REPA, surprising. Maybe explained by fact that DINOv2 has poor semantic representation of domain specific images.
