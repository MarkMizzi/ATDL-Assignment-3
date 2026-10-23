- 22/10 Mark. Downloaded the EuroSAT dataset with:
  ```python
  import kagglehub 
  kagglehub.login()
  path = kagglehub.dataset_download("apollo2506/eurosat-dataset", output_dir="data")
  ```

- 23/10 Mark. Converted EuroSAT dataset to format used by REPA scripts using
  ```python
  python REPA/preprocessing/dataset_tools.py convert --source EuroSAT-data/EuroSAT/ --dest images --resolution 256x256 --transform=center-crop-dhariwal
  ```

- 23/10 Mark. Generated EuroSAT latents on UCloud server using
  ```python
  python REPA/preprocessing/dataset_tools.py encode --source=images --dest=vae-sd
  ```
  Version of REPA code used is latest of personal fork.
