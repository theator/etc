# ETC: Estimated time to surgical procedure completion

Official code for models and loss functions described in the paper:

Estimated time to surgical procedure completion: An exploration of video analysis methods. MICCAI 2023.\
Barak Ariel, Yariv Colbeci, Judith Rapoport Ferman, Dotan Asselmann, Omri Bar.

---
## Requirements
- Python >= 3.8 (development was done using 3.8.12)
- PyTorch >= 1.12.1 (development was done using 1.12.1+cu113)

## Models:
See [`model.py`](./model.py) for code implementation of: 
- ETC-LSTM
- ETCouple
- ETCFormer

## Loss functions:
See [`losses.py`](./losses.py) for code implementation of:
- MAE loss
- Smooth L1 loss
- SMAPE loss
- Corridor loss
- Interval L1 los
- Total variation denoising loss 

---

## License
ETC code is released under the [Apache 2.0 license](LICENSE).

