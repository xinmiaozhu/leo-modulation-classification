# STARNet reproduction provenance

The implementation registered as `paper_starnet` is a PyTorch port of the
authors' public `starnet.py` for Zhang et al., *IEEE Transactions on Wireless
Communications*, vol. 23, no. 10, 2024, DOI 10.1109/TWC.2024.3400754.

Official source: <https://github.com/wangzishuodylan/STARNet-Automatic-Modulation-Classiffcation>

Preserved elements are the native 128-point amplitude/phase input, 10% temporal
masking, attentive lightweight convolution branch, two 32-unit GRUs, shared
64-dimensional representation, 128-by-2 reconstruction head, and the official
0.6 reconstruction / 0.4 classification loss weighting. Necessary study
adaptations are: 11 to 7 output classes; compensated I/Q supplied by the common
receiver front end; the unmasked version of each training window used as the
reconstruction target; and mean-logit aggregation over native-length windows.
The original repository does not publish pretrained weights for this dataset.
