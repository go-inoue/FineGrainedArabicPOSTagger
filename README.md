# FineGrainedArabicPOSTagger
An implementation of fine-grained Arabic POS tagger proposed in the following paper:
- Joint Prediction of Morphosyntactic Categories for Fine-Grained Arabic Part-of-Speech Tagging Exploiting Tag Dictionary Information

## Important Notes
- We cannot release the dictionary extracted from SAMA3.1 database (LDC2010L01) due to the licensing issue, however, a sample dictionary is available at `data/ud_feat_dict.piclke`. `data/sample_dict.py` also includes a sample format of the dictionary.

## Requirement
- Python 3
- DyNet Ver.1.1
  - To install, please refer to [this section of DyNet 1.1 documentation](https://dynet.readthedocs.io/en/v1.1/python.html#manual-installation). If you are using OS X later than El Capitan (10.7), you might need to disable System Integrity Protection (SIP).

## How to train
- You can run `sh train.sh [joint|separate] [no_dict|dict_all|dict_one]` to train a model.
- e.g.) `sh train.sh joint no_dict` : This will train a joint model without dictionary information.

## How to test
- You can run `sh test.sh [joint|separate] [no_dict|dict_all|dict_one]` to test a model.
- e.g.) `sh test.sh joint no_dict` : This will test a joint model without dictionary information.
