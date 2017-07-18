# FineGrainedArabicPOSTagger
An implementation of fine-grained Arabic POS tagger proposed in the following paper:
- 「アラビア語の高粒度な品詞タグ付けのための辞書情報を活用した形態統語的カテゴリの同時予測」

## Requirement
- Python 3
- DyNet

## Data Format
We assume CoNLL data format as follows:
    TOKEN_NUMBER\tTOKEN\tCATEGORY1\tCATEGORY2 ... \tCATEGORY14
