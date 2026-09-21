# Week 2 Notes

## Tokenizer comparison

## 実験

### Hypothesis / Prediction
### Changed variable
### Fixed conditions
### What is not controlled
### Metrics / Result
### Interpretation（言えること / 言えないこと）

## まだ分からないこと
## 参照した資料

## outputメモ
BPEでの学習方法はまず単語ごとに分けて前後のペアになった組の数をカウントする。そしてもっとも多かった
ペアを単語として保存して繰り返す。

tokenizerの処理の流れはまずnormalizationで大文字を小文字にしたりする。
その次にpre-tokenzationで空白ごとに分ける。
その次にmodelに入れてモデルごとの処理を行い、tokenを分ける。

