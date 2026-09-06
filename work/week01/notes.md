# Week 1 Notes

## 今週理解したこと

## 重要な図・式・コード（1つ）

## 実験

### Prediction
### Changed variable
### Fixed conditions
### What is not controlled
### Result / Observation
### Interpretation（言えること / 言えないこと）

## まだ分からないこと
## 参照した資料

## outputメモ

bigramLMは前の文字だけ確認して次の文字を予測する。なので前の単語や文の流れは考慮しない。
embedding tableを作成してlogitsを取り出せるようにする。embedding tableは特徴量みたいなもので
特定の行を入力するとそれに合った列の値がすべて返ってきてそのまま渡す。この中身の値を変えることで
学習を行う。
logitsとtarget(正解)からcrossentropyを行う。logitsの正解の文字の値が大きく他の文字が小さいと良い。
学習の流れはembedding tableでlogitsを取得、その後crossentropyを使って損失の取得する。

generateの流れはembedding tableでlogitsを取得、最後の予測だけを知りたいので最後のlogitsだけ
取り出す。その後softmaxで確率にし、ある関数を使って確率からサンプリングを行う。そして最後に予測した
単語を入力に追加して再び入力を行う。

トレーニングプロセスはembedding tableでlogitsを取得、その後logitsと正解の損失をcrossentoropyで求める。その後optimizerの勾配をリセットし、逆伝播を行う。そして勾配を更新。これの繰り返し。