# Week 1 Notes

## 今週理解したこと

## 重要な図・式・コード（1つ）

## 実験

### Prediction
causal-maskは未来の情報を与えないようにweightsを0にするものだからno-causal-maskを行うことによってweightsが0になるところがなくなる。
### Changed variable
### Fixed conditions
### What is not controlled
### Result / Observation
```
input:            (1, 4, 8) = (B, T, C)
Q/K/V projected:  (1, 4, 8) = (B, T, C)
Q/K/V per head:   (1, 2, 4, 4) = (B, H, T, D)
scores Q @ K^T:   (1, 2, 4, 4) = (B, H, T, T)
weights:          (1, 2, 4, 4) = (B, H, T, T)
weights @ V:      (1, 2, 4, 4) = (B, H, T, D)
merged heads:     (1, 4, 8) = (B, T, C)
output projection: (1, 4, 8) = (B, T, C)
causal mask: on
head 0 attention weights:
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.3445, 0.6555, 0.0000, 0.0000],
        [0.3284, 0.3047, 0.3670, 0.0000],
        [0.2522, 0.2522, 0.2355, 0.2601]])

input:            (1, 4, 8) = (B, T, C)
Q/K/V projected:  (1, 4, 8) = (B, T, C)
Q/K/V per head:   (1, 2, 4, 4) = (B, H, T, D)
scores Q @ K^T:   (1, 2, 4, 4) = (B, H, T, T)
weights:          (1, 2, 4, 4) = (B, H, T, T)
weights @ V:      (1, 2, 4, 4) = (B, H, T, D)
merged heads:     (1, 4, 8) = (B, T, C)
output projection: (1, 4, 8) = (B, T, C)
causal mask: off
head 0 attention weights:
tensor([[0.2201, 0.2896, 0.2586, 0.2317],
        [0.1982, 0.3771, 0.2111, 0.2136],
        [0.2548, 0.2364, 0.2848, 0.2239],
        [0.2522, 0.2522, 0.2355, 0.2601]])
```
### Interpretation（言えること / 言えないこと）
causal-maskが未来の情報を与えないようにweightsを0にしている部分がある。
maskの有無によってshapeは変わらない

## Standardの結果
```
input:            (2, 4, 8) = (B, T, C)
Q/K/V projected:  (2, 4, 8) = (B, T, C)
Q/K/V per head:   (2, 2, 4, 4) = (B, H, T, D)
scores Q @ K^T:   (2, 2, 4, 4) = (B, H, T, T)
weights:          (2, 2, 4, 4) = (B, H, T, T)
weights @ V:      (2, 2, 4, 4) = (B, H, T, D)
merged heads:     (2, 4, 8) = (B, T, C)
output projection: (2, 4, 8) = (B, T, C)
causal mask: on
head 0 attention weights:
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.5769, 0.4231, 0.0000, 0.0000],
        [0.2135, 0.6488, 0.1377, 0.0000],
        [0.2441, 0.1603, 0.3358, 0.2598]])
input:            (2, 4, 8) = (B, T, C)
Q/K/V projected:  (2, 4, 8) = (B, T, C)
Q/K/V per head:   (2, 2, 4, 4) = (B, H, T, D)
scores Q @ K^T:   (2, 2, 4, 4) = (B, H, T, T)
weights:          (2, 2, 4, 4) = (B, H, T, T)
weights @ V:      (2, 2, 4, 4) = (B, H, T, D)
merged heads:     (2, 4, 8) = (B, T, C)
output projection: (2, 4, 8) = (B, T, C)
causal mask: off
head 0 attention weights:
tensor([[0.1924, 0.3058, 0.1896, 0.3122],
        [0.2642, 0.1938, 0.3029, 0.2392],
        [0.1568, 0.4765, 0.1011, 0.2656],
        [0.2441, 0.1603, 0.3358, 0.2598]])
```
d_kは4
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

bigramLMでは一つの文字しか見れなくて良い予測ができなかった。なので全体の情報を取る必要がある。
そのためには全体の平均を取る方法がある。未来の単語の情報をとらないように直接平均を求める方法があるが、情報ごとに取り出す割合を求めて情報にかけることで得る方法もある。

アテンションにはまずquery,key,valueがある。queryは今見ているトークンが欲しい情報、keyはトークンたいが持っている情報、valueはその情報自体の値である。アテンションで行っていることはqueryとkeyで内積を求める。その後未来の情報は与えないようにし、与える情報の割合を求める。その後valueをかける。

マルチヘッドアテンションはアテンションでは一つの概念にしか注目できないが実際の文章では色々考慮するところがある。そこを全て考慮するためにマルチヘッドアテンションにしている。一つ一つのアテンションからでた結果はまずcatでとりあえず繋げたあとprojで線形変換を行いただ繋げただけのものから新しい表現として使うことができる。

transformerでは正規化を行った後にマルチヘッドアテンションに値を入れている。これは入力を勾配爆発などをさけるため。その後feedforwardを入れている。

最終的にgptでの学習はembeding tableとposition tableからlogitsを取り出すその後円ヘッドアテンションに値を入れる。その後logitsとtargetから損失を求め逆伝播を行う。

encoderに翻訳前の文章を入れてdecoderには正解の文章、もしくは生成した文章を入れる。encoderではマルチヘッドアテンションを入れて、add&norm、feed forwardにいれてdecoderに合流する。decoderはまずmaskマルチヘッドアテンションに入れて正解の文章を入れていた場合未来の情報をいれないようにする。その後Qはdecoder,KとVはencoderから持ってきてdecoderにとって必要な情報をencoderから取ってくるような形にする。