# Preweek Standard：GPT-2のtemperature比較

## 実行方法

リポジトリのルートで実行する。先に下の「実行前の予想」を記入する。

```bash
uv run python work/preweek/standard_temperature.py
```

同じpromptでtemperatureを0.5、1.0、1.5に変え、それぞれseed 42、43、44で生成する。
最初の次tokenについて、生のlogitsとtemperature適用後の確率も表示する。
表示する上位5候補の確率は、語彙全体で計算しているため、この5個だけでは合計1にならない。

設定を変える例：

```bash
uv run python work/preweek/standard_temperature.py \
  --prompt "The future of artificial intelligence is" \
  --temperatures 0.3 0.7 1.0 1.5 --seeds 42 43 44 --max-new-tokens 60
```

初回はHugging Faceからモデルとtokenizerをダウンロードする。
キャッシュ済みなら`--local-files-only`を付けるとオフラインで実行できる。
デバイスは利用可能なMPS、CUDA、CPUの順で自動選択する。
CPUで実行したい場合は`--device cpu`を付ける。
GPT-2は英語モデルなので、最初は英語の文章の書き出しをpromptに使う。

## 実行前の予想

- temperatureを低くすると、最有力候補の確率はどうなると思うか：単語の確率の差が小さくなる
- 生成される文章はどう変わると思うか：temperatureを大きくするほどおかしな文章が増えそう
- temperatureを変えても変わらないものは何だと思うか：logits

## 実行条件

- 実行日時：9月6日(日)
- prompt：'Once upon a time, in a small village,'
- temperature：0.5,1.0,1.5
- seeds：42,43,44
- max-new-tokens：50
- device：mps

## 観察記録

必要に応じて行を追加し、出力から短く抜粋する。

| temperature | seed | 最有力の次tokenと確率 | 生成された続きの抜粋 | 気づいたこと |
| --- | --- | --- | --- | --- |
| 0.5 | 42 | a,0.51 | | 全体的にすじの通った分ができている |
| 1.0 | 42 | a,0.14 | | 小さな村という設定が消えているが文章はまだすじが通っている |
| 1.5 | 42 | a,0.02 | | 文章につながりがなくなり何を言っているのかよくわからない |

- ほかのseedでも見られた傾向／見られなかった傾向：temperatureを増やすと文章のつながりが消えていく
- 予想と一致した点・違った点：temperatureを増やすとおかしな文章になっていく。
- 変化したもの：文章、確率、
- 変わらなかったもの：単語の確率のランキング
- この観察だけでは判断できないこと：

## 解釈するときの注意

- temperatureは`softmax(logits / temperature)`に作用する。モデルのweightsを更新する処理ではない。
- 最初の次tokenは同じpromptから得たlogitsを比較する。生成の途中で選ばれるtokenが異なると、その後は入力履歴が異なるため、生のlogitsも異なり得る。
- `do_sample=True`で確率に従って選ぶ。候補を制限するtop-k/top-pは無効にしてある。
- 同じseedでもtemperatureを変えると文章は変わり得る。異なるデバイスやライブラリのバージョンで同じ出力になる保証もない。
- 数例の文章だけで一般的な品質やモデルの能力を断定しない。EOS（終了token）が出た場合、指定した最大token数より早く終了する。

参考：[GPT-2モデルカード](https://huggingface.co/openai-community/gpt2)、[Hugging Faceの生成設定](https://huggingface.co/docs/transformers/main_classes/text_generation)。
