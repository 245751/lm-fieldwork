"""GPT-2で、同じpromptに対するtemperatureの影響を観察する。"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from lm_fieldwork.device import select_device


MODEL_NAME = "gpt2"  # Hugging FaceのGPT-2（openai-community/gpt2）


def positive_temperature(value: str) -> float:
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise argparse.ArgumentTypeError("temperatureは有限の正の値にしてください。")
    return temperature


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("1以上の整数にしてください。")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Once upon a time, in a small village,")
    parser.add_argument(
        "--temperatures", type=positive_temperature, nargs="+", default=[0.5, 1.0, 1.5]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-new-tokens", type=positive_int, default=50)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--local-files-only", action="store_true", help="ダウンロードせずキャッシュだけを使う"
    )
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("promptには空白以外の文字を含めてください。")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("seedは0以上、2**32 - 1以下にしてください。")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print("実行前の予想をstandard_observations.mdに記録してから比較してください。")
    print(f"model: {MODEL_NAME} / device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, local_files_only=args.local_files_only, use_safetensors=True
    ).to(device)
    model.eval()  # dropoutを無効にし、学習ではなく推論を行う
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("promptのtoken数 + max-new-tokensがモデルの文脈長を超えています。")

    print(f"prompt: {args.prompt!r}")
    print(f"prompt tokens: {prompt_length} / max new tokens: {args.max_new_tokens}")
    print(f"seeds: {args.seeds}")
    print("sampling: do_sample=True, top_k=0, top_p=1.0, repetition_penalty=1.0")

    # input_ids: [1, promptの長さ]
    # logits: [1, promptの長さ, 語彙数] → 最後の位置だけを取り出す → [語彙数]
    # 同じpromptの生のlogitsを一度だけ計算し、temperatureごとに再利用する。
    next_logits = model(**inputs).logits[0, -1].float().cpu()
    candidate_ids = next_logits.topk(5).indices.tolist()

    for temperature in args.temperatures:
        print(f"\n{'=' * 60}\ntemperature = {temperature}")
        probabilities = torch.softmax(next_logits / temperature, dim=-1)
        print("最初の次tokenの上位5候補（確率は語彙全体で正規化）")
        print("token ID | token | 生のlogit | 確率")
        for token_id in candidate_ids:
            token = tokenizer.decode([token_id])
            print(
                f"{token_id:>8} | {token!r} | {next_logits[token_id].item():.4f}"
                f" | {probabilities[token_id].item():.6f}"
            )

        for seed in args.seeds:
            # 同じseedを各temperatureで設定し直す。複数seedで偶然の影響も観察する。
            set_seed(seed)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,  # 上位k個で候補を切り捨てず、temperatureの効果を観察する
                top_p=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation_ids = output_ids[0, prompt_length:]
            continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            print(f"\nseed = {seed} / generated tokens = {len(continuation_ids)}")
            print(f"続き: {continuation!r}")

    print("\n観察後は、予想との違いと、この実験だけでは判断できないことを記録してください。")


if __name__ == "__main__":
    main()
