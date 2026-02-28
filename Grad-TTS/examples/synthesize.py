"""Basic Romanian TTS synthesis example.

Usage:
    # From HuggingFace Hub (once models are published):
    python examples/synthesize.py

    # From local checkpoints:
    python examples/synthesize.py \
        --model-path checkpts/grad-tts-ro.pt \
        --vocoder-path checkpts/hifigan.pt \
        --vocoder-config checkpts/hifigan-config.json

    # With custom text:
    python examples/synthesize.py --text "Buna ziua, cum te cheama?"

    # With a specific model variant from HuggingFace:
    python examples/synthesize.py --variant bas-950_100
"""

import argparse
import os

from gradtts_ro import GradTTSRomanian


SAMPLE_TEXTS = [
    "Bună ziua! Aceasta este o demonstrație de sinteză vocală în limba română.",
    "România este o țară frumoasă, cu munți înalți și văi adânci.",
    "Inteligența artificială transformă modul în care interacționăm cu tehnologia.",
]


def main():
    parser = argparse.ArgumentParser(description="Romanian Grad-TTS synthesis example")
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text to synthesize. If not provided, uses built-in sample texts.",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to local Grad-TTS checkpoint. If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--vocoder-path", type=str, default=None,
        help="Path to local HiFi-GAN vocoder checkpoint.",
    )
    parser.add_argument(
        "--vocoder-config", type=str, default=None,
        help="Path to local HiFi-GAN config JSON.",
    )
    parser.add_argument(
        "--config-path", type=str, default=None,
        help="Path to model config JSON. Uses defaults if not provided.",
    )
    parser.add_argument(
        "--variant", type=str, default="base",
        help="Model variant for HuggingFace download (default: base). "
             "Options: base, bas-10_100, bas-950_100, sgs-10_100, sgs-950_100",
    )
    parser.add_argument(
        "--output-dir", type=str, default="out",
        help="Directory to save generated WAV files (default: out)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50,
        help="Number of diffusion reverse steps (default: 50)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5,
        help="Sampling temperature (default: 1.5)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cpu', 'cuda', or auto-detect if not specified.",
    )
    args = parser.parse_args()

    # Load model
    if args.model_path:
        print(f"Loading model from local path: {args.model_path}")
        pipe = GradTTSRomanian.from_local(
            model_path=args.model_path,
            vocoder_path=args.vocoder_path,
            vocoder_config_path=args.vocoder_config,
            config_path=args.config_path,
            device=args.device,
        )
    else:
        print(f"Loading model from HuggingFace Hub (variant: {args.variant})...")
        pipe = GradTTSRomanian.from_pretrained(
            variant=args.variant,
            device=args.device,
        )

    print(f"Model loaded on device: {pipe.device}")

    # Prepare texts
    texts = [args.text] if args.text else SAMPLE_TEXTS

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Synthesize
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Synthesizing: {text[:60]}{'...' if len(text) > 60 else ''}")
        audio = pipe(
            text,
            n_timesteps=args.timesteps,
            temperature=args.temperature,
        )
        out_path = os.path.join(args.output_dir, f"sample_{i}.wav")
        pipe.save_wav(audio, out_path)
        print(f"  Saved: {out_path} ({len(audio) / pipe.sample_rate:.2f}s)")

    print(f"\nDone! Generated {len(texts)} sample(s) in '{args.output_dir}/'")


if __name__ == "__main__":
    main()
