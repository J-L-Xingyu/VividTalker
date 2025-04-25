import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Initialize models globally to avoid redundant loading
def initialize_models(language='English'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_base = f'/share/home/liuxingy/real3dportrait/OpenVoice/checkpoints/base_speakers/EN'
    ckpt_converter = '/share/home/liuxingy/real3dportrait/OpenVoice/checkpoints/converter'

    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

    return base_speaker_tts, tone_color_converter, source_se, device


# Load models once
global_base_speaker_tts, global_tone_color_converter, global_source_se, global_device = initialize_models()


def clone_voice(reference_audio_path: str, text: str, output_path: str, language='English', speed=1.0):
    """
    Generates a new speech audio with the cloned voice from a reference audio,
    keeping the original voice identity and style while changing only the text.

    :param reference_audio_path: Path to the reference audio file (original speaker audio)
    :param text: The text to be synthesized
    :param output_path: Path to save the generated audio
    :param language: Language of the synthesized text (default: 'English')
    :param speed: Speech speed (default: 1.0)
    """
    base_speaker_tts = global_base_speaker_tts
    tone_color_converter = global_tone_color_converter
    source_se = global_source_se
    device = global_device

    # Extract target speaker embeddings (keeping voice identity and style)
    target_se, _ = se_extractor.get_se(reference_audio_path, tone_color_converter, vad=True)

    # Generate base TTS audio with the new text
    tmp_tts_path = output_path.replace('.wav', '_tmp.wav')
    base_speaker_tts.tts(text, tmp_tts_path, speaker='default', language=language, speed=speed)

    # Convert generated speech to match reference speaker’s voice identity
    tone_color_converter.convert(
        audio_src_path=tmp_tts_path,
        src_se=source_se,  # Default speaker embedding
        tgt_se=target_se,  # Target speaker embedding (cloned voice)
        output_path=output_path,
        message="@OpenVoiceAI"  # Optional watermark
    )

    # Remove temporary TTS file
    os.remove(tmp_tts_path)

    print(f"✅ Voice cloned successfully! New speech saved at: {output_path}")
