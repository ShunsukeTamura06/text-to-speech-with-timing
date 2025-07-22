#!/usr/bin/env python3
"""
テキスト音声変換の使用例
"""

from text_to_speech_processor import TextToSpeechProcessor, quick_process, batch_process_texts

def main():
    """メインの使用例"""
    # GradioサーバーのURL（自分の環境に合わせて変更）
    gradio_url = "http://127.0.0.1:7860/"
    
    # 1. シンプルな使用例
    print("=== シンプルな使用例 ===")
    text = "こんにちは。(time:100)私は加藤です。(time:500)今日はITについて話します。"
    output_file = quick_process(text, "example_output.wav", gradio_url)
    print(f"出力ファイル: {output_file}")
    
    # 2. 詳細なパラメータ指定
    print("\n=== 詳細なパラメータ指定 ===")
    processor = TextToSpeechProcessor(gradio_url)
    
    result = processor.process_text_to_speech(
        text="はじめまして。(time:500)よろしくお願いします。(time:200)今後ともよろしくお願いいたします。",
        output_path="detailed_example.wav",
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        chunk_length=300
    )
    print(f"出力ファイル: {result}")
    
    # 3. バッチ処理の例
    print("\n=== バッチ処理の例 ===")
    texts = [
        "第一章。(time:1000)はじめに。(time:500)この章では基本概念を説明します。",
        "第二章。(time:1000)実装について。(time:500)具体的な実装方法を見ていきましょう。",
        "第三章。(time:1000)まとめ。(time:500)最後に今回の内容をまとめます。(time:1000)ありがとうございました。"
    ]
    
    results = batch_process_texts(
        texts,
        output_dir="batch_examples",
        gradio_url=gradio_url,
        temperature=0.8
    )
    
    print(f"バッチ処理結果: {len([r for r in results if r])}/{len(texts)} 成功")

if __name__ == "__main__":
    main()
