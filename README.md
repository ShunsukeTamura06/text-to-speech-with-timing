# Text-to-Speech with Timing

タイムタグ付きテキストを音声に変換し、時間指定の余白で結合するツール

## 概要

このツールは、特定のタイムタグが含まれたテキストを解析し、各セグメントを音声に変換して、指定された時間の余白を挟んで1つの音声ファイルに結合します。

## 機能

- タイムタグ `(time:ミリ秒)` でテキストを分割
- 各テキストセグメントを音声に変換
- 指定された時間の無音を挿入
- 全セグメントを順番通りに結合してWAVファイルに出力

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用例

```python
from text_to_speech_processor import quick_process

# タイムタグ付きテキスト
text = "こんにちは。(time:100)私は加藤です。(time:500)今日はITについて話します。"

# 音声ファイルを生成
output_file = quick_process(text, "output.wav", "http://your-gradio-server:7860/")
```

### 詳細なパラメータ指定

```python
from text_to_speech_processor import TextToSpeechProcessor

processor = TextToSpeechProcessor("http://your-gradio-server:7860/")

result = processor.process_text_to_speech(
    text="はじめまして。(time:500)よろしくお願いします。",
    output_path="greeting.wav",
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1
)
```

### バッチ処理

```python
from text_to_speech_processor import batch_process_texts

texts = [
    "第一章。(time:1000)はじめに。(time:500)基本概念を説明します。",
    "第二章。(time:1000)実装について。(time:500)具体的な方法を見ていきましょう。",
]

results = batch_process_texts(texts, output_dir="batch_outputs")
```

## 使用例の実行

```bash
python example.py
```

## タイムタグの形式

- `(time:100)` - 100ミリ秒の無音を挿入
- `(time:1000)` - 1秒の無音を挿入

例：
```
こんにちは。(time:100)私は加藤です。(time:500)今日はITについて話します。
```

## 前提条件

- Fish SpeechのGradio WebUIサーバーが起動していること
- サーバーのURLを正しく指定すること

## ライセンス

MIT License
