import re
import numpy as np
import soundfile as sf
from gradio_client import Client
from typing import List, Tuple, Union
import os

class TextToSpeechProcessor:
    def __init__(self, gradio_url: str = "http://127.0.0.1:7860/"):
        """
        テキストを音声に変換し、タイムタグに基づいて結合するプロセッサ
        
        Args:
            gradio_url: GradioサーバーのURL
        """
        self.client = Client(gradio_url)
        self.default_sample_rate = 44100  # デフォルトサンプルレート
        
    def parse_text_with_timing(self, text: str) -> List[Tuple[str, Union[str, int]]]:
        """
        テキストをタイムタグで分割し、(content, type)のリストを返す
        
        Args:
            text: 分割するテキスト（例: "こんにちは。(time:100)私は加藤です。"）
            
        Returns:
            List[Tuple[str, Union[str, int]]]: [("こんにちは。", "text"), (100, "time"), ...]
        """
        # (time:数字) パターンでマッチ
        pattern = r'\(time:(\d+)\)'
        
        # テキストを分割
        parts = re.split(pattern, text)
        
        result = []
        for i, part in enumerate(parts):
            if not part:  # 空文字列はスキップ
                continue
                
            if i % 2 == 0:  # 偶数インデックスはテキスト
                if part.strip():  # 空白のみでない場合
                    result.append((part.strip(), "text"))
            else:  # 奇数インデックスは時間（ms）
                result.append((int(part), "time"))
                
        return result
    
    def generate_speech(self, text: str, **generation_params) -> str:
        """
        テキストから音声を生成してファイルパスを返す
        
        Args:
            text: 音声化するテキスト
            **generation_params: 音声生成パラメータ
            
        Returns:
            str: 生成された音声ファイルのパス
        """
        # デフォルトパラメータ
        params = {
            "max_new_tokens": 1024,
            "chunk_length": 300,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "temperature": 0.8,
            "seed": 0,
        }
        params.update(generation_params)
        
        try:
            # client.predict()はwav_file_pathと不要な値を返す
            wav_file_path, _ = self.client.predict(
                text,                          # text
                "",                           # reference_id
                None,                         # reference_audio
                "",                           # reference_text
                params["max_new_tokens"],     # max_new_tokens
                params["chunk_length"],       # chunk_length
                params["top_p"],              # top_p
                params["repetition_penalty"], # repetition_penalty
                params["temperature"],        # temperature
                params["seed"],               # seed
                "on",                         # use_memory_cache
                api_name="/predict"
            )
            
            return wav_file_path
            
        except Exception as e:
            print(f"音声生成に失敗しました: {e}")
            raise
    
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        音声ファイルを読み込む
        
        Args:
            file_path: 音声ファイルのパス
            
        Returns:
            Tuple[np.ndarray, int]: (audio_array, sample_rate)
        """
        try:
            audio_array, sample_rate = sf.read(file_path)
            return audio_array, sample_rate
        except Exception as e:
            print(f"音声ファイルの読み込みに失敗しました: {e}")
            raise
    
    def generate_silence(self, duration_ms: int, sample_rate: int) -> np.ndarray:
        """
        指定された時間の無音を生成
        
        Args:
            duration_ms: 無音の長さ（ミリ秒）
            sample_rate: サンプルレート
            
        Returns:
            np.ndarray: 無音の音声データ
        """
        samples = int(duration_ms * sample_rate / 1000)
        return np.zeros(samples, dtype=np.float32)
    
    def process_text_to_speech(
        self, 
        text: str, 
        output_path: str = "combined_output.wav",
        **generation_params
    ) -> str:
        """
        タイムタグ付きテキストを処理して音声ファイルを生成
        
        Args:
            text: 処理するテキスト（タイムタグ含む）
            output_path: 出力ファイルパス
            **generation_params: 音声生成パラメータ
            
        Returns:
            str: 出力ファイルパス
        """
        print(f"入力テキスト: {text}")
        
        # テキストを分割
        segments = self.parse_text_with_timing(text)
        print(f"分割結果: {segments}")
        
        # 音声データを格納するリスト
        audio_segments = []
        sample_rate = None
        temp_files = []  # 一時ファイルの管理
        
        try:
            for i, (content, segment_type) in enumerate(segments):
                if segment_type == "text":
                    print(f"音声生成中 ({i+1}/{len(segments)}): {content}")
                    
                    # 音声生成（ファイルパスが返される）
                    wav_file_path = self.generate_speech(content, **generation_params)
                    temp_files.append(wav_file_path)
                    
                    # 音声ファイルを読み込み
                    audio_array, sr = self.load_audio_file(wav_file_path)
                    
                    if sample_rate is None:
                        sample_rate = sr
                    elif sample_rate != sr:
                        print(f"警告: サンプルレートが異なります ({sample_rate} vs {sr})")
                    
                    audio_segments.append(audio_array)
                    print(f"音声生成完了: {len(audio_array)} samples")
                    
                elif segment_type == "time":
                    if sample_rate is None:
                        sample_rate = self.default_sample_rate
                    
                    print(f"無音追加: {content}ms")
                    silence = self.generate_silence(content, sample_rate)
                    audio_segments.append(silence)
                    print(f"無音生成完了: {len(silence)} samples")
            
            # 全ての音声セグメントを結合
            if not audio_segments:
                raise ValueError("音声セグメントが見つかりません")
            
            print("音声データを結合中...")
            combined_audio = np.concatenate(audio_segments)
            
            # ファイルに保存
            sf.write(output_path, combined_audio, sample_rate)
            
            total_duration = len(combined_audio) / sample_rate
            print(f"音声結合完了！")
            print(f"出力ファイル: {output_path}")
            print(f"総再生時間: {total_duration:.2f}秒")
            print(f"サンプルレート: {sample_rate}Hz")
            
        finally:
            # 一時ファイルを削除
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"一時ファイルを削除: {temp_file}")
                except Exception as e:
                    print(f"一時ファイルの削除に失敗: {temp_file}, エラー: {e}")
        
        return output_path


# 使用例とヘルパー関数
def quick_process(text: str, output_file: str = "output.wav", gradio_url: str = "http://127.0.0.1:7860/"):
    """
    シンプルなワンライナー処理関数
    """
    processor = TextToSpeechProcessor(gradio_url)
    return processor.process_text_to_speech(text, output_file)


def batch_process_texts(
    texts: List[str], 
    output_dir: str = "batch_outputs",
    gradio_url: str = "http://127.0.0.1:7860/",
    **generation_params
):
    """
    複数のタイムタグ付きテキストを一括処理
    
    Args:
        texts: 処理するテキストのリスト
        output_dir: 出力ディレクトリ
        gradio_url: GradioサーバーのURL
        **generation_params: 音声生成パラメータ
    """
    os.makedirs(output_dir, exist_ok=True)
    processor = TextToSpeechProcessor(gradio_url)
    
    results = []
    for i, text in enumerate(texts):
        print(f"\n=== バッチ処理 {i+1}/{len(texts)} ===")
        output_path = os.path.join(output_dir, f"batch_{i+1:03d}.wav")
        
        try:
            result_path = processor.process_text_to_speech(
                text, 
                output_path, 
                **generation_params
            )
            results.append(result_path)
            print(f"✅ 完了: {result_path}")
        except Exception as e:
            print(f"❌ エラー: {e}")
            results.append(None)
    
    success_count = len([r for r in results if r])
    print(f"\n🎉 バッチ処理完了: {success_count}/{len(texts)} 成功")
    return results


if __name__ == "__main__":
    # 基本的な使用例
    text_input = "こんにちは。(time:100)私は加藤です。(time:500)今日はITについて話します。"
    
    processor = TextToSpeechProcessor()
    
    try:
        output_file = processor.process_text_to_speech(
            text_input,
            output_path="combined_speech.wav",
            temperature=0.8,
            top_p=0.8
        )
        print(f"✅ 処理完了: {output_file}")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    
    # 複数の例を試す
    test_cases = [
        "はじめまして。(time:200)よろしくお願いします。",
        "第一章。(time:1000)音声合成について。(time:500)それでは始めましょう。",
        "おはようございます。(time:100)今日は(time:50)良い天気ですね。(time:300)いかがお過ごしでしょうか。"
    ]
    
    for i, test_text in enumerate(test_cases):
        print(f"\n--- テストケース {i+1} ---")
        try:
            output_file = quick_process(test_text, f"test_output_{i+1}.wav")
            print(f"✅ テストケース {i+1} 完了: {output_file}")
        except Exception as e:
            print(f"❌ テストケース {i+1} エラー: {e}")
