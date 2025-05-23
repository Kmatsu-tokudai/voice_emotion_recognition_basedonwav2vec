# voice_emotion_recognition_basedonwav2vec
Wav2Vec2.0の日本語事前学習済みモデルをベースに，ラベル付き日本語感情音声コーパスをpytorch-lightningで学習したモデル

テストデータとして，日本声優統計学会の公開している音声ファイルを使用してモデル性能を確認。

https://voice-statistics.github.io/


    | precision | recall | f1-score | support 
--- | --- | --- | --- | --- 
ang | 1.00 | 0.56 | 0.72 | 225
joy | 0.84 | 0.99 | 0.91 | 299
neu | 0.38 | 1.00 | 0.55 | 28
accuracy | 0.82 | 0.82 | 0.82 | 0.82
macro avg | 0.74 | 0.85 | 0.73 | 552
weighted avg | 0.88 | 0.82 | 0.82 | 552
