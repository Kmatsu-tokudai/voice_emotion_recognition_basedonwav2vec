import pandas as pd
import torchaudio
import torch
import librosa
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import Wav2Vec2Model
from torch.utils.data import random_split, DataLoader
from transformers import Wav2Vec2FeatureExtractor

# 感情ラベルのマッピング
EMOTION_LABELS = {'ang': 0, 'joy': 1, 'dis': 2, 'fea': 3, 'neu': 4, 'sad': 5, 'sur': 6}
EMOS = ['ang', 'joy', 'dis', 'fea', 'neu', 'sad', 'sur']


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("rinna/japanese-wav2vec2-base")

class EmotionDataset(Dataset):
    def __init__(self, df, feature_extractor, audio_dir):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        label = EMOTION_LABELS[row["emotion"]]
        waveform, sr = torchaudio.load(path)
        waveform = librosa.resample(waveform.numpy()[0], orig_sr=sr, target_sr=16000)
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs["labels"] = torch.tensor(label)
        return inputs


class EmotionClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("rinna/japanese-wav2vec2-base")
        self.classifier = torch.nn.Linear(self.wav2vec2.config.hidden_size, len(EMOTION_LABELS))

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        pooled = hidden_states.mean(dim=1)         # 平均プーリング [B, H]
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        input_values = batch["input_values"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]

        logits = self(input_values, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_values = batch["input_values"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]

        logits = self(input_values, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer


# 推論関数
def predict_emotion(audio_path, model, feature_extractor, device):
    # 音声読み込み
    waveform, sample_rate = torchaudio.load(audio_path)

    # サンプリングレート変換（必要に応じて）
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # モノラル変換
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 入力処理
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # モデル推論
    with torch.no_grad():
        logits = model(**inputs)
        probs = F.softmax(logits, dim=1)
        pred_label = EMOS[probs.argmax(dim=1).item()]
        return pred_label, probs.squeeze().tolist()



def predict_sample():
    # TSVファイルの読み込み
    df = pd.read_csv("sample_metadata.tsv", sep="\t")

    checkpoint = "lightning_logs/version_0/checkpoints/epoch=9-step=47390.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier.load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    wf = open('./result_of_samples.txt', 'w')
    wf.write('path\tcps_type\tcorr_emotion\tpred_emotion\tprobability\ttext\n')
    for i in range(len(df)):
        txt = df.iloc[i]['text']
        lab = df.iloc[i]['emotion']
        sp = df.iloc[i]['cps_type']        
        wavpath = df.iloc[i]['path']

        audio_file = wavpath  # ここに予測したい音声ファイルのパスを指定
        label, probabilities = predict_emotion(audio_file, model, feature_extractor, device)
        pros = ''
        for i, prob in enumerate(probabilities):
            pros +=  f'{EMOS[i]}:{prob:.4f} '
        pros = pros.rstrip(' ')
        wf.write(f'{wavpath}\t{sp}\t{lab}\t{label}\t{pros}\t{txt}\n')

    wf.close


if __name__ == '__main__':
    predict_sample()
