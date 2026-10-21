"""Export ParakeetForTDT to two fp32 ONNX graphs (opset 18, TorchScript exporter).

encoder_model.onnx
  in   input_features  f32 [B, T, 128]   normalized log-mel (preprocessor_config.json)
       attention_mask  i64 [B, T]        1 for valid mel frames
  out  encoder_outputs f32 [B, T', 640]  encoder output after encoder_projector (the joint's encoder input)
       encoder_lengths i64 [B]           valid encoder frames, ceil-ish(T_valid / 8)

decoder_joint_model.onnx: one greedy TDT step (prediction network + joint)
  in   encoder_outputs f32 [B, 1, 640]   one frame of the encoder output
       input_ids       i64 [B, 1]        last emitted non-blank token (blank 8192 before the first)
       state_h, state_c f32 [2, B, 640]  LSTM state before input_ids (zeros at the start)
  out  token_logits    f32 [B, 8193]     8192 pieces + blank (id 8192)
       duration_logits f32 [B, 5]        over config.durations [0, 1, 2, 3, 4]
       state_h_out, state_c_out          LSTM state after input_ids

    python export.py --src hf --out raw
"""
import argparse, os
import torch
from torch import nn
from transformers import ParakeetForTDT

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

model = ParakeetForTDT.from_pretrained(a.src, dtype=torch.float32).eval()
cfg = model.config
V, H, L = cfg.vocab_size, cfg.decoder_hidden_size, cfg.num_decoder_layers


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_features, attention_mask):
        enc = self.m.encoder(input_features=input_features, attention_mask=attention_mask, output_attention_mask=True)
        return self.m.encoder_projector(enc.last_hidden_state), enc.attention_mask.sum(-1).to(torch.int64)


class DecoderJoint(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.dec, self.joint = m.decoder, m.joint

    def forward(self, encoder_outputs, input_ids, state_h, state_c):
        y, (h, c) = self.dec.lstm(self.dec.embedding(input_ids), (state_h, state_c))
        y = self.dec.decoder_projector(y)
        logits = self.joint(decoder_hidden_states=y, encoder_hidden_states=encoder_outputs)[:, 0]
        return logits[:, :V], logits[:, V:], h, c


with torch.inference_mode():
    torch.onnx.export(
        Encoder(model), (torch.randn(1, 801, 128), torch.ones(1, 801, dtype=torch.int64)),
        f"{a.out}/encoder_model.onnx",
        input_names=["input_features", "attention_mask"], output_names=["encoder_outputs", "encoder_lengths"],
        dynamic_axes={"input_features": {0: "batch", 1: "frames"}, "attention_mask": {0: "batch", 1: "frames"},
                      "encoder_outputs": {0: "batch", 1: "encoder_frames"}, "encoder_lengths": {0: "batch"}},
        opset_version=18, dynamo=False, external_data=True)
    torch.onnx.export(
        DecoderJoint(model),
        (torch.randn(1, 1, H), torch.full((1, 1), cfg.blank_token_id, dtype=torch.int64),
         torch.zeros(L, 1, H), torch.zeros(L, 1, H)),
        f"{a.out}/decoder_joint_model.onnx",
        input_names=["encoder_outputs", "input_ids", "state_h", "state_c"],
        output_names=["token_logits", "duration_logits", "state_h_out", "state_c_out"],
        dynamic_axes={"encoder_outputs": {0: "batch"}, "input_ids": {0: "batch"}, "state_h": {1: "batch"},
                      "state_c": {1: "batch"}, "token_logits": {0: "batch"}, "duration_logits": {0: "batch"},
                      "state_h_out": {1: "batch"}, "state_c_out": {1: "batch"}},
        opset_version=18, dynamo=False)
print("exported")
