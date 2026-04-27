"""
tagger_model.py
===============
Transformer модель для генерации тегов.
Используется и для обучения и для инференса.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TagTransformer(nn.Module):
    """
    Decoder-only Transformer (как GPT) для генерации тегов.

    Обучается предсказывать следующий тег по предыдущим.
    Одна модель работает в двух режимах:
      - Абстракт: на входе текстовый запрос (токенизированный как теги)
      - Аддитор:  на входе существующие теги + <SEP> токен
    """
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=6, dim_feedforward=1024,
                 max_seq_len=128, dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model    = d_model
        self.pad_id     = pad_id
        self.max_seq_len = max_seq_len

        # Эмбеддинги токенов
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer decoder layers (causal / autoregressive)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN — стабильнее обучается
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Выходной слой — предсказываем следующий токен
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Memory для decoder-only: используем нулевой тензор как "пустую память"
        # (это позволяет нам использовать TransformerDecoder как decoder-only)
        self.register_buffer("_empty_memory", torch.zeros(1, 1, d_model))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len, device):
        """Маска чтобы модель не видела будущие токены."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        """
        x: [batch, seq_len] — индексы токенов
        Возвращает логиты [batch, seq_len, vocab_size]
        """
        B, T = x.shape
        device = x.device

        # Padding mask
        pad_mask = (x == self.pad_id)  # [batch, seq_len]

        # Эмбеддинги + позиции
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)  # [batch, seq_len, d_model]

        # Causal mask
        causal_mask = self._causal_mask(T, device)

        # Пустая память для decoder-only режима
        memory = self._empty_memory.expand(B, -1, -1)

        out = self.transformer(
            tgt=emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )

        logits = self.output_proj(out)  # [batch, seq_len, vocab_size]
        return logits

    @torch.no_grad()
    def generate(self, input_ids, vocab,
                 max_new_tags=30,
                 temperature=1.0,
                 top_p=0.9,
                 forbidden_ids=None,
                 whitelist_ids=None):
        """
        Генерация тегов авторегрессивно.

        input_ids:    [1, seq_len] — начальные токены (BOS + входные теги)
        max_new_tags: сколько новых тегов генерировать
        temperature:  близость к запросу (низкая = предсказуемо, высокая = творчески)
        top_p:        разброс (низкий = близкие темы, высокий = далёкие)
        forbidden_ids: множество ID которые нельзя генерировать повторно
        """
        self.eval()
        device = input_ids.device

        bos_id = vocab["special_tokens"]["BOS"]
        eos_id = vocab["special_tokens"]["EOS"]
        pad_id = vocab["special_tokens"]["PAD"]

        generated    = input_ids.clone()
        forbidden    = set(forbidden_ids or [])
        forbidden.add(pad_id)

        new_tags_ids = []

        for _ in range(max_new_tags):
            # Обрезаем если слишком длинная последовательность
            inp = generated[:, -self.max_seq_len:]

            logits = self.forward(inp)           # [1, seq_len, vocab_size]
            next_logits = logits[0, -1, :]       # [vocab_size] — предсказание след. токена

            # Запрещаем уже сгенерированные и служебные токены
            already_generated = set(generated[0].tolist())
            for fid in (forbidden | already_generated):
                if fid < next_logits.size(0):
                    next_logits[fid] = float("-inf")

            # Whitelist — запрещаем всё что не в списке
            if whitelist_ids:
                mask = torch.ones_like(next_logits, dtype=torch.bool)
                for wid in whitelist_ids:
                    if wid < mask.size(0):
                        mask[wid] = False
                next_logits[mask] = float("-inf")

            # Запрещаем EOS пока не набрали нужное количество
            if len(new_tags_ids) < max_new_tags:
                next_logits[eos_id] = float("-inf")

            # Применяем температуру
            next_logits = next_logits / max(temperature, 1e-8)

            # Проверяем что хоть что-то доступно
            if (next_logits == float("-inf")).all():
                break  # все токены заблокированы — заканчиваем

            # Top-p (nucleus) sampling — контролирует разброс
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)

            # Убираем токены за пределами top-p
            cutoff = (cumsum - sorted_probs) >= top_p
            sorted_probs[cutoff] = 0.0

            # Если после cutoff ничего не осталось — берём топ-1
            if sorted_probs.sum() == 0:
                sorted_probs[0] = 1.0
            else:
                sorted_probs /= sorted_probs.sum()

            # Сэмплируем
            chosen_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_id    = sorted_ids[chosen_idx].view(1, 1)

            next_token = next_id.item()

            # Всегда запрещаем EOS пока не набрали нужное количество
            if next_token == eos_id:
                break

            new_tags_ids.append(next_token)
            generated = torch.cat([generated, next_id], dim=1)

        return new_tags_ids
