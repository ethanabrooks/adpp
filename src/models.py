import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from encoder import Encoder


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class EinLinear(nn.Module):
    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
        input : [ B x n_models x input_dim ]
        """
        ## [ B x n_models x output_dim ]
        output = torch.einsum("eoi,bei->beo", self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        attn_pdrop: float,
        context_size: int,
        n_embd: int,
        n_head: int,
        resid_pdrop: float,
        step_dim: int,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_size, context_size)).view(
                1, 1, context_size, context_size
            ),
        )
        ## mask previous value estimates
        self.mask.squeeze()[:, step_dim - 1 :: step_dim] = 0
        ##
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        ## [ B x n_heads x T x head_dim ]
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        ## [ B x n_heads x T x T ]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        ## [ B x n_heads x T x head_size ]
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ## [ B x T x embedding_dim ]
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class TransformerLayer(nn.Module):
    def __init__(
        self,
        causal_self_attention_args: dict,
        n_embd: int,
        resid_pdrop: float,
        **kwargs,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            **causal_self_attention_args,
            **kwargs,
            n_embd=n_embd,
            resid_pdrop=resid_pdrop,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        embd_pdrop: float,
        layer_args: dict,
        n_embd: int,
        n_head: int,
        n_layer: int,
        n_tokens: int,
        step_dim: int,
        steps_per_context: int,
    ):
        super().__init__()
        self.encoder = encoder
        context_size = steps_per_context * step_dim - 1

        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(n_tokens * step_dim + 1, n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, context_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        # Use Huggingface's GPT2:
        config = GPT2Config(
            vocab_size=n_tokens + 1,
            n_positions=context_size,
            n_ctx=context_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            # Add other configuration parameters if needed
        )
        self.gpt2_model = GPT2Model(config)
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(context_size * n_embd, context_size * n_embd),
        )
        self.head2 = nn.Linear(n_embd, n_tokens + 1, bias=False)
        self.head = EinLinear(step_dim, n_embd, n_tokens + 1, bias=False)

        self.vocab_size = n_tokens
        self.stop_token = n_tokens * step_dim
        self._context_size = context_size
        self.transition_dim = step_dim

        self.embedding_dim = n_embd
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def offset_tokens(self, idx):
        _, t = idx.shape
        n_states = int(np.ceil(t / self.transition_dim))
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(idx.device)
        offset_idx = idx + offsets[:t]
        offset_idx[idx == self.vocab_size] = self.stop_token
        return offset_idx

    def pad_to_full_observation(self, x, verify=False):
        b, t, _ = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
        padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device)
        ## [ B x T' x embedding_dim ]
        x_pad = torch.cat([x, padding], dim=1)
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
        if verify:
            self.verify(x, x_pad)
        return x_pad, n_pad

    def verify(self, x, x_pad):
        b, t, embedding_dim = x.shape
        n_states = int(np.ceil(t / self.transition_dim))
        inds = torch.arange(0, self.transition_dim).repeat(n_states)[:t]
        for i in range(self.transition_dim):
            x_ = x[:, inds == i]
            t_ = x_.shape[1]
            x_pad_ = x_pad[:, i].view(b, n_states, embedding_dim)[:, :t_]
            print(i, x_.shape, x_pad_.shape)
            assert (x_ == x_pad_).all()

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor = None,
        weights: torch.Tensor = None,
    ):
        """
        sequence : [ B x (T+1) ]
        """
        inputs = sequence[:, :-1].contiguous()
        targets = sequence[:, 1:].contiguous()
        original_inputs = inputs

        inputs = self.encoder.encode(inputs)

        b, t = inputs.size()
        assert t <= self._context_size, "Cannot forward, model block size is exhausted."

        # offset_idx = self.offset_tokens(inputs)
        ## [ B x T x embedding_dim ]
        # forward the GPT model
        token_embeddings = self.tok_emb(
            inputs
        )  # each index maps to a (learnable) vector
        ## [ 1 x T x embedding_dim ]
        # position_embeddings = self.pos_emb[
        #     :, :t, :
        # ]  # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        # x = self.drop(token_embeddings + position_embeddings)
        x = self.gpt2_model(inputs_embeds=token_embeddings).last_hidden_state
        ## [ B x T x embedding_dim ]
        # x = self.ln_f(x)
        # x = self.mlp(x.view(b, -1))
        # x = x.reshape(b, t, -1)

        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        # x_pad, n_pad = self.pad_to_full_observation(x)
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        logits = self.head2(x)
        ## [ B x T' x (vocab_size + 1) ]
        # logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        # logits = logits[:, :t]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            self.encoder.encode(targets).view(-1),
            reduction="none",
        )

        # if we are given some desired targets also calculate the loss
        if weights is None:
            loss = loss.mean()
        else:
            loss = loss * weights[:, 1:].reshape(-1)
            if mask is not None:
                mask = mask[:, 1:].contiguous()
                loss = (loss * mask.view(-1)).mean()
                # if inputs[mask == 1].unique().numel() > 1:

        return logits, loss

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        prediction = torch.multinomial(probs, num_samples=1)
        return self.encoder.decode(prediction)


class NextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, n_embd=768, n_layer=12, n_head=12):
        super(NextTokenPredictor, self).__init__()

        # Initialize GPT-2 config
        self.config = GPT2Config(
            vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head
        )

        # Initialize GPT-2 model
        self.gpt2 = GPT2LMHeadModel(self.config)

    def forward(self, sequence, mask=None):
        """
        sequence: a batch of input sequences. Shape: [batch_size, seq_length]
        mask: an optional attention mask. Shape: [batch_size, seq_length]
        """
        sequence = sequence.long()

        # Get logits
        outputs = self.gpt2(input_ids=sequence[:, :-1])
        logits = outputs.logits

        # Shift the input_ids and mask to the right for next token prediction
        shifted_input_ids = sequence[:, 1:]
        shifted_mask = mask[:, 1:] if mask is not None else None

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), shifted_input_ids.reshape(-1)
        )

        return logits, loss
