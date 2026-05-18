# Transformers — From-Scratch Implementations

From-scratch implementations of two transformer variants: a full encoder-decoder transformer
and a decoder-only (GPT-style) transformer. Both are built in PyTorch without high-level
sequence modelling abstractions.

This project is part of a larger sequential modelling project:
[sequential-modelling](https://github.com/Dorcas-Joy-Kahunguka/sequence-modelling)

---

## Motivation

The transformer architecture displaced recurrent models as the dominant approach to sequence
modelling by replacing sequential hidden state computation with parallelisable self-attention.
Building both the encoder-decoder and decoder-only variants from scratch makes the architectural
choices concrete — what the encoder contributes, why the decoder masks future tokens, and how
positional encoding substitutes for the implicit order information that recurrence provided
for free.

The attention mechanism here extends directly from the Bahdanau attention built in
[seq-models-01-rnn-to-lstm](https://github.com/Dorcas-Joy-Kahunguka/seq-models-01-rnn-to-lstm).
The conceptual thread is continuous: cross-attention in the encoder-decoder transformer is the
same weighted sum over encoder states, now computed via learned query-key dot products rather
than a learned alignment function.

---

## Implementations

### Encoder-decoder transformer

A full transformer following the original architecture introduced in
*Attention Is All You Need* (Vaswani et al., 2017). The encoder processes the full input
sequence via stacked multi-head self-attention and feed-forward layers. The decoder generates
output tokens autoregressively, attending to both its own prior outputs (masked self-attention)
and the full encoder output (cross-attention).

**Task:** sequence-to-sequence — translation or summarisation.

**Key components:**
- Multi-head self-attention (encoder and decoder)
- Masked multi-head self-attention (decoder)
- Cross-attention (decoder over encoder output)
- Position-wise feed-forward layers
- Positional encoding
- Residual connections and layer normalisation

### Decoder-only transformer

A GPT-style transformer consisting of stacked decoder blocks with causal (masked) self-attention.
No encoder. The model is trained as a causal language model — predicting the next token given
all prior tokens.

**Task:** causal language modelling — autoregressive text generation.

**Key components:**
- Masked multi-head self-attention
- Position-wise feed-forward layers
- Positional encoding
- Residual connections and layer normalisation

---

## Structure

```
02-transformers/
    notebooks/
        01-encoder-decoder-transformer.ipynb
        02-decoder-only-transformer.ipynb
    src/
        attention.py
        encoder.py
        decoder.py
        transformer.py
        positional-encoding.py
        train.py
        generate.py
    README.md
    environment.yml
    requirements.txt
```

---

## Notes

- Both implementations share a common attention module and feed-forward layer implementation.
- The decoder-only architecture provides direct grounding for the mechanistic interpretability
  work in
  [seq-models-03-mechinterp-typoglycemia](https://github.com/Dorcas-Joy-Kahunguka/seq-models-03-mechinterp-typoglycemia),
  which probes the internals of GPT-style models.
- All components are implemented from scratch. No use of `nn.Transformer` or equivalent
  high-level abstractions.

