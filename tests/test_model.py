import torch

from lora_llm.model import ModelConfig, TinyTransformer


def _model() -> TinyTransformer:
    torch.manual_seed(0)
    return TinyTransformer(
        ModelConfig(vocab_size=50, d_model=32, n_layer=2, n_head=4, d_ff=64, max_len=24)
    )


def test_forward_shapes_and_loss():
    model = _model()
    x = torch.randint(0, 50, (3, 12))
    logits, loss = model(x, targets=x)
    assert logits.shape == (3, 12, 50)
    assert loss is not None and loss.item() > 0


def test_generate_extends_sequence():
    model = _model()
    prompt = torch.randint(0, 50, (1, 4))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape[1] == 9


def test_generate_stops_at_eos():
    model = _model()
    # Force EOS as the argmax: make the final hidden state a constant
    # ones-vector (via ln_f affine) and point only row 7 of the head at it.
    with torch.no_grad():
        model.ln_f.weight.zero_()
        model.ln_f.bias.fill_(1.0)
        model.lm_head.weight.zero_()
        model.lm_head.weight[7].fill_(1.0)
    prompt = torch.randint(0, 50, (1, 3))
    out = model.generate(prompt, max_new_tokens=10, eos_id=7)
    assert out.shape[1] == 4  # one generated token then stop
    assert int(out[0, -1]) == 7


def test_param_count_positive():
    model = _model()
    assert model.num_parameters() > 10_000
