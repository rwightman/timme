import torch
import timm
import pytest

import timme
from timme.models.gemma4_vit import Gemma4Vit, Gemma4VitCfg


# timme uses the clean (encoder-first) names; timm stores the weights under '_enc'.
GEMMA4_ARCHES = {'gemma4_vit_167m', 'gemma4_vit_570m'}
TIMM_ENC = {'gemma4_vit_167m': 'gemma4_vit_167m_enc', 'gemma4_vit_570m': 'gemma4_vit_570m_enc'}


def _timm_enc_state_dict(timme_name):
    """Random-init timm encoder weights — used to prove forward parity without download."""
    return timm.create_model(TIMM_ENC[timme_name], pretrained=False).eval().state_dict()


def test_gemma4_variants_are_registered():
    assert GEMMA4_ARCHES <= set(timme.list_models('gemma4*'))
    assert GEMMA4_ARCHES <= set(timme.list_models('gemma4*', pretrained=True))
    # The timm '_enc' names are NOT timme arch names.
    assert 'gemma4_vit_167m_enc' not in timme.list_models('gemma4*')


def test_gemma4_pretrained_lookup_is_aliased_to_enc_repo():
    # Clean timme name must resolve its pretrained_cfg from timm's '_enc' repo.
    encoder = timme.create_encoder('gemma4_vit_167m')
    assert encoder.pretrained_cfg['hf_hub_id'] == 'timm/gemma4_vit_167m_enc.gemma4_e4b_it'
    # Tag is preserved through the alias.
    tagged = timme.create_encoder('gemma4_vit_167m.gemma4_e4b_it')
    assert tagged.pretrained_cfg['hf_hub_id'] == 'timm/gemma4_vit_167m_enc.gemma4_e4b_it'
    # A string pretrained_cfg override naming the timme arch honors the alias too.
    override = timme.create_encoder('gemma4_vit_167m', pretrained_cfg='gemma4_vit_167m')
    assert override.pretrained_cfg['hf_hub_id'] == 'timm/gemma4_vit_167m_enc.gemma4_e4b_it'


def test_gemma4_encoder_uses_factory_defaults():
    encoder = timme.create_encoder('gemma4_vit_167m')

    assert isinstance(encoder, Gemma4Vit)
    assert encoder.output_fmt == 'NLC'
    assert encoder.output_dim == 768
    assert encoder.num_prefix_tokens == 0
    assert encoder.output_pool == 'soft'
    assert encoder.patch_size == (16, 16)
    # 167m (E4B) ships clamp buffers; no standardization.
    assert encoder.use_clipped_linears is True
    assert encoder.std_bias is None
    assert encoder.pretrained_cfg['input_size'] == (3, 768, 768)
    assert encoder.pretrained_cfg['num_classes'] == 0


def test_gemma4_570m_has_standardization_buffers():
    encoder = timme.create_encoder('gemma4_vit_570m')
    assert encoder.std_bias is not None and encoder.std_scale is not None
    assert encoder.std_bias.shape == (1152,)


def test_gemma4_cfg_json_round_trip():
    cfg = Gemma4VitCfg(embed_dim=768, depth=16, standardize=True)
    assert Gemma4VitCfg.from_json(cfg.to_json()) == cfg


@pytest.mark.parametrize(
    "output_pool, expected",
    (
        ('soft', (2, 4, 768)),   # 6x6 grid -> 2x2 soft tokens
        ('avg', (2, 768)),
        ('none', (2, 36, 768)),
    ),
)
def test_gemma4_forward_shapes(output_pool, expected):
    encoder = timme.create_encoder('gemma4_vit_167m', output_pool=output_pool).eval()
    with torch.no_grad():
        out = encoder(torch.randn(2, 3, 96, 96))
    assert out.shape == expected


def test_gemma4_soft_pool_requires_conformant_size():
    encoder = timme.create_encoder('gemma4_vit_167m').eval()
    # 64 is not divisible by patch_size * pooling_kernel_size = 48.
    with pytest.raises(ValueError, match='divisible'):
        encoder(torch.randn(1, 3, 64, 64))


@torch.no_grad()
def test_gemma4_167m_forward_matches_timm_all_pools():
    """Bit-exact parity vs timm for every pool mode (weights copied, no download)."""
    name, timm_name = 'gemma4_vit_167m', 'gemma4_vit_167m_enc'
    sd = _timm_enc_state_dict(name)
    x = torch.randn(2, 3, 96, 96)

    for pool in ('soft', 'avg', 'none'):
        tm = timm.create_model(timm_name, pretrained=False, global_pool=pool).eval()
        tm.load_state_dict(sd, strict=False)  # pool carries no params
        enc = timme.create_encoder(name, output_pool=pool).eval()
        # State-dict keys must match timm exactly for checkpoints to load.
        assert set(enc.state_dict()) == set(sd)
        enc.load_state_dict(sd, strict=True)
        torch.testing.assert_close(enc(x), tm(x), rtol=0, atol=0)


@torch.no_grad()
def test_gemma4_intermediates_match_timm():
    name, timm_name = 'gemma4_vit_167m', 'gemma4_vit_167m_enc'
    sd = _timm_enc_state_dict(name)
    x = torch.randn(2, 3, 96, 96)

    tm = timm.create_model(timm_name, pretrained=False).eval()
    tm.load_state_dict(sd)
    encoder = timme.create_encoder(name, out_indices=(3, 7, 11)).eval()
    encoder.load_state_dict(sd, strict=True)

    expected = tm.forward_intermediates(x, indices=[3, 7, 11], intermediates_only=True, output_fmt='NCHW')
    actual = encoder(x)
    assert len(actual) == 3
    assert all(t.shape == (2, 768, 6, 6) for t in actual)
    assert all(torch.equal(p, q) for p, q in zip(expected, actual))


@torch.no_grad()
def test_gemma4_naflex_padded_input_matches_timm():
    """Pre-patchified NaFlex dict input with padding — exercises attn-mask + masked pool."""
    from timm.models.naflexvit import batch_patchify

    name, timm_name = 'gemma4_vit_167m', 'gemma4_vit_167m_enc'
    sd = _timm_enc_state_dict(name)

    B, ph, pw, C = 2, 16, 16, 3
    img = torch.randn(B, C, 96, 96)
    patches, _ = batch_patchify(img, (ph, pw), pad=False, channels_last=False)
    N = patches.shape[1]
    patches_ppc = patches.view(B, N, C, ph, pw).permute(0, 1, 3, 4, 2).reshape(B, N, ph * pw * C)
    pH = pW = 96 // 16
    ys, xs = torch.meshgrid(torch.arange(pH), torch.arange(pW), indexing='ij')
    coord = torch.stack([ys.flatten(), xs.flatten()], dim=-1).unsqueeze(0).expand(B, -1, -1).contiguous()
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[1, -9:] = False           # still divisible by k^2 = 9
    coord[1, -9:] = -1
    inp = dict(patches=patches_ppc, patch_coord=coord, patch_valid=valid)

    tm = timm.create_model(timm_name, pretrained=False).eval()
    tm.load_state_dict(sd)
    enc = timme.create_encoder(name).eval()
    enc.load_state_dict(sd, strict=True)

    torch.testing.assert_close(enc(dict(inp)), tm(dict(inp)), rtol=0, atol=0)
    # dict input and explicit kwargs are equivalent
    assert torch.equal(enc(dict(inp)), enc(patches_ppc, patch_coord=coord, patch_valid=valid))


def test_gemma4_classifier_builds_and_runs():
    model = timme.create_model('gemma4_vit_167m', num_classes=10, img_size=96).eval()
    # Encoder native pool is bypassed for the classifier (raw patch tokens);
    # the head owns pooling. These are independent knobs.
    assert model.encoder.output_pool == 'none'
    assert model.head.pool_type == 'avg'
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 96, 96))
    assert logits.shape == (2, 10)


def test_gemma4_classifier_global_pool_is_head_side():
    # create_model's global_pool drives the head, never the encoder's output_pool.
    model = timme.create_model('gemma4_vit_167m', num_classes=10, global_pool='max', img_size=96)
    assert model.head.pool_type == 'max'
    assert model.encoder.output_pool == 'none'


@torch.no_grad()
def test_gemma4_pretrained_forward_matches_timm():
    name, timm_name = 'gemma4_vit_167m', 'gemma4_vit_167m_enc'
    try:
        timm_model = timm.create_model(timm_name, pretrained=True).eval()
    except Exception as exc:
        pytest.skip(f'pretrained weights for {timm_name} are not available: {exc}')
    # Clean timme name pulls the same '_enc' weights via the pretrained alias.
    encoder = timme.create_encoder(name, pretrained=True).eval()

    x = torch.randn(2, 3, 96, 96)
    torch.testing.assert_close(encoder(x), timm_model(x), rtol=0, atol=0)
