import torch
import timm
import pytest

import timme
from timme.models.eva import Eva, _eva_filter_fn


DINO_V3_EVA_ARCHES = {
    'vit_7b_patch16_dinov3',
    'vit_base_patch16_dinov3',
    'vit_base_patch16_dinov3_qkvb',
    'vit_huge_plus_patch16_dinov3',
    'vit_huge_plus_patch16_dinov3_qkvb',
    'vit_large_patch16_dinov3',
    'vit_large_patch16_dinov3_qkvb',
    'vit_small_patch16_dinov3',
    'vit_small_patch16_dinov3_qkvb',
    'vit_small_plus_patch16_dinov3',
    'vit_small_plus_patch16_dinov3_qkvb',
    'vit_tiny_patch16_dinov3_qkvb',
}


def _create_timm_pretrained_or_skip(name):
    try:
        return timm.create_model(name, pretrained=True, img_size=32)
    except Exception as exc:
        pytest.skip(f'pretrained weights for {name} are not available: {exc}')


def test_dinov3_eva_variants_are_registered():
    assert DINO_V3_EVA_ARCHES <= set(timme.list_models('*dinov3*'))
    assert DINO_V3_EVA_ARCHES <= set(timme.list_models('*dinov3*', pretrained=True))


def test_dinov3_eva_encoder_uses_factory_defaults():
    encoder = timme.create_encoder('vit_small_patch16_dinov3')

    assert isinstance(encoder, Eva)
    assert encoder.patch_embed.img_size == (256, 256)
    assert encoder.pretrained_cfg['input_size'] == (3, 256, 256)
    assert encoder.pretrained_cfg['num_classes'] == 0
    assert encoder.dynamic_img_size
    assert encoder.pos_embed is None
    assert encoder.num_prefix_tokens == 5
    assert encoder.reg_token.shape == (1, 4, 384)
    assert encoder.cfg.norm_eps == 1e-5
    assert encoder.blocks[0].norm1.eps == 1e-5


def test_dinov3_eva_defaults_to_token_pool():
    # timme intentionally diverges from timm here: timm's dinov3 builders leave
    # global_pool at 'avg', but these models use the CLS token.
    encoder = timme.create_encoder('vit_small_patch16_dinov3', img_size=32)
    assert encoder.cfg.global_pool == 'token'

    model = timme.create_model('vit_small_patch16_dinov3', num_classes=10, img_size=32)
    assert model.head.pool_type == 'token'
    # token pool -> no fc_norm in the head (final norm stays on the encoder)
    assert encoder._head_uses_fc_norm is False


def test_dinov3_eva_small_forward_with_rope_and_register_tokens():
    encoder = timme.create_encoder('vit_small_patch16_dinov3', img_size=32)
    encoder.eval()

    with torch.no_grad():
        output = encoder(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 9, 384)


@torch.no_grad()
@pytest.mark.parametrize(
    "name",
    (
        'vit_tiny_patch16_dinov3_qkvb',
        'vit_small_patch16_dinov3',
        'vit_small_patch16_dinov3_qkvb',
        'vit_small_plus_patch16_dinov3',
        'vit_base_patch16_dinov3',
    ),
)
def test_dinov3_eva_pretrained_forward_matches_timm(name):
    x = torch.randn(2, 3, 32, 32)

    timm_model = _create_timm_pretrained_or_skip(name)
    timme_encoder = timme.create_encoder(name, pretrained=True, img_size=32)
    timm_model.eval()
    timme_encoder.eval()

    expected = timm_model.forward_features(x)
    actual = timme_encoder(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dinov3_eva_qkvb_variant_has_separate_qv_bias():
    encoder = timme.create_encoder('vit_small_patch16_dinov3_qkvb', img_size=32)
    attn_params = dict(encoder.blocks[0].attn.named_parameters())

    assert 'q_bias' in attn_params
    assert 'v_bias' in attn_params
    assert attn_params['q_bias'].shape == (384,)
    assert attn_params['v_bias'].shape == (384,)


def test_dinov3_eva_checkpoint_filter_renames_storage_tokens_and_qv_bias():
    encoder = timme.create_encoder('vit_small_patch16_dinov3_qkvb', img_size=32)
    state_dict = {
        'storage_tokens': torch.zeros(1, 4, 384),
        'blocks.0.attn.qkv.weight': torch.zeros(1152, 384),
        'blocks.0.attn.qkv.bias': torch.arange(1152, dtype=torch.float32),
        'blocks.0.ls1.gamma': torch.ones(384),
        'blocks.0.ls2.gamma': torch.ones(384),
        'mask_token': torch.zeros(1, 1, 384),
        'local_cls_norm.weight': torch.ones(384),
    }

    filtered = _eva_filter_fn(state_dict, encoder)

    assert 'reg_token' in filtered
    assert 'blocks.0.attn.q_bias' in filtered
    assert 'blocks.0.attn.v_bias' in filtered
    assert 'blocks.0.gamma_1' in filtered
    assert 'blocks.0.gamma_2' in filtered
    assert 'mask_token' not in filtered
    assert 'local_cls_norm.weight' not in filtered
    assert torch.equal(filtered['blocks.0.attn.q_bias'], torch.arange(384, dtype=torch.float32))
    assert torch.equal(filtered['blocks.0.attn.v_bias'], torch.arange(768, 1152, dtype=torch.float32))
