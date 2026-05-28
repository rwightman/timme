import torch
import timm
import pytest

import timme
from timme.models.vision_transformer import VisionTransformer


TIPSV2_ARCHES = {
    'vit_base_patch14_reg1_tipsv2',
    'vit_large_patch14_reg1_tipsv2',
    'vit_so400m_patch14_reg1_tipsv2',
    'vit_giant_patch14_reg1_tipsv2',
}


def _create_timm_pretrained_or_skip(name, **kwargs):
    try:
        return timm.create_model(name, pretrained=True, num_classes=0, **kwargs)
    except Exception as exc:
        pytest.skip(f'pretrained weights for {name} are not available: {exc}')


def test_tipsv2_variants_are_registered():
    assert TIPSV2_ARCHES <= set(timme.list_models('*tipsv2*'))
    assert TIPSV2_ARCHES <= set(timme.list_models('*tipsv2*', pretrained=True))


def test_tipsv2_encoder_uses_factory_defaults():
    encoder = timme.create_encoder('vit_base_patch14_reg1_tipsv2')

    assert isinstance(encoder, VisionTransformer)
    # img_size + num_classes are sourced from the .webli pretrained_cfg
    assert encoder.patch_embed.img_size == (448, 448)
    assert encoder.pretrained_cfg['input_size'] == (3, 448, 448)
    assert encoder.pretrained_cfg['num_classes'] == 0
    # DINOv2-style: 1 register token + cls token, embedded after pos_embed
    assert encoder.num_prefix_tokens == 2
    assert encoder.no_embed_class
    assert encoder.reg_token.shape == (1, 1, 768)
    # LayerScale initialised at 1.0
    assert encoder.blocks[0].ls1.gamma.shape == (768,)


def test_tipsv2_giant_uses_swiglu_silu():
    encoder = timme.create_encoder('vit_giant_patch14_reg1_tipsv2', img_size=28)
    mlp = encoder.blocks[0].mlp

    assert type(mlp).__name__ == 'GluMlp'
    assert isinstance(mlp.act, torch.nn.SiLU)


@pytest.mark.parametrize(
    "name, embed_dim",
    (
        ('vit_base_patch14_reg1_tipsv2', 768),
        ('vit_large_patch14_reg1_tipsv2', 1024),
        ('vit_so400m_patch14_reg1_tipsv2', 1152),
        ('vit_giant_patch14_reg1_tipsv2', 1536),
    ),
)
def test_tipsv2_forward_shapes(name, embed_dim):
    encoder = timme.create_encoder(name, img_size=28).eval()

    with torch.no_grad():
        output = encoder(torch.randn(1, 3, 28, 28))

    # 2x2 patch grid (28 / 14) + cls + 1 register token
    assert output.shape == (1, 6, embed_dim)


@pytest.mark.parametrize(
    "name",
    (
        'vit_base_patch14_reg1_tipsv2',
        'vit_large_patch14_reg1_tipsv2',
        'vit_so400m_patch14_reg1_tipsv2',
    ),
)
@torch.no_grad()
def test_tipsv2_pretrained_forward_matches_timm(name):
    x = torch.randn(2, 3, 28, 28)

    timm_model = _create_timm_pretrained_or_skip(name, img_size=28)
    timme_encoder = timme.create_encoder(name, pretrained=True, img_size=28)
    timm_model.eval()
    timme_encoder.eval()

    expected = timm_model.forward_features(x)
    actual = timme_encoder(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
