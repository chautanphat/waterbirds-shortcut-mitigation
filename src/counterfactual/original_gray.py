"""Build a genuinely paired gray view from benchmark RGB, never CUB RGB.

Spatial preprocessing is shared before alpha composition so opaque model-view
pixels are exactly unchanged. Soft alpha uses the historical bilinear coverage.
"""
import numpy as np
from PIL import Image,ImageOps,ImageFilter

from .common import pixel_hash

ORIENTATION={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,
    4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,
    6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}


def decode_pair(image_path,mask_path):
    with Image.open(image_path) as source:
        original_mode=source.mode;orientation=int(source.getexif().get(274,1))
        raw=source.convert('RGB');raw_hash=pixel_hash(raw)
        rgb=ImageOps.exif_transpose(source).convert('RGB')
    with Image.open(mask_path) as source:
        if source.mode not in ['L','1']:raise ValueError('Mask is not grayscale coverage')
        mask=source.convert('L')
    if raw.size!=mask.size:raise ValueError('Native image/mask dimensions differ; no guessed mask resize allowed')
    if orientation in ORIENTATION:mask=mask.transpose(ORIENTATION[orientation])
    elif orientation!=1:raise ValueError('Unknown EXIF orientation')
    if rgb.size!=mask.size:raise ValueError('Transposed mask alignment differs')
    return rgb,mask,dict(original_mode=original_mode,orientation=orientation,native_width=raw.width,
        native_height=raw.height,source_decoded_pixel_sha256=raw_hash)


def build_gray(rgb,mask,gray_rgb=(127,127,127)):
    """Only RGB/alpha/constants enter generation; no class/group/source flags."""
    if rgb.mode!='RGB' or mask.mode!='L' or rgb.size!=mask.size:raise ValueError('Aligned RGB/L mask required')
    values=np.asarray(mask)
    if values.max()==0 or values.min()>0:raise ValueError('Empty or full-photo mask')
    # Same square256/bilinear geometry as Runner; no foreground crop/reposition.
    full256=rgb.resize((256,256),Image.Resampling.BILINEAR)
    alpha256=mask.resize((256,256),Image.Resampling.BILINEAR)
    layer=full256.convert('RGBA');layer.putalpha(alpha256)
    gray256=Image.alpha_composite(Image.new('RGBA',(256,256),tuple(gray_rgb)+(255,)),layer).convert('RGB')
    full=full256.crop((16,16,240,240));gray=gray256.crop((16,16,240,240));alpha=alpha256.crop((16,16,240,240))
    a=np.asarray(alpha);f=np.asarray(full);g=np.asarray(gray)
    opaque=a==255;outside=a==0;soft=(a>0)&(a<255)
    # Erosion is audit-only, never used to alter the inference mask.
    eroded=np.asarray(Image.fromarray(opaque.astype('uint8')*255).filter(ImageFilter.MinFilter(5)))>0
    core=eroded if eroded.any() else opaque
    # Quantized CUB masks can have scattered opaque pixels or soft-only support.
    # Do not manufacture opacity or reject valid soft coverage via erosion.
    if not (a>0).any():raise ValueError('Empty mask in fixed model crop')
    if not outside.any():raise ValueError('No background survives fixed original crop')
    if not np.array_equal(f[opaque],g[opaque]):raise ValueError('Opaque foreground RGB changed')
    if not np.array_equal(f[core],g[core]):raise ValueError('Core foreground RGB changed')
    if not (g[outside]==np.asarray(gray_rgb)).all():raise ValueError('Background replacement not exact')
    expected=np.rint(f.astype(float)*a[...,None]/255+np.asarray(gray_rgb)*(1-a[...,None]/255))
    blend_error=int(abs(expected-g).max())
    if blend_error>1:raise ValueError('Unexpected soft-alpha blend')
    # Confirm alpha geometry matches the existing premultiplied RGBA resizing.
    native_rgba=rgb.convert('RGBA');native_rgba.putalpha(mask)
    prior_alpha=native_rgba.convert('RGBa').resize((256,256),Image.Resampling.BILINEAR).convert('RGBA').getchannel('A')
    if prior_alpha.tobytes()!=alpha256.tobytes():raise ValueError('Alpha resize differs from historical convention')
    support=a>0
    boundary=((support[1:,:]!=support[:-1,:]).sum()+(support[:,1:]!=support[:,:-1]).sum())
    background_luma=f.mean(2);dx=abs(np.diff(background_luma,axis=1));dy=abs(np.diff(background_luma,axis=0))
    complexity=float(np.r_[dx[outside[:,1:]&outside[:,:-1]],dy[outside[1:,:]&outside[:-1,:]]].mean()) if outside.sum()>10 else 0.
    audit=dict(canvas_width=256,canvas_height=256,model_width=224,model_height=224,
        mask_alpha_min=int(values.min()),mask_alpha_max=int(values.max()),
        mask_is_soft=bool(((values>0)&(values<255)).any()),
        mask_support_fraction=float(support.mean()),mask_boundary_ratio=float(boundary/max(1,support.sum())),
        background_texture_score=complexity,opaque_pixels=int(opaque.sum()),core_pixels=int(core.sum()),
        eroded_core_available=bool(eroded.any()),opaque_available=bool(opaque.any()),
        outside_pixels=int(outside.sum()),soft_boundary_pixels=int(soft.sum()),
        opaque_rgb_exact=True,core_rgb_exact=True,outside_gray_exact=True,spatial_alignment_exact=True,
        soft_boundary_mean_abs_change=float(abs(f[soft].astype(float)-g[soft]).mean()) if soft.any() else 0.,
        soft_blend_rounding_error_max=blend_error,alpha_matches_historical_resize=True,
        original_model_input_sha256=pixel_hash(full),gray_raw_input_sha256=pixel_hash(gray256),
        gray_model_input_sha256=pixel_hash(gray),model_mask_alpha_sha256=pixel_hash(alpha))
    return gray256,full,gray,alpha,audit


def contact_tiles(full,gray,alpha):
    a=np.asarray(alpha).astype(float)/255
    f=np.asarray(full).astype(float);overlay=f.copy()
    overlay[...,1]=overlay[...,1]*(1-.3*a)+255*.3*a
    diff=np.clip(abs(f-np.asarray(gray))*4,0,255).astype('uint8')
    return [full,gray,Image.fromarray(np.clip(overlay,0,255).astype('uint8')),Image.fromarray(diff)]
