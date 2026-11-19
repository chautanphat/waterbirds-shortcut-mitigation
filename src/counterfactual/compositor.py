"""Mask-based pixel operations. No generative editing or whole-photo blending."""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.nn import functional as F

from .common import pixel_hash


def aligned(image, size, fit='cover'):
    """Mixer alignment semantics, applied only to the background."""
    if fit == 'stretch':
        return image.convert('RGB').resize(size, Image.Resampling.BILINEAR)
    if fit == 'cover':
        return ImageOps.fit(image.convert('RGB'), size, method=Image.Resampling.BILINEAR, centering=(.5, .5))
    if fit == 'contain':
        return ImageOps.pad(image.convert('RGB'), size, method=Image.Resampling.BILINEAR,
                            color=(127, 127, 127), centering=(.5, .5))
    raise ValueError('Unknown Mixer background alignment')


def _gaussian(array, sigma):
    if sigma == 0:
        return array.copy()
    radius = int(np.ceil(3*sigma))
    coords = torch.arange(-radius, radius+1, dtype=torch.float32)
    kernel = torch.exp(-.5*(coords/sigma)**2)
    kernel /= kernel.sum()
    x = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    channels = x.shape[1]
    # Separable normalized convolution; fixed zero padding outside layer support.
    x = F.conv2d(x, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
                 padding=(0, radius), groups=channels)
    x = F.conv2d(x, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
                 padding=(radius, 0), groups=channels)
    return x[0].numpy().transpose(1, 2, 0)


def foreground_layer(rgba, sigma, opacity=1., *, canvas_size=(256, 256),
                     operator='gaussian', strength=None, seed=0):
    """Fixed square geometry matches Runner; sigma is measured in its crop pixels.

    Crop has no rescale after the 256-square canvas, so sigma=1.5 here is 1.5
    pixels in the exact 224 model view. Original alpha support is never blurred.
    """
    if not np.isfinite(sigma) or not 0 <= sigma <= 30:
        raise ValueError('Sigma must be finite and between 0 and 30')
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('Opacity must be between 0 and 1')
    if operator not in ('gaussian','contrast','occlusion'):
        raise ValueError('Choose one foreground operator per experiment')
    if operator!='gaussian' and (sigma!=0 or strength is None or not np.isfinite(strength) or not 0<=strength<=1):
        raise ValueError('Contrast/occlusion need sigma=0 and a strength between 0 and 1; operators cannot be mixed')
    if rgba.mode != 'RGBA':
        raise ValueError('A real RGBA cutout or explicit validated mask is required')
    alpha_min,alpha_max=rgba.getchannel('A').getextrema()
    if alpha_max==0 or alpha_min>0:
        raise ValueError('Mask is empty or covers the entire photo; provide a bird segmentation')
    # Premultiplied resampling prevents original-background or black edge halos.
    layer = rgba.convert('RGBa').resize(canvas_size, Image.Resampling.BILINEAR).convert('RGBA')
    a = np.asarray(layer.getchannel('A'), dtype=np.float32)[..., None]/255
    rgb = np.asarray(layer.convert('RGB'), dtype=np.float32)
    if operator=='gaussian' and sigma:
        numerator = _gaussian(rgb*a, float(sigma))
        denominator = _gaussian(a, float(sigma))
        rgb = numerator/np.maximum(denominator, 1e-8)
        rgb[a[..., 0] == 0] = 0
    elif operator=='contrast':
        # Weighted bird-color mean; foreground/background never mix in this operator.
        mean=(rgb*a).sum(axis=(0,1))/max(float(a.sum()),1e-8)
        rgb=mean+float(strength)*(rgb-mean)
        rgb[a[...,0]==0]=0
    elif operator=='occlusion' and strength:
        coords=np.argwhere(a[...,0]>0)
        # One source-seeded center and a nested radial support set: increasing
        # severity covers more of the same region, without moving the bird/mask.
        rng=np.random.default_rng(seed)
        center=coords[int(rng.integers(len(coords)))]
        distances=((coords-center)**2).sum(axis=1)
        order=np.lexsort((coords[:,1],coords[:,0],distances))
        selected=coords[order[:int(round(len(coords)*float(strength)))]]
        rgb[selected[:,0],selected[:,1]]=[127,127,127]
    out = np.empty((*a.shape[:2], 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype('uint8')
    out[..., 3] = np.rint(a[..., 0]*255*opacity).astype('uint8')
    return Image.fromarray(out, 'RGBA')


def composite(layer, background, *, fit='cover'):
    bg = aligned(background, layer.size, fit)
    image = Image.alpha_composite(bg.convert('RGBA'), layer).convert('RGB')
    a = np.asarray(layer.getchannel('A'))
    im, b, f = np.asarray(image), np.asarray(bg), np.asarray(layer)[..., :3]
    audit = dict(foreground_layer_sha256=pixel_hash(layer), foreground_dimensions=list(layer.size),
        aligned_background_sha256=pixel_hash(bg), opaque_foreground_pixels=int((a == 255).sum()),
        opaque_foreground_exact=bool(np.array_equal(im[a == 255], f[a == 255])),
        outside_support_exact=bool(np.array_equal(im[a == 0], b[a == 0])), alpha_sha256=pixel_hash(layer.getchannel('A')))
    if not audit['opaque_foreground_exact'] or not audit['outside_support_exact']:
        raise ValueError('Pixel invariance audit failed')
    return image, audit


def mask_overlay(rgba):
    rgb = rgba.convert('RGB')
    overlay = Image.new('RGBA', rgba.size, (0, 200, 140, 0))
    overlay.putalpha(rgba.getchannel('A').point(lambda v: round(v*.35)))
    return Image.alpha_composite(rgb.convert('RGBA'), overlay).convert('RGB')


def difference_heatmap(a, b):
    """Absolute RGB pixel difference, explicitly not model saliency."""
    if a.size != b.size:
        raise ValueError('Difference inputs must share geometry')
    d = np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)).mean(2)
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    out[..., 0] = np.clip(d*3, 0, 255).astype('uint8')
    out[..., 1] = np.clip(d, 0, 255).astype('uint8')
    return Image.fromarray(out)
