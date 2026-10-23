"""Untrusted VLM selection -> measured identity; no motion/map authority."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def requested_interface(instruction):
    """Explicit limited task language, not a pretend general-language parser.

    These are the existing known hardware paint identities, NOT shape classes.
    The check only rejects mismatches; it never replaces a model selection.
    """
    match = re.fullmatch(r'Select the (red|green|blue) interface\.', instruction.strip())
    if match is None:
        raise ValueError('unsupported_task_language_use_Select_the_COLOR_interface')
    return {'red':'bolt', 'green':'washer', 'blue':'bracket'}[match[1]]


def validate_intent(instruction, binding):
    if requested_interface(instruction) != binding['measured_id']:
        raise ValueError('vlm_selection_disagrees_with_measured_requested_colour')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_request(obs, estimates, raw_path, instruction, output):
    """Crop *measured* interface projections, not simulator boxes or IDs."""
    from PIL import Image, ImageDraw
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rgb = np.asarray(obs.rgb)
    if rgb.dtype != np.uint8:
        rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb[:, :, :3])
    candidates = []
    for e in estimates.values():
        if e.kind != 'magnetic_interface' or abs(e.stamp - obs.stamp) > 1e-6:
            continue
        local = (np.asarray(e.position) - obs.camera_world[:3, 3]) @ obs.camera_world[:3, :3]
        if -local[2] <= 0:
            continue
        u = obs.K[0, 0] * local[0] / -local[2] + obs.K[0, 2]
        v = -obs.K[1, 1] * local[1] / -local[2] + obs.K[1, 2]
        if 0 <= u < image.width and 0 <= v < image.height:
            candidates.append((float(u), float(v), e))
    candidates.sort(key=lambda x: (x[0], x[1]))
    if not 1 <= len(candidates) <= 3:
        raise ValueError('need_one_to_three_current_measured_interfaces')
    gallery = Image.new('RGB', (224 * len(candidates), 256), 'white')
    draw = ImageDraw.Draw(gallery)
    bindings = {}
    for i, (u, v, e) in enumerate(candidates):
        label = chr(65 + i)
        # Nearby objects can occur in the same wide crop. Restrict each tile to
        # its interface neighbourhood using measured projection separation.
        separation = min((np.hypot(u-x, v-y) for j,(x,y,_) in enumerate(candidates) if j != i), default=60.)
        radius = max(4, min(24, int(.40*separation)))
        if separation < 10:
            raise ValueError('candidate_projections_too_close_for_unambiguous_tiles')
        box = (max(0, round(u)-radius), max(0, round(v)-radius), min(image.width, round(u)+radius), min(image.height, round(v)+radius))
        crop = image.crop(box).resize((224, 224), Image.Resampling.NEAREST)
        gallery.paste(crop, (224*i, 0))
        draw.text((224*i+100, 228), label, fill='black', font_size=24)
        bindings[label] = dict(measured_id=e.name, position_m=np.asarray(e.position).tolist(),
                               covariance_m2=np.asarray(e.covariance).tolist(), pixel_center=[u, v], crop_box=list(box))
    image.save(output / 'sensor_rgb.png')
    gallery.save(output / 'candidates.png')
    request = dict(schema='rgbd-vlm-selection-v1', instruction=instruction,
                   labels=list(bindings), image='candidates.png', image_sha256=digest(output/'candidates.png'),
                   raw_rgbd_sha256=digest(raw_path), raw_rgbd_file=str(Path(raw_path).resolve()),
                   scope='RGB crops at measured interface projections; labels are arbitrary, no truth geometry or semantic names in model prompt')
    (output/'request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    (output/'measured_bindings.json').write_text(json.dumps(bindings, indent=2), encoding='utf-8')
    return request, bindings


def parse_selection(text, labels):
    text = text.strip()
    if text == 'NONE':
        raise ValueError('vlm_abstained')
    if not re.fullmatch(r'[A-C]\.?', text) or text[0] not in labels:
        raise ValueError('invalid_vlm_selection_no_fallback')
    return text[0]


def validate_response(folder):
    folder = Path(folder)
    request = json.loads((folder/'request.json').read_text(encoding='utf-8'))
    response = json.loads((folder/'response.json').read_text(encoding='utf-8'))
    if response['request_sha256'] != digest(folder/'request.json'):
        raise ValueError('vlm_response_request_mismatch')
    if response['image_sha256'] != request['image_sha256'] or digest(folder/request['image']) != request['image_sha256']:
        raise ValueError('vlm_image_mismatch')
    if digest(request['raw_rgbd_file']) != request['raw_rgbd_sha256']:
        raise ValueError('vlm_raw_evidence_mismatch')
    return parse_selection(response['text'], request['labels'])


def validate_reobservation(binding, estimate, stamp, tolerance_m=.01):
    if estimate is None or estimate.kind != 'magnetic_interface' or abs(estimate.stamp-stamp) > 1e-6:
        raise ValueError('selected_interface_not_freshly_measured')
    if estimate.name != binding['measured_id']:
        raise ValueError('selected_identity_changed')
    if not np.isfinite(estimate.position).all() or np.linalg.norm(np.asarray(estimate.position)-binding['position_m']) > tolerance_m:
        raise ValueError('selected_interface_moved_replan_task')
