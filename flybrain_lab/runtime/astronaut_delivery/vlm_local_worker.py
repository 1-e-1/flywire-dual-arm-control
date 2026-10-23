"""Offline image/text inference in a disposable process; never imports robot code."""
import argparse
import json
import time
from pathlib import Path

from .vlm_task_contract import digest


def run(folder, model_path, device='cuda'):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(4)
    folder = Path(folder)
    req = json.loads((folder/'request.json').read_text(encoding='utf-8'))
    if digest(folder/req['image']) != req['image_sha256']:
        raise ValueError('image_changed')
    model_path = str(Path(model_path).resolve())
    prompt = (req['instruction']+'\nWhich image tile matches: '+', '.join(req['labels'])+'? '
              'Reply with only the letter, or NONE if no tile matches.')
    (folder/'prompt.txt').write_text(prompt, encoding='utf-8')
    started = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
                attn_implementation='sdpa').to(device).eval()
    loaded = time.perf_counter()
    messages = [dict(role='user', content=[dict(type='image'), dict(type='text', text=prompt)])]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[Image.open(folder/req['image']).convert('RGB')], return_tensors='pt')
    inputs = inputs.to(device=device, dtype=model.dtype)
    if device == 'cuda': torch.cuda.synchronize()
    inference = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    if device == 'cuda': torch.cuda.synchronize()
    elapsed = time.perf_counter()-inference
    answer = processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    result = dict(model='HuggingFaceTB/SmolVLM-500M-Instruct', model_path=model_path,
                  model_config_sha256=digest(Path(model_path)/'config.json'), device=device,
                  model_weights_sha256={p.name:digest(p) for p in sorted(Path(model_path).glob('*.safetensors'))},
                  transformers=transformers.__version__, torch=torch.__version__, text=answer,
                  request_sha256=digest(folder/'request.json'), image_sha256=req['image_sha256'],
                  load_s=loaded-started, inference_s=elapsed, worker_inference_total_s=time.perf_counter()-started,
                  generated_tokens=int(ids.shape[1]-inputs['input_ids'].shape[1]),
                  gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
                  cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
                  cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,
                  memory_scope='This inference process torch allocator only; excludes simulation, driver and display')
    (folder/'response.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--request-dir', required=True);p.add_argument('--model', required=True)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda');a=p.parse_args()
    run(a.request_dir, a.model, a.device)
