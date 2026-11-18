import os, resource, torch
from wan.modules.t5 import T5EncoderModel

ASSETS = os.path.expanduser('~/ai/models/lingbot-shared-assets')
def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print('start', flush=True)
m = T5EncoderModel(
    text_len=512,
    dtype=torch.bfloat16,
    device=torch.device('cpu'),
    checkpoint_path=os.path.join(ASSETS, 'models_t5_umt5-xxl-enc-bf16.pth'),
    tokenizer_path=os.path.join(ASSETS, 'google/umt5-xxl'),
)
print(f'loaded  maxrss={rss():.0f} MB', flush=True)
ctx = m(['a serene lake'], torch.device('cpu'))
print(f'encoded {tuple(ctx[0].shape)}  maxrss={rss():.0f} MB', flush=True)
print('OK', flush=True)
