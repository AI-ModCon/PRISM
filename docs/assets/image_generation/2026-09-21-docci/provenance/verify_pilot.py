import hashlib,json,math
from pathlib import Path
b=Path('/lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921')
r=b/'runs/pilot-500-01'
j=json.loads((r/'report.json').read_text())
s=[json.loads(line) for line in (r/'steps.jsonl').read_text().splitlines()]
train={json.loads(line)['id'] for line in (b/'webdataset/train.jsonl').read_text().splitlines()}
val={json.loads(line)['id'] for line in (b/'webdataset/validation.jsonl').read_text().splitlines()}
seen=[i for row in s for batch in row['microbatches'] for i in batch['ids']]
latents={}
for sample in j['samples']:latents.setdefault(sample['id'],set()).add(sample['initial_latent_sha256'])
checks={
 'completed_real_run':j['status']=='completed' and j['evidence_kind']=='real_checkpoint_connector_webdataset_pilot',
 '500_updates':j['completed_steps']==500 and [x['step'] for x in s]==list(range(1,501)),
 '500_distinct_train_examples':len(seen)==500 and len(set(seen))==500 and set(seen)<=train and not set(seen)&val,
 'finite_losses_gradients':all(math.isfinite(x['loss']) and all(math.isfinite(v) for v in x['gradient_norms'].values()) and any(v>0 for v in x['gradient_norms'].values()) for x in s),
 'connector_only':j['trainable_parameter_count']==4200448 and len(j['trainable_parameters'])==4 and all(n.startswith('decoders.image.connector.') for n in j['trainable_parameters']),
 'all_connector_tensors_received_gradients':set(j['trainable_parameters'])==set(j['nonzero_gradient_parameters']),
 'frozen_unchanged':j['frozen_state_unchanged'] is True and j['frozen_hashes_before']==j['frozen_hashes_after'],
 'full_validation':j['evaluations'][-1]['full_validation'] and j['evaluations'][-1]['splits']['validation']['count']==100,
 'sample_noise_replayed':all(len(v)==1 for v in latents.values()),
 'sampling_target_free':len(j['samples'])==10 and all(v['sampling_target_free'] for v in j['samples']),
 'sampler_progress':j['sampler_state']['examples_seen']==500 and j['sampler_state']['epoch']==0,
}
print(json.dumps(checks,indent=2),flush=True)
assert all(checks.values())
checkpoint=j['checkpoints'][-1];p=Path(checkpoint['path']);h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==checkpoint['sha256'] and checkpoint['step']==500
checks['final_checkpoint_verified']={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
(b/'provenance/pilot-acceptance-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'duration_seconds':j['duration_seconds'],'peak_allocated_bytes':j.get('peak_allocated_bytes'),'final_checkpoint':checks['final_checkpoint_verified'],'examples_seen':len(seen)},indent=2))
