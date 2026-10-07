"""Audit a completed dense DOCCI pilot and package small evidence artifacts."""
import hashlib, io, json, math, sys, tarfile
from pathlib import Path
b=Path('/lus/flare/projects/ModCon/sandeep/prism-docci-joint-20260921')
name=sys.argv[1]
assert '/' not in name and name.startswith(('smoke-', 'pilot-'))
r=b/'runs'/name
j=json.loads((r/'report.json').read_text())
s=[json.loads(line) for line in (r/'steps.jsonl').read_text().splitlines()]
indexes={split:Path(j['settings'][split+'_index']) for split in ('train','validation')}
records={split:[json.loads(line) for line in p.read_text().splitlines()] for split,p in indexes.items()}
ids={split:{row['id'] for row in rows} for split,rows in records.items()}
seen=[i for row in s for batch in row['microbatches'] for i in batch['ids']]
latents={}
for sample in j['samples']:latents.setdefault(sample['id'],set()).add(sample['initial_latent_sha256'])
steps=j['settings']['steps']
checks={
 'completed_real_run':j['status']=='completed' and j['evidence_kind']=='real_checkpoint_connector_diffusion_webdataset_pilot',
 'all_requested_updates':j['completed_steps']==steps and [x['step'] for x in s]==list(range(1,steps+1)),
 'training_ids_only':bool(seen) and set(seen)<=ids['train'] and not set(seen)&ids['validation'],
 'finite_losses_and_group_gradients':all(math.isfinite(x['loss']) and all(math.isfinite(x['optimizer_diagnostics']['groups'][g]['gradient_norm_before_clip']) and x['optimizer_diagnostics']['groups'][g]['nonzero_gradient_tensors']>0 for g in ('connector','diffusion')) for x in s),
 'complete_dense_scope':j['trainable_parameter_counts']=={'connector':4200448,'diffusion':3967161400},
 'both_groups_changed':j['trainable_groups_changed']=={'connector':True,'diffusion':True},
 'frozen_unchanged':j['frozen_state_unchanged'] is True and j['frozen_hashes_before']==j['frozen_hashes_after'],
 'sampler_progress':j['sampler_state']['examples_seen']==len(seen),
}
if j['samples']:
 checks.update(sample_noise_replayed=all(len(v)==1 for v in latents.values()), sampling_target_free=all(v['sampling_target_free'] for v in j['samples']),native_baseline_precedes_updates=all(x['step']==0 for x in j['samples'] if x['stage']=='native-pretrained'))
if not j['settings']['final_validation_count']:
 checks['full_validation']=j['evaluations'][-1]['full_validation'] and j['evaluations'][-1]['splits']['validation']['count']==len(ids['validation'])
print(json.dumps(checks,indent=2),flush=True)
assert all(checks.values())
checkpoint=j['checkpoints'][-1];p=Path(checkpoint['path']);h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==checkpoint['sha256'] and checkpoint['step']==steps
checks['final_checkpoint_verified']={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
checks['sampling_exercised']=bool(j['samples'])
checks['examples_consumed']=len(seen);checks['unique_training_examples']=len(set(seen))
(b/'provenance'/(name+'-acceptance-checks.json')).write_text(json.dumps(checks,indent=2)+'\n')
summary={k:j[k] for k in ('completed_steps','duration_seconds','peak_allocated_bytes','trainable_parameter_counts','trainable_groups_changed','frozen_state_unchanged')}
summary.update(optimizer_step_seconds=[x['duration_seconds'] for x in s],checkpoint=checks['final_checkpoint_verified'])
(r/'result-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
selected={x['id'] for x in j['samples']}
selected_records=[row for rows in records.values() for row in rows if row['id'] in selected]
files=[p for p in r.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl','.png'}]
files += [b/'runs/numerical-01/manifest.json', b/'provenance'/(name+'-acceptance-checks.json'), b/'jobs'/name/'submission.json',b/'jobs'/name/'job.pbs',b/'jobs'/name/'worker.sh',b/'provenance/prism-joint-training-overlay.json',b/'provenance/pilot-warm-start-replay.json',b/'provenance/pilot-pbs-final.txt',b/'provenance/prism-joint-pilot-command.json',b/'provenance/pilot-dry-run.json']
with tarfile.open(str(b/(name+'-evidence.tar.gz')),'w:gz') as tar:
 for p in files:
  if p.is_file():tar.add(str(p),arcname=str(p.relative_to(b)),recursive=False)
 for row in selected_records:
  with (indexes[row['split']].parent/row['shard']).open('rb') as stream:
   stream.seek(row['data_offset']);payload=stream.read(row['size'])
  assert hashlib.sha256(payload).hexdigest()==row['image_sha256']
  info=tarfile.TarInfo('runs/'+name+'/targets/'+row['id']+'.jpg');info.size=len(payload)
  tar.addfile(info,io.BytesIO(payload))
 payload=(json.dumps({'source':'Google DOCCI','license':'CC-BY-4.0','source_url':'https://google.github.io/docci/','records':selected_records},indent=2)+'\n').encode()
 info=tarfile.TarInfo('runs/'+name+'/gallery-targets.json');info.size=len(payload)
 tar.addfile(info,io.BytesIO(payload))
print(json.dumps(summary,indent=2),flush=True)
