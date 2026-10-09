import hashlib, io, json, sys, tarfile
from pathlib import Path
b = Path('/lus/flare/projects/ModCon/sandeep/prism-docci-qwen3-1p7b-20260921')
name = sys.argv[1]
assert name in {'smoke-01', 'pilot-500-01'}
run = b / 'runs' / name
report = json.loads((run / 'report.json').read_text())
assert report['status'] == 'completed' and report['frozen_state_unchanged'] is True
files = [str(p.relative_to(b)) for p in run.iterdir() if p.is_file() and p.suffix in {'.json', '.jsonl', '.png'}]
files += ['runs/numerical-01/manifest.json', 'provenance/smoke-acceptance-checks.json', 'provenance/pilot-acceptance-checks.json', 'jobs/smoke-01/submission.json', 'jobs/pilot-500-01/submission.json']
selected = {s['id'] for s in report['samples']}
records = []
for split in ['train', 'validation']:
 for line in (b / 'webdataset' / (split + '.jsonl')).read_text().splitlines():
  row = json.loads(line)
  if row['id'] in selected: records.append(row)
with tarfile.open(str(b / (name + '-evidence.tar.gz')), 'w:gz') as tar:
 for relative in files:
  if (b / relative).is_file(): tar.add(str(b / relative), arcname=relative, recursive=False)
 for row in records:
  with (b / 'webdataset' / row['shard']).open('rb') as h:
   h.seek(row['data_offset']); payload = h.read(row['size'])
  assert hashlib.sha256(payload).hexdigest() == row['image_sha256']
  info = tarfile.TarInfo('runs/' + name + '/targets/' + row['id'] + '.jpg'); info.size = len(payload)
  tar.addfile(info, io.BytesIO(payload))
 payload = (json.dumps({'source': 'Google DOCCI', 'license': 'CC-BY-4.0', 'source_url': 'https://google.github.io/docci/', 'records': records}, indent=2) + '\n').encode()
 info = tarfile.TarInfo('runs/' + name + '/gallery-targets.json'); info.size = len(payload)
 tar.addfile(info, io.BytesIO(payload))
print(json.dumps({'archive':str(b / (name + '-evidence.tar.gz')),'bytes':(b / (name + '-evidence.tar.gz')).stat().st_size,'samples':len(report['samples']),'target_images':len(records)}))
