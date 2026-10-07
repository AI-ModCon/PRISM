from pathlib import Path
import hashlib,json,tarfile
root=Path('/lus/flare/projects/ModCon/sandeep/prism-docci-alignment-20260922')
manifest=json.loads((root/'provenance/region-alignment-source-overlay.json').read_text())
archive=root/'region-alignment-source-overlay.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
dest=root/'prism-feature-alignment-regions'
assert dest.is_dir()
assert not (root/'jobs/feature-alignment-regions32-1000-01/submission.json').exists()
assert not (root/'runs/feature-alignment-regions32-1000-01').exists()
expected=manifest['sha256']; allowed=set(expected)
with tarfile.open(str(archive),'r:gz') as tf:
 members=tf.getmembers()
 assert len(members)==len(allowed) and {m.name for m in members}==allowed
 for member in members:
  relative=Path(member.name)
  assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
  assert member.size<4*1024*1024
  data=tf.extractfile(member).read()
  assert hashlib.sha256(data).hexdigest()==expected[member.name]
  target=dest/relative
  assert not target.is_symlink()
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(data)
for name,digest in expected.items(): assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest
print(json.dumps({'source':str(dest),'verified_files':len(expected),'job_submitted':False},indent=2))
