"""Render a recorded DOCCI pilot comparison without invoking a model."""
import hashlib
import json
import sys
import os
import textwrap
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/prism-docci-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/prism-docci-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

root = Path(sys.argv[1]).resolve()
report = json.loads((root / 'report.json').read_text())
assert report['status'] == 'completed' and report['frozen_state_unchanged'] is True
records = {r['id']:r for r in json.loads((root / 'gallery-targets.json').read_text())['records']}
steps = report['settings']['sampling_steps']
last = report['completed_steps']
outputs = {}
for split in ('train','validation'):
 ids = [s['id'] for s in report['samples'] if s['stage']=='untrained' and s['split']==split]
 if not ids:continue
 stages = [('target','DOCCI target (loss only)'),('untrained','Before connector training'),('trained','Connector at step '+str(last))]
 if all(any(s['id']==i and s['stage']=='native' for s in report['samples']) for i in ids):
  stages.append(('native','Native OmniGen2'))
 height = 4.3*len(ids)+1.6
 fig = plt.figure(figsize=(3.3*len(stages),height),facecolor='white')
 grid = fig.add_gridspec(len(ids)*2,len(stages),height_ratios=[1,.20]*len(ids),left=.035,right=.98,top=1-1.2/height,bottom=.72/height,hspace=.27,wspace=.08)
 title = 'Training examples' if split=='train' else 'Validation examples'
 fig.suptitle('DOCCI / PRISM Qwen3-1.7B — '+title,x=.035,y=1-.22/height,ha='left',fontsize=17,fontweight='bold')
 fig.text(.035,1-.73/height,str(steps)+' diffusion steps · fixed starting noise per case · only the connector is trained',fontsize=10)
 for row,i in enumerate(ids):
  for col,(stage,label) in enumerate(stages):
   ax = fig.add_subplot(grid[row*2,col]);ax.axis('off')
   if stage=='target':
    p=root/'targets'/(i+'.jpg'); expected=records[i]['image_sha256']
   else:
    sample=next(s for s in report['samples'] if s['id']==i and s['stage']==stage and (stage!='trained' or s['step']==last))
    p=root/Path(sample['path']).name;expected=sample['sha256']
    assert sample['sampling_target_free']
   assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
   with Image.open(p) as im:
    pixels=ImageOps.exif_transpose(im).convert('RGB').resize((report['settings']['width'],report['settings']['height']),Image.Resampling.BICUBIC)
   ax.imshow(pixels);ax.set_title(label,fontsize=11,pad=8)
  textax=fig.add_subplot(grid[row*2+1,:]);textax.axis('off')
  caption=records[i]['prompt'];excerpt=caption[:280].rsplit(' ',1)[0]+'…' if len(caption)>280 else caption
  prefix=i+(' (optimized training example)' if split=='train' else ' (held out from connector optimization)')
  textax.text(0,1,prefix,va='top',fontsize=11,fontweight='bold',transform=textax.transAxes)
  textax.text(0,.57,textwrap.fill('Caption excerpt: '+excerpt,width=150 if len(stages)==4 else 110),va='top',fontsize=9,transform=textax.transAxes)
 fig.text(.035,.20/height,'Targets: Google DOCCI (CC BY 4.0). Generated images use captions alone. Qualitative diagnostics; no benchmark claim.',fontsize=9,color='#444444')
 path=root/(split+'-comparison.png');fig.savefig(path,dpi=150);plt.close(fig);outputs[split]=str(path)
print(json.dumps(outputs,indent=2))
