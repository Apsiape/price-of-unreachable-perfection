"""Clean standalone build with ordinary pdflatex and bibtex."""
from pathlib import Path
import argparse, subprocess, shutil, re, json, hashlib, os

p=argparse.ArgumentParser()
p.add_argument('destination',type=Path,help='New empty directory for a clean build')
p.add_argument('--deliver',action='store_true',help='Also copy the PDF beside this script')
args=p.parse_args()
src=Path(__file__).resolve().parent
dst=args.destination.resolve()
if dst.exists() and any(dst.iterdir()):
    raise SystemExit('Build destination must be new or empty')
dst.mkdir(parents=True,exist_ok=True)
for file in src.rglob('*'):
    if file.is_file() and file.suffix in ('.tex','.bib') and 'arxiv' not in file.relative_to(src).parts:
        target=dst/file.relative_to(src)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,target)
commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'],['bibtex','main']]
commands += [['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']]*3
records=[]
for i,cmd in enumerate(commands,1):
    result=subprocess.run(cmd,cwd=dst,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
    (dst/('pass-'+str(i)+'.txt')).write_text(result.stdout,encoding='utf-8')
    records.append({'command':cmd,'exit_code':result.returncode})
    if result.returncode:
        raise SystemExit('Build command failed; inspect pass log')
log=(dst/'main.log').read_text(encoding='utf-8',errors='replace')
blg=(dst/'main.blg').read_text(encoding='utf-8',errors='replace')
problems=re.findall(r'^.*(?:Warning|Overfull|Underfull|undefined|^!).*$',log,re.M)
problems+=re.findall(r'^(?:Warning--.*|.*[1-9][0-9]* error messages?.*)$',blg,re.M)
if problems:
    print('\n'.join(problems))
    raise SystemExit('Strict build did not pass')
pdf=dst/'main.pdf'
result={'commands':records,'warnings':0,'undefined_references':0,'overfull_boxes':0,'underfull_boxes':0,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':int(re.search(r'Output written on .*?\((\d+) pages',log,re.S)[1])}
(dst/'build-result.json').write_text(json.dumps(result,indent=2)+'\n')
if args.deliver:
    shutil.copyfile(pdf,src/'main.pdf')
    shutil.copyfile(dst/'main.bbl',src/'main.bbl')
print(json.dumps(result,indent=2))
