#!/usr/bin/env python3
"""Export a small, checksummed snapshot for merging existing MOF results locally.

No model weights, training datasets, environments or simulation restarts are
needed to inspect and combine already computed result tables and plots.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
MAX_BYTES=8*1024*1024
RESULT_SUFFIXES={'.json','.csv','.png','.svg','.md','.txt'}
CODE_SUFFIXES={'.py','.sh','.json','.md','.template','.toml','.yaml','.yml'}
EXCLUDED_PARTS={'__pycache__','.cache','deps','assets','datasets','checkpoints','prediction_cache',
                'configurations','models','restart','training','venv','.venv','node_modules'}
RAW_NAMES={'10bar_history.csv','umof_mh_adsorption_finetuned_run-12345_train.txt'}


def result_candidate(path):
    relative=path.relative_to(ROOT)
    if path.suffix.lower() not in RESULT_SUFFIXES:return False,'non-summary format'
    if any(p in EXCLUDED_PARTS or p.startswith('checkpoints_') or p.startswith('checkpoint_') for p in relative.parts):return False,'models/raw-data/cache directory'
    if any('technical_smoke' in p for p in relative.parts):return False,'technical smoke test'
    if path.name in RAW_NAMES or path.name.endswith('_debug.txt'):return False,'raw per-step history'
    return True,None


def git_output(*args):
    return subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True).stdout.strip()


def add_bytes(archive,name,data):
    info=tarfile.TarInfo('MOF_benchwork_results_snapshot/'+name)
    info.size=len(data);info.mode=0o644;info.mtime=int(datetime.now(timezone.utc).timestamp())
    archive.addfile(info,io.BytesIO(data))


def export(output,scope='all'):
    selected={};omitted=[]
    def choose(path,category):
        if path.is_symlink() or not path.is_file():return
        rel=path.relative_to(ROOT).as_posix()
        size=path.stat().st_size
        if size>MAX_BYTES:
            omitted.append({'path':rel,'bytes':size,'reason':'larger than 8 MiB; detailed configuration/raw export'})
            # Keep compact metadata/metrics from oversized potential comparison results.
            if path.name=='potential_comparison.json':
                obj=json.loads(path.read_text())
                compact={k:v for k,v in obj.items() if k not in {'configurations','configuration_manifest'}}
                compact['transfer_note']='Per-configuration arrays omitted; see original server file for full data.'
                selected[rel+'.summary.json']=('derived_summary',json.dumps(compact,indent=2).encode())
            return
        selected[rel]=(category,path)
    for path in sorted((ROOT/'outputs').rglob('*')):
        if not path.is_file():continue
        if scope=='umof' and not ('umof' in path.as_posix().lower()):continue
        keep,reason=result_candidate(path)
        if keep:choose(path,'result')
        elif path.suffix.lower() in RESULT_SUFFIXES:
            omitted.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'reason':reason})
    for folder in ['analysis','modules','pipeline','engines','converter','converters','resolvers','scripts','tests','input_json_files']:
        directory=ROOT/folder
        if directory.exists():
            for path in sorted(directory.rglob('*')):
                if path.suffix in CODE_SUFFIXES and '__pycache__' not in path.parts:choose(path,'code_or_config')
    for path in ROOT.iterdir():
        if path.is_file() and (path.suffix=='.md' or path.name=='benchmark.py' or (path.name.startswith('requirements') and path.suffix=='.txt')):
            choose(path,'project_documentation')
    for folder in ['data','guidelines_for_project_developement']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.suffix in {'.json','.csv','.md','.extxyz','.def','.docx','.pdf'}:choose(path,'reference_or_documentation')
    created=datetime.now(timezone.utc).isoformat()
    provenance={'created_at_utc':created,'source_host':socket.gethostname(),'source_project':str(ROOT),
                'git_commit':git_output('rev-parse','HEAD'),'git_status':git_output('status','--short'),
                'scope':scope,'purpose':'Merge/view already computed results; not a simulation/training restart bundle.',
                'file_size_limit_bytes':MAX_BYTES,'snapshot_note':'Files from running campaigns may be provisional. Preserve their status and convergence labels.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    temporary=output.with_suffix(output.suffix+'.partial')
    inventory=[];references=set();captured=[]
    try:
        with tarfile.open(temporary,'w:gz',compresslevel=6) as archive:
            for relative,(category,source) in sorted(selected.items()):
                if isinstance(source,bytes):
                    data=source;mtime=None
                else:
                    before=source.stat();data=source.read_bytes();after=source.stat()
                    if before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=after.st_size or len(data)!=after.st_size:
                        omitted.append({'path':relative,'bytes':after.st_size,'reason':'changed while exporting; rerun for newer snapshot'})
                        continue
                    mtime=datetime.fromtimestamp(after.st_mtime,timezone.utc).isoformat()
                    if source.suffix=='.json':
                        try:json.loads(data)
                        except (ValueError,UnicodeError):
                            omitted.append({'path':relative,'bytes':len(data),'reason':'incomplete/invalid JSON during snapshot'})
                            continue
                add_bytes(archive,relative,data)
                inventory.append({'path':relative,'category':category,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source_mtime_utc':mtime})
                captured.append(relative)
                if relative.endswith(('.json','.md','.csv')):
                    for match in re.findall(rb'(?:CRAFTED-2\.0\.0/ISOTHERM_FILES/[^\s"\\<>]+\.csv|isodb-library/[^\s"\\<>]+\.json)',data):
                        references.add(match.decode('utf-8',errors='strict'))
            for relative in sorted(references):
                path=ROOT/relative
                if relative in captured or not path.is_file() or not path.resolve().is_relative_to(ROOT):continue
                if path.stat().st_size>MAX_BYTES:continue
                data=path.read_bytes();add_bytes(archive,relative,data)
                inventory.append({'path':relative,'category':'referenced_source_data','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source_mtime_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()})
            provenance['file_count']=len(inventory);provenance['uncompressed_bytes']=sum(x['bytes'] for x in inventory)
            add_bytes(archive,'TRANSFER_MANIFEST.json',json.dumps({**provenance,'files':inventory,'omitted':omitted},indent=2).encode())
            checksums=''.join(f"{row['sha256']}  {row['path']}\n" for row in inventory)
            add_bytes(archive,'CHECKSUMS.sha256',checksums.encode())
            text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
            add_bytes(archive,'RESULT_FILES.csv',text.getvalue().encode())
            instructions='''# MOF result snapshot for local consolidation

The outputs/ tree preserves the server's project-relative paths. It contains
result JSON/CSV, plots, status/provenance and analysis metadata across the
selected modules. Code/config snapshots and curated reference metadata are
included. Referenced CRAFTED isotherms found in exported metadata are included.

Start with RESULT_FILES.csv to locate reports, final_assessment.json,
isotherm_summary.csv, reference_metrics.csv and per-module result summaries.
TRANSFER_MANIFEST.json records the source commit, timestamps, checksums and
files omitted. Existing report/convergence status is authoritative: transfer
does not turn a provisional result into a completed validation.

Extract into a NEW directory. Compare paths with your local project before
copying files over existing results. Do not overwrite same-named local runs
blindly. Paths embedded in some original JSON may still name the server; use
the preserved relative tree to resolve matching files. Server absolute paths
have not been rewritten inside scientific provenance.

On Linux/macOS, from this extracted directory:

    shasum -a 256 -c CHECKSUMS.sha256

Code is a snapshot for reference, not a request to replace your local checkout.
Use the recorded git commit to compare versions. Models, training datasets,
checkpoints, environments, large raw comparison arrays, binary MC histories,
full DFT inputs/restarts and technical smoke artifacts are excluded. Therefore
this package is for result consolidation, not rerunning all simulations or
training. Old paths pointing at excluded files are expected.

A CRAFTED table validates the declared comparison against that reference;
retain labels distinguishing reference sources, pressure ranges and sampling
limitations. DFT stop/provisional states are preserved as recorded.
'''
            add_bytes(archive,'TRANSFER_README.md',instructions.encode())
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    # Verify every archived payload against the exported inventory.
    expected={x['path']:x['sha256'] for x in inventory}
    with tarfile.open(output,'r:gz') as archive:
        for member in archive:
            relative=member.name.removeprefix('MOF_benchwork_results_snapshot/')
            if relative in expected:
                payload=archive.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest()!=expected.pop(relative):raise RuntimeError('Archive integrity failure')
    if expected:raise RuntimeError('Archive files missing')
    summary={**provenance,'archive':str(output),'archive_bytes':output.stat().st_size,'archive_sha256':hashlib.file_digest(output.open('rb'),'sha256').hexdigest(),'omitted_file_count':len(omitted),'integrity_verified':True}
    output.with_name(output.name+'.json').write_text(json.dumps(summary,indent=2)+'\n')
    output.with_name(output.name+'.sha256').write_text(summary['archive_sha256']+'  '+output.name+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--scope',choices=['all','umof'],default='all')
    args=parser.parse_args()
    output=args.output or ROOT/'exports'/('MOF_benchwork_results_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')+'.tar.gz')
    export(output.resolve(),args.scope)


if __name__=='__main__':main()
