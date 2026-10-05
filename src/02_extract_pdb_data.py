#!/usr/bin/env python
"""Extract protein chains, author-chain reference sequences, and PDB sequences from mmCIF files."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO, Select, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from tqdm import tqdm

DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_CIF_OUTDIR = "data/cifs"
DEFAULT_CHAIN_OUTDIR = "data/chains/pdb/"
DEFAULT_CHAIN_SEQS_OUTDIR = "data/chains/pdb_seqs"
DEFAULT_CHAIN_REFSEQS_OUTDIR = "data/chains/pdb_refseqs"

###################################### CONFIG
def get_ncores():
    for key in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value=os.environ.get(key)
        if value and value.isdigit(): return max(1,int(value))
    try: return max(1,len(os.sched_getaffinity(0)))
    except Exception: return max(1,os.cpu_count() or 1)

def resolve_config(path, defaults, no_json):
    if no_json: return dict(defaults)
    p=Path(path); cfg={}
    if p.exists():
        try: cfg=json.loads(p.read_text())
        except Exception: cfg={}
    changed=False
    for k,v in defaults.items():
        if k not in cfg: cfg[k]=v; changed=True
    if changed or not p.exists():
        p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(cfg,indent=2)+"\n")
    return {k:cfg[k] for k in defaults}

###################################### EXTRACTION
class ChainSelect(Select):
    def __init__(self, chain_id): self.chain_id=chain_id
    def accept_chain(self, chain): return chain.id==self.chain_id
    def accept_residue(self, residue): return residue.id[0]==" "

def clean_seq(s): return "".join(str(s).split()).replace("(","").replace(")","").upper()

def process_cif(task):
    cif_path,cfg=task; cif_path=Path(cif_path)
    try:
        structure=MMCIFParser(QUIET=True).get_structure(cif_path.stem,str(cif_path)); mm=MMCIF2Dict(str(cif_path))
    except Exception as e: return cif_path.name,0,0,0,str(e)
    dirs={k:Path(cfg[k]) for k in ("chain_outdir","chain_seqs_outdir","chain_refseqs_outdir")}
    for p in dirs.values(): p.mkdir(parents=True,exist_ok=True)
    eids=mm.get("_entity_poly.entity_id",[]); seqs=mm.get("_entity_poly.pdbx_seq_one_letter_code",[])
    if isinstance(eids,str): eids=[eids]
    if isinstance(seqs,str): seqs=[seqs]
    entity_seq={str(e):clean_seq(s) for e,s in zip(eids,seqs) if clean_seq(s)}
    asym=mm.get("_struct_asym.id",[]); ent=mm.get("_struct_asym.entity_id",[]); auth=mm.get("_atom_site.auth_asym_id",[]); label=mm.get("_atom_site.label_asym_id",[])
    if isinstance(asym,str): asym=[asym]
    if isinstance(ent,str): ent=[ent]
    if isinstance(auth,str): auth=[auth]
    if isinstance(label,str): label=[label]
    label_auth={}
    for l,a in zip(label,auth): label_auth.setdefault(str(l),str(a))
    ref={label_auth.get(str(a),str(a)):entity_seq.get(str(e),"") for a,e in zip(asym,ent)}
    npdb=nseq=nref=0
    for model in structure:
        for chain in model:
            cid=str(chain.id); stem=f"{cif_path.stem}_{cid}"
            pdb=dirs["chain_outdir"]/f"{stem}.pdb"; seqfile=dirs["chain_seqs_outdir"]/f"{stem}.fasta"; reffile=dirs["chain_refseqs_outdir"]/f"{stem}.fasta"
            PDBIO().set_structure(structure)
            io=PDBIO(); io.set_structure(structure); io.save(str(pdb),select=ChainSelect(cid))
            seq="".join(protein_letters_3to1_extended[r.get_resname().strip().upper()] for r in chain if r.id[0]==" " and r.get_resname().strip().upper() in protein_letters_3to1_extended)
            if not seq: continue
            seqfile.write_text(f">{stem}\n{seq}\n"); npdb+=1; nseq+=1
            r=ref.get(cid,"")
            if r: reffile.write_text(f">{stem}|author_chain_{cid}\n{r}\n"); nref+=1
        break
    return cif_path.name,npdb,nseq,nref,""

###################################### LOGGING
def finish_log(start,msg):
    elapsed=time.perf_counter()-start; usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; mem=usage/1024 if usage<10**7 else usage/1024**2; now=datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"); print(f"[PYTHON-INFO] {now} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem:.1f} MB")

###################################### MAIN EXECUTION
def main():
    start=time.perf_counter(); p=argparse.ArgumentParser(); p.add_argument("-i","--input", help="mmCIF file or directory (defaults to cif_outdir in config)"); p.add_argument("-o","--output", help="PDB chain output directory (defaults to chain_outdir in config)"); p.add_argument("-j","--json",nargs="?",const=DEFAULT_CONFIG_PATH,default=DEFAULT_CONFIG_PATH); p.add_argument("--no-json",action="store_true"); a=p.parse_args()
    defaults={"cif_outdir":DEFAULT_CIF_OUTDIR,"chain_outdir":DEFAULT_CHAIN_OUTDIR,"chain_seqs_outdir":DEFAULT_CHAIN_SEQS_OUTDIR,"chain_refseqs_outdir":DEFAULT_CHAIN_REFSEQS_OUTDIR}; cfg=resolve_config(a.json,defaults,a.no_json); src=Path(a.input or cfg["cif_outdir"]); files=sorted(src.glob("*.cif")) if src.is_dir() else [src]; n=get_ncores(); print(f"[PYTHON-INFO] Extracting {len(files)} CIF files with {n} worker(s)")
    totals=[0,0,0]
    with mp.Pool(n) as pool:
        jobs=[pool.apply_async(process_cif,((str(f),cfg),)) for f in files]
        for j in tqdm(jobs,total=len(jobs),desc="Extracting PDB data"):
            name,npdb,nseq,nref,err=j.get(); totals[0]+=npdb; totals[1]+=nseq; totals[2]+=nref
            if err: print(f"[PYTHON-INFO] {name}: {err}")
    finish_log(start,f"Saved PDB={totals[0]} sequences={totals[1]} RefSeqs={totals[2]} from {len(files)} CIF files using {n} workers")

if __name__=="__main__": main()

