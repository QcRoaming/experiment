"""Interleave module runs on the same two CPU IDs and saved numerical-test inputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from itertools import permutations

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('benchmark',type=Path)
parser.add_argument('parameters',type=Path)
parser.add_argument('inputs',type=Path)
parser.add_argument('--prefix',action='append',required=True)
parser.add_argument('--module',action='append',required=True,help='label=vmfb')
args = parser.parse_args()
modules = dict(v.split('=',1) for v in args.module)
cpus = ','.join(map(str,sorted(os.sched_getaffinity(0))[::2][:2]))
writer = csv.writer(sys.stdout,lineterminator='\n')
writer.writerow(('shape','round','position','module','time_ms','calls','cpu_ids'))
for prefix in args.prefix:
    files = sorted(args.inputs.glob(prefix+'*.npy'), key=lambda f:int(re.search(r'(\d+)\.npy$',f.name)[1]))
    assert files
    orders = list(permutations(modules))
    if len(modules)==2:
        orders *= 3
    for k,order in enumerate(orders):
        for position,name in enumerate(order):
            cmd=['taskset','-c',cpus,args.benchmark,'--function=main_graph','--device=local-task',
                 '--task_topology_cpu_ids='+cpus,'--parameters=model='+str(args.parameters),'--module='+modules[name],
                 *['--input=@'+str(f) for f in files],'--benchmark_min_time=0.08s',
                 '--benchmark_min_warmup_time=0.04','--benchmark_format=json']
            measure, = json.loads(subprocess.check_output(cmd,text=True))['benchmarks']
            assert measure['time_unit']=='ms'
            writer.writerow((prefix,k+1,position+1,name,measure['real_time'],measure['iterations'],cpus))
