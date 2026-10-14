"""Rebased DeST stage1 entry; no training occurs for --help or --selfcheck."""
import _bootstrap
from _bootstrap import ROOT
import sys,runpy
from topic3.paths import new_run
if '--help' not in sys.argv and '-h' not in sys.argv and '--selfcheck' not in sys.argv:
    args=sys.argv[1:]
    name=next((x.split('=',1)[1] for x in args if x.startswith('--run=')),None)
    if '--run' in args:
        index=args.index('--run');name=args[index+1] if index+1<len(args) else None
    if not name:raise SystemExit('Training requires --run NEW_NAME; historical output cannot be overwritten.')
    new_run(name)
    timebase=next((x.split('=',1)[1] for x in args if x.startswith('--timebase=')),'annotation15')
    if '--timebase' in args:timebase=args[args.index('--timebase')+1]
    if timebase!='annotation15':raise SystemExit('Historical encoded clock is archived; current entry requires annotation15.')
runpy.run_module('topic3.models.dest_stage1',run_name='__main__')
