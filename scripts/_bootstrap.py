from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'runs/_cache/numba')
os.environ['MPLCONFIGDIR']=str(ROOT/'runs/_cache/matplotlib')
sys.path.insert(0,str(ROOT/'src'))
