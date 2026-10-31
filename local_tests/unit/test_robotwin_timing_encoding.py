"""Regression for a C locale after native setup, without loading a simulator."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('utf8_mode',['0','1'])
def test_c_locale_reads_python_and_json_without_changing_process_locale(tmp_path,utf8_mode):
    # Python source must respect PEP 263; JSON and evidence journals are UTF-8.
    source=tmp_path/'unicode_source.py';source.write_text('# \u5c06 NumPy \u6570\u7ec4\u8f6c\u6362\nvalue = 1\n',encoding='utf-8')
    latin=tmp_path/'latin_source.py';latin.write_bytes(b'# coding: latin-1\n# caf\xe9\nvalue = 1\n')
    program=r'''
import json,locale,pathlib,sys
from services.robotwin.control_timing_observer import source_scope
from services.robotwin.collection import atomic_json
from scripts.diagnose_robotwin_capture_timing import read
p=pathlib.Path(sys.argv[1]);before=locale.setlocale(locale.LC_CTYPE)
assert before=='C'
old_decode_failed=False
try:(p/'unicode_source.py').read_text()
except UnicodeDecodeError:old_decode_failed=True
assert old_decode_failed == (sys.flags.utf8_mode==0)
scope=source_scope([p/'unicode_source.py',p/'latin_source.py'])
assert scope['passed']
atomic_json(p/'evidence.json',{'unicode':'\u4e2d\u6587'})
assert read(p/'evidence.json')=={'unicode':'\u4e2d\u6587'}
assert (p/'evidence.json').read_bytes().decode('utf-8').find('\u4e2d\u6587')>=0
assert locale.setlocale(locale.LC_CTYPE)==before
print(json.dumps({'passed':True,'preferred_encoding':locale.getpreferredencoding(False),
                  'utf8_mode':sys.flags.utf8_mode,'old_decode_failed':old_decode_failed,'native_scene_created':False}))
'''
    # Keep argv ASCII; Python evaluates the Unicode literals inside the child.
    env=dict(os.environ,LC_ALL='C',LANG='C',PYTHONUTF8=utf8_mode,PYTHONCOERCECLOCALE='0',
             PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    result=subprocess.run([sys.executable,'-c',program,str(tmp_path)],env=env,cwd=ROOT,
                          capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['passed']
