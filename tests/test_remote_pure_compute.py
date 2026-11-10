import json
import os
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation_backend'))
from tools import code_execution_tool as tool
from tools.remote_pure_compute import execute


class RemotePureComputeTests(unittest.TestCase):
    def test_untrusted_host_is_rejected_without_connecting(self):
        with patch('subprocess.Popen') as popen:
            result = json.loads(execute('print(1)', 'another-host'))
        self.assertEqual(result['status'], 'error')
        popen.assert_not_called()

    def test_ssh_failure_returns_error_without_local_execution(self):
        with patch('subprocess.Popen', side_effect=OSError('ssh unavailable')):
            result = json.loads(execute('print(1)', 'Lab3090'))
        self.assertEqual(result['status'], 'error')
        self.assertIn('ssh unavailable', result['error'])

    def test_none_policy_routes_through_guard_to_isolated_backend(self):
        with patch.dict(os.environ, {'HERMES_REMOTE_PURE_COMPUTE': 'Lab3090'}), \
             patch('tools.terminal_tool._get_env_config', return_value={'env_type': 'docker'}), \
             patch.object(tool, '_get_filesystem_access', return_value='none'), \
             patch.object(tool, '_get_network_access', return_value='none'), \
             patch.object(tool, '_get_tool_access', return_value='none'), \
             patch.object(tool, '_get_execution_mode', return_value='strict'), \
             patch.object(tool, '_load_config', return_value={'timeout': 10}), \
             patch('tools.approval.check_execute_code_guard', return_value={'approved': True}) as guard, \
             patch('tools.remote_pure_compute.execute', return_value='{"status":"success"}') as remote, \
             patch.object(tool, '_execute_remote') as unrestricted:
            result = json.loads(tool.execute_code('print(17*19)'))
        self.assertEqual(result['status'], 'success')
        guard.assert_called_once()
        remote.assert_called_once_with('print(17*19)', 'Lab3090', 10)
        unrestricted.assert_not_called()

    def test_remote_error_does_not_fall_back_to_unrestricted_backend(self):
        with patch.dict(os.environ, {'HERMES_REMOTE_PURE_COMPUTE': 'Lab3090'}), \
             patch('tools.terminal_tool._get_env_config', return_value={'env_type': 'docker'}), \
             patch.object(tool, '_get_filesystem_access', return_value='none'), \
             patch.object(tool, '_get_network_access', return_value='none'), \
             patch.object(tool, '_get_tool_access', return_value='none'), \
             patch.object(tool, '_get_execution_mode', return_value='strict'), \
             patch.object(tool, '_load_config', return_value={'timeout': 10}), \
             patch('tools.approval.check_execute_code_guard', return_value={'approved': True}), \
             patch('tools.remote_pure_compute.execute', return_value='{"status":"error","error":"offline"}'), \
             patch.object(tool, '_execute_remote') as unrestricted:
            result = json.loads(tool.execute_code('print(1)'))
        self.assertEqual(result['status'], 'error')
        unrestricted.assert_not_called()

    def test_other_policy_is_rejected_before_execution(self):
        with patch.dict(os.environ, {'HERMES_REMOTE_PURE_COMPUTE': 'Lab3090'}), \
             patch('tools.terminal_tool._get_env_config', return_value={'env_type': 'docker'}), \
             patch.object(tool, '_get_filesystem_access', return_value='full'), \
             patch.object(tool, '_get_network_access', return_value='none'), \
             patch.object(tool, '_get_tool_access', return_value='none'), \
             patch.object(tool, '_get_execution_mode', return_value='strict'), \
             patch('tools.remote_pure_compute.execute') as remote:
            result = json.loads(tool.execute_code('print(1)'))
        self.assertEqual(result['status'], 'error')
        remote.assert_not_called()


@unittest.skipUnless(os.environ.get('GLASSEEK_TEST_REMOTE_SANDBOX') == '1', 'remote sandbox tests are opt-in')
class RemotePureComputeIntegrationTests(unittest.TestCase):
    def test_compute_file_and_network_isolation(self):
        created = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'Lab3090',
            'd=$(mktemp -d /tmp/glasseek-isolation-test.XXXXXXXX) && printf sentinel > "$d/secret" && printf "%s" "$d"'],
            capture_output=True, text=True, timeout=15, check=True)
        directory = created.stdout.strip()
        self.assertRegex(directory, r'^/tmp/glasseek-isolation-test\.[A-Za-z0-9]+$')
        try:
            code = f'''from pathlib import Path
print("CALC", 17*19)
try:
 print("HOST_READ", Path({(directory + '/secret')!r}).read_text())
except OSError: print("HOST_BLOCKED")
try:
 Path("/tmp/test-write").write_text("x")
 print("WRITE_ALLOWED")
except OSError: print("WRITE_BLOCKED")
try:
 Path("/usr/test-write").write_text("x")
 print("ROOT_WRITE_ALLOWED")
except OSError: print("ROOT_WRITE_BLOCKED")
import socket
s=socket.socket();s.settimeout(2)
try:
 s.connect(("1.1.1.1",443));print("NETWORK_ALLOWED")
except OSError: print("NETWORK_BLOCKED")
try:
 socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM);print("VSOCK_ALLOWED")
except OSError: print("VSOCK_BLOCKED")
'''
            result = json.loads(execute(code, 'Lab3090', 15))
            self.assertEqual(result['status'], 'success', result)
            self.assertEqual(result['output'].splitlines(), ['CALC 323', 'HOST_BLOCKED', 'WRITE_ALLOWED', 'ROOT_WRITE_BLOCKED', 'NETWORK_BLOCKED', 'VSOCK_BLOCKED'])
            self.assertTrue(result['cleanup_verified'])
        finally:
            subprocess.run(['ssh', '-o', 'BatchMode=yes', 'Lab3090', 'rm -rf -- ' + directory], timeout=15, check=True)

    def test_timeout_parallel_and_process_isolation(self):
        timeout = json.loads(execute('import time; time.sleep(10)', 'Lab3090', 1))
        self.assertEqual(timeout['status'], 'timeout', timeout)
        code = '''import threading
event=threading.Event()
threads=[]
try:
 for i in range(100):
  t=threading.Thread(target=event.wait,daemon=True);t.start();threads.append(t)
except RuntimeError: pass
assert 2 <= len(threads) < 64, len(threads)
event.set()
for t in threads: t.join()
print("THREADS_BOUNDED")
'''
        self.assertEqual(json.loads(execute(code, 'Lab3090', 15))['output'].strip(), 'THREADS_BOUNDED')
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda n: json.loads(execute(f'print({n}*7)', 'Lab3090', 15)), [2, 3]))
        self.assertEqual([(x['status'], x['output'].strip()) for x in results], [('success', '14'), ('success', '21')])

    def test_scientific_stack_and_actual_cgroup_limits(self):
        code = '''import json, pathlib, io, concurrent.futures
import numpy as np, pandas as pd, scipy, sympy, matplotlib
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import matplotlib.pyplot as plt
assert np.allclose(np.linalg.solve([[2.,0.],[0.,4.]],[4.,8.]), [2,2])
assert abs(quad(lambda x:x*x,0,1)[0]-1/3)<1e-8
assert abs(minimize_scalar(lambda x:(x-3)**2).x-3)<1e-6
assert pd.read_csv(io.StringIO("a,b\\n1,2\\n"))["b"].sum()==2
x=sympy.Symbol('x');assert sympy.expand((x+1)**2)==x*x+2*x+1
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 assert list(pool.map(lambda n: float(np.ones((8,8)).sum())+n,[1,2]))==[65.,66.]
plt.plot([1,2]);plt.savefig('/tmp/plot.png');assert pathlib.Path('/tmp/plot.png').stat().st_size>0
root=pathlib.Path('/sys/fs/cgroup')
def read(*names):
 for n in names:
  if (root/n).exists():return (root/n).read_text().strip()
 raise AssertionError(names)
assert read('pids.max','pids/pids.max')=='64'
assert read('memory.max','memory/memory.limit_in_bytes')=='2147483648'
if (root/'cpu.max').exists():
 q,p=map(int,read('cpu.max').split());assert q/p==2
else:
 assert int(read('cpu/cpu.cfs_quota_us','cpu,cpuacct/cpu.cfs_quota_us'))/int(read('cpu/cpu.cfs_period_us','cpu,cpuacct/cpu.cfs_period_us'))==2
print('SCIENCE_AND_LIMITS_OK')
'''
        result=json.loads(execute(code,'Lab3090',30))
        self.assertEqual(result['status'],'success',result)
        self.assertIn('SCIENCE_AND_LIMITS_OK',result['output'])
        self.assertTrue(result['cleanup_verified'])

    def test_output_memory_limits_and_fresh_tmp(self):
        output=json.loads(execute('while True: print("x"*8192)','Lab3090',15))
        self.assertEqual(output['status'],'output_limit',output)
        self.assertTrue(output['cleanup_verified'])
        memory=json.loads(execute('x=bytearray(3*1024**3)','Lab3090',30))
        self.assertEqual(memory['status'],'memory_limit',memory)
        self.assertTrue(memory['cleanup_verified'])
        first=json.loads(execute('open("/tmp/glasseek-test-marker","w").write("private")','Lab3090',15))
        self.assertEqual(first['status'],'success',first)
        second=json.loads(execute('import os; print(os.path.exists("/tmp/glasseek-test-marker"))','Lab3090',15))
        self.assertEqual(second['output'].strip(),'False',second)

    def test_private_tmp_quota_is_enforced(self):
        code = '''import errno
chunk=b'x'*(1024*1024)
try:
 for i in range(200):
  with open('/tmp/quota-'+str(i),'wb') as f: f.write(chunk)
except OSError as exc:
 assert exc.errno==errno.ENOSPC, exc
 print('TMP_QUOTA_ENFORCED')
else: raise AssertionError('tmp quota not enforced')
'''
        result=json.loads(execute(code,'Lab3090',20))
        self.assertEqual(result['status'],'success',result)
        self.assertEqual(result['output'].strip(),'TMP_QUOTA_ENFORCED')
        self.assertTrue(result['cleanup_verified'])


if __name__ == '__main__':
    unittest.main()
