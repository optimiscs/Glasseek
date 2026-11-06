import ast,concurrent.futures,json,os,tempfile,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'/root/autodl-tmp/Glasseek/Evaluation_backend')
from agent.glasseek_tool_budget import initialize,reserve,exhausted

def allocate(_):return reserve(10)
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'budget.json';os.environ['GLASSEEK_TOOL_CALL_BUDGET_FILE']=str(p);initialize(p,120)
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:assert sum(pool.map(allocate,range(30)))==120
 assert exhausted();assert initialize(p,120,0)['used']==120
 p=Path(d)/'partial.json';os.environ['GLASSEEK_TOOL_CALL_BUDGET_FILE']=str(p);initialize(p,120,118)
 tree=ast.parse(Path('Evaluation_backend/run_agent.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='AIAgent');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_execute_tool_calls');mod=ast.Module(body=[method],type_ignores=[]);ns={'_should_parallelize_tool_batch':lambda x:False};exec(compile(ast.fix_missing_locations(mod),'dispatch','exec'),ns)
 calls=[SimpleNamespace(id=str(i),function=SimpleNamespace(name='web_search')) for i in range(5)];message=SimpleNamespace(tool_calls=calls);history=[];executed=[]
 class Agent:
  def _execute_tool_calls_sequential(self,msg,messages,*args):executed.extend(msg.tool_calls)
 ns['_execute_tool_calls'](Agent(),message,history,'task',1)
 assert len(executed)==2 and len(history)==3 and len(message.tool_calls)==5
 assert all(json.loads(x['content'])['executed'] is False for x in history)
 assert reserve(1)==0
 del os.environ['GLASSEEK_TOOL_CALL_BUDGET_FILE'];assert reserve(5)==5
print('PASS: process-safe shared cap; persisted resume; partial batch 118+5 admits only 2; denied calls never dispatched; disabled mode unchanged')
