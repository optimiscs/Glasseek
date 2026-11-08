import json,unittest
from tool_xml import serialize,parse_body,normalize_text
from ingest import normalize
class XMLTests(unittest.TestCase):
 def test_types_and_delimiters_roundtrip(self):
  call={'name':'web_search','arguments':{'query':'2042\n & 中文','limit':5,'enabled':True,'optional':None,'arr':[1,'2'],'obj':{'x':'<tag>'}}}
  text=serialize(call);converted,calls=normalize_text(text,[call])
  self.assertEqual(calls,[call]);self.assertEqual(converted,text)
 def test_delimiter_collision_is_not_silently_escaped(self):
  with self.assertRaises(ValueError):serialize({'name':'web_search','arguments':{'query':'literal </parameter>'}})
 def test_native_and_json_body_converge(self):
  call={'name':'web_extract','arguments':{'urls':['https://example.org/?a=1&b=2']}}
  native=normalize_text('',[{'function':{'name':call['name'],'arguments':json.dumps(call['arguments'])}}])
  tagged=normalize_text('<tool_call>'+json.dumps(call)+'</tool_call>')
  self.assertEqual(native,tagged)
 def test_legacy_xml_and_multiple_calls(self):
  text='<tool_call><function=web_search><parameter=query>中文 query</parameter></function></tool_call>'
  output,calls=normalize_text(text+'\n'+text)
  self.assertEqual(len(calls),2);self.assertEqual(calls[0]['arguments']['query'],'中文 query')
  self.assertEqual(normalize_text(output)[0],output)
 def test_reasoning_examples_are_not_executable_calls(self):
  text='<think>Syntax example <tool_call></tool_call>.</think>\n<tool_call>{"name":"search","arguments":{"query":"x"}}</tool_call>'
  output,calls=normalize_text(text);self.assertEqual(len(calls),1);self.assertTrue(output.startswith('<think>Syntax example'))
 def test_invalid_conflicting_and_truncated_calls_fail_closed(self):
  for text in ['<tool_call>{}', '<tool_call></tool_call>', '<tool_call><function=x><parameter=a>1</parameter><parameter=a>2</parameter></function></tool_call>']:
   with self.assertRaises(ValueError):normalize_text(text)
  with self.assertRaises(ValueError):normalize_text('<tool_call>{"name":"x","arguments":{}}</tool_call>',[{'name':'y','arguments':{}}])
 def test_observations_unchanged(self):
  observation='<tool_response>quotes & <xml> stay byte exact</tool_response>'
  row={'question':'q','messages':[{'role':'assistant','content':'<tool_call>{"name":"search","arguments":{"query":"q"}}</tool_call>'},{'role':'tool','content':observation},{'role':'assistant','content':'done'}]}
  result=normalize(row,'test','fixture',0)
  self.assertEqual(result['steps'][0]['observation'],observation+'\n');self.assertIn('<function=search>',result['steps'][0]['assistant'])
if __name__=='__main__':unittest.main()
