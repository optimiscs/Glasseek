import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Shared'))
from glasseek_shared.hermes.url_registry import URLRegistry, convert_messages, annotate_schemas, rewrite_assistant, runtime_registry, persist

class URLTests(unittest.TestCase):
    def search(self,url='https://example.org/a?token=abc&x=2'):
        return json.dumps({'data':{'web':[{'url':url,'title':'A','description':'Evidence'}]}})

    def test_raw_identity(self):
        r=URLRegistry('raw');args={'urls':['https://example.org/']};text=self.search()
        self.assertIs(r.resolve_args('web_extract',args),args)
        self.assertEqual(r.observe('web_search',text),text)
        schemas=[{'function':{'name':'web_extract'}}];self.assertIs(annotate_schemas(schemas,'raw'),schemas)
        messages=[{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]
        self.assertEqual(convert_messages(messages,'raw')[0],messages)

    def test_future_and_same_batch_refs_fail(self):
        for mode in ('hybrid','ref','evidence'):
            r=URLRegistry(mode);r.begin_turn();r.observe('web_search',self.search())
            with self.assertRaises(ValueError):r.resolve('U1')
            r.begin_turn();self.assertEqual(r.resolve('U1'),'https://example.org/a?token=abc&x=2')
            with self.assertRaises(ValueError):r.resolve('U2')

    def test_xml_references_are_exact_causal_addresses(self):
        for mode in ('hybrid','ref','evidence'):
            r=URLRegistry(mode);args={'urls':['<url> U1 </url>']}
            with self.assertRaises(ValueError):r.resolve_args('web_extract',args)
            output=r.observe('web_search',self.search())
            self.assertIn('<url>U1</url>',output)
            with self.assertRaises(ValueError):r.resolve_args('web_extract',args)
            r.begin_turn()
            self.assertEqual(r.resolve_args('web_extract',args)['urls'],['https://example.org/a?token=abc&x=2'])
            self.assertEqual(args,{'urls':['<url> U1 </url>']})
            for invalid in ('https://example.org/?v=<url> U1 </url>','<url> U1 U2 </url>','<url> U1','<url> U999 </url>'):
                with self.assertRaises(ValueError):r.resolve(invalid)
            self.assertEqual(r.expand_references('Source <url> U1 </url>'), 'Source https://example.org/a?token=abc&x=2')
            self.assertEqual(r.expand_references('<url> U999 </url>'),'<url> U999 </url>')

    def test_exact_signed_and_punctuation_urls(self):
        for url in ('https://example.org/A_(B)?x=%2F&sig=A%2BB','https://example.org/a?','https://example.org/a.'):
            r=URLRegistry('ref');r.observe('web_search',self.search(url));r.begin_turn()
            self.assertEqual(r.resolve('U1'),url)

    def test_sft_causal_and_nonmutating(self):
        url='https://example.org/future'
        messages=[{'role':'assistant','content':url,'tool_calls':[{'id':'a','function':{'name':'web_extract','arguments':json.dumps({'urls':[url]})}}]},
                  {'role':'tool','tool_call_id':'a','content':json.dumps({'results':[{'url':url,'content':'Evidence'}]})},
                  {'role':'assistant','content':url,'tool_calls':[{'id':'b','function':{'name':'web_extract','arguments':json.dumps({'urls':[url]})}}]}]
        original=copy.deepcopy(messages);converted,state=convert_messages(messages,'ref')
        self.assertEqual(messages,original);self.assertEqual(converted[0]['content'],url)
        self.assertEqual(json.loads(converted[0]['tool_calls'][0]['function']['arguments'])['urls'],[url])
        self.assertEqual(converted[2]['content'],'<url>U1</url>')
        self.assertEqual(json.loads(converted[2]['tool_calls'][0]['function']['arguments'])['urls'],['U1'])

    def test_hybrid_first_occurrence_and_query_code_protected(self):
        r=URLRegistry('hybrid');url='https://example.org/a'
        self.assertIn(url,r.observe('web_search',self.search(url)));r.begin_turn()
        self.assertNotIn(url,r.observe('web_search',self.search(url)))
        text='<tool_call><function=web_search><parameter=query>'+url+'</parameter></function></tool_call>'
        self.assertEqual(rewrite_assistant(text,r),text)
        self.assertEqual(rewrite_assistant('```python\nurl="'+url+'"\n```',r),'```python\nurl="'+url+'"\n```')
        xml='<tool_call><function=web_extract><parameter=url>'+url+'</parameter></function></tool_call>'
        self.assertIn('<parameter=urls>["U1"]',rewrite_assistant(xml,r))

    def test_evidence_version_and_no_hidden_url_publication(self):
        r=URLRegistry('evidence');url='https://example.org/page';hidden='https://hidden.example/a'
        content='x'*4000+' '+hidden
        first=r.observe('web_extract',json.dumps({'results':[{'url':url,'content':content}]}))
        self.assertIn('D1@v1',first);self.assertNotIn(hidden,r.by_url);r.begin_turn()
        cached=r.cached_evidence('web_extract',{'urls':['D1@v1#E3']})
        self.assertIn(hidden,cached)
        rendered=r.observe('web_extract',cached)
        self.assertIn('U2',rendered)
        with self.assertRaises(ValueError):r.resolve('U2')
        r.begin_turn();self.assertEqual(r.resolve('U2'),hidden)
        second=r.observe('web_extract',json.dumps({'results':[{'url':url,'content':'changed'}]}))
        self.assertIn('D1@v2',second)
        self.assertEqual(r.documents['D1@v1']['chunks'][2],content[3200:4800])

    def test_malformed_source_data_does_not_create_guessed_addresses(self):
        r=URLRegistry('ref')
        broken='{"results":[{"url":"https://example.org/incomplete'
        self.assertEqual(r.observe('web_search',broken),broken)
        self.assertEqual(r.entries,{})
        bad=self.search('https://[invalid')
        self.assertIn('https://[invalid',r.observe('web_search',bad))
        self.assertEqual(r.entries,{})

    def test_compression_keeps_registry_without_cross_task_leak(self):
        from glasseek_shared.hermes.url_registry import carry_registry
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'GLASSEEK_URL_STATE_DIR':directory,'GLASSEEK_URL_MODE':'ref'}):
            old=runtime_registry('old');old.observe('web_search',self.search());persist(old)
            carry_registry('old','compressed')
            self.assertEqual(runtime_registry('compressed').resolve('U1'),'https://example.org/a?token=abc&x=2')
            with self.assertRaises(ValueError):runtime_registry('unrelated').resolve('U1')

    def test_persistence_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'GLASSEEK_URL_STATE_DIR':directory}):
            a=runtime_registry('task-a','ref');a.observe('web_search',self.search());persist(a)
            b=runtime_registry('task-b','ref')
            with self.assertRaises(ValueError):b.resolve('U1')
            restored=URLRegistry('ref',json.loads(a.state_path.read_text()));self.assertEqual(restored.resolve('U1'),'https://example.org/a?token=abc&x=2')

if __name__=='__main__':unittest.main()
