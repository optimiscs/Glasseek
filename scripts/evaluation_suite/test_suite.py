import json
import tempfile
import unittest
from pathlib import Path

from suite import Overlap,apportion,canonical,dump,jsonl,normalize_parent,select,sample_target
from run import prepare_score
from score import verify_packet


class SamplingTests(unittest.TestCase):
    def test_per_dataset_top_up_quotas(self):
        config={'fraction':.05,'per_benchmark':{'browsecomp':37,'hle':0}}
        self.assertEqual(sample_target(config,'browsecomp',1266),37)
        self.assertEqual(sample_target(config,'hle',2158),0)
        with self.assertRaises(ValueError):sample_target(config,'gaia',127)
        for invalid in [-1,True,1.5]:
            with self.assertRaises(ValueError):sample_target({'per_benchmark':invalid},'hle',2158)
        self.assertEqual(sample_target({'fraction':.05},'hle',2158),108)
        self.assertEqual(sample_target({'per_benchmark':100},'gaia',127),100)

    def test_corpus_events_are_scoped_to_the_current_run_and_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);release=root/'release';pred=root/'pred/current_browsecomp_plus/suite_browsecomp_plus'
            jsonl(release/'private/references.jsonl',[{'benchmark':'browsecomp_plus','id':'1','opaque_id':'t1',
                  'question':'question','answer':'a','reference':{}}])
            dump(release/'manifest.json',{});dump(release/'summary.json',[{'benchmark':'browsecomp_plus','protocol':'native'}])
            dump(pred/'conv/t1/r.json',{'question_id':'t1','question':'question','status':'success',
                                     'final_assistant_valid':True,'model_response':'answer'})
            events=root/'events.jsonl'
            jsonl(events,[{'opaque_id':'t1','namespace':'other_run','corpus_revision':'rev','retrieved_docids':['future'],'tool_call_counts':{'web_search':99}},
                {'opaque_id':'t1','namespace':'current_browsecomp_plus','corpus_revision':'rev','retrieved_docids':['seen'],'tool_call_counts':{'web_search':1}}])
            config=root/'corpus.json';dump(config,{'docid_events':str(events),'corpus_revision':'rev'})
            packet=root/'packet';prepare_score(release,packet,pred,'browsecomp_plus',config)
            item=json.loads((packet/'corpus-events.jsonl').read_text())
            self.assertEqual(item['retrieved_docids'],['seen']);self.assertEqual(item['tool_call_counts'],{'web_search':1})

    def test_quota_total_and_order_invariance(self):
        rows=[{'id':str(i),'stratum':str(i%3)} for i in range(127)]
        a,q=select(rows,6,20261108,'example');b,_=select(rows[::-1],6,20261108,'example')
        self.assertEqual(a,b);self.assertEqual(sum(q.values()),6)
        self.assertEqual(apportion({'a':10,'b':120},7),{'a':1,'b':6})
        self.assertNotEqual(a,select(rows,6,42,'example')[0])

    def test_exact_and_confirmed_near_exclusions(self):
        q='Find the research institution established in 1927 that published the report on coastal ecology in 2018, and identify its founding director.'
        o=Overlap({'b':[{'id':'1','question':q}]})
        o.add(q.upper().replace(',',''),{'source':'current34'})
        self.assertEqual(o.hits['b::1'][0]['method'],'exact')
        o=Overlap({'b':[{'id':'1','question':q}]})
        o.add(q.replace('Find','Locate',1),{'source':'train'})
        self.assertEqual(o.hits['b::1'][0]['method'],'near_jaccard')
        o=Overlap({'b':[{'id':'1','question':q}]})
        o.add('Follow these long harness instructions. '*10+'Question: '+q+' Provide a detailed answer.',{'source':'train'})
        self.assertEqual(o.hits['b::1'][0]['method'],'contained_question')
        o=Overlap({'b':[{'id':'1','question':q}]})
        o.add('Find the research institution founded in another country and explain the climate methodology in its annual study.',{'source':'train'})
        self.assertFalse(o.hits)
        self.assertNotEqual(canonical('year 1927'),canonical('year 1928'))

    def test_checklists_are_grouped_not_individual_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'lrb.jsonl'
            jsonl(p,[{'qid':'a','question_no_placeholder':'question a','category':'cat','checklist_no_placeholder':str(i)} for i in range(3)])
            rows,_=normalize_parent('liveresearchbench',{'files':[str(p)],'expected_population':1})
            self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]['reference']['checklists']),3)
            with self.assertRaises(ValueError):normalize_parent('liveresearchbench',{'files':[str(p)],'expected_population':3})

    def test_private_binary_metadata_is_losslessly_encoded(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'r.json';dump(p,{'binary':b'abc'})
            self.assertEqual(json.loads(p.read_text())['binary'],{'__bytes_base64__':'YWJj'})

    def test_packet_preserves_gold_ids_and_failed_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);release=root/'release';pred=root/'pred'
            jsonl(release/'private/references.jsonl',[
                {'benchmark':'gaia','id':'official-A','opaque_id':'t1','question':'question A','answer':'42','reference':{}},
                {'benchmark':'gaia','id':'official-B','opaque_id':'t2','question':'question B','answer':'7','reference':{}}])
            dump(release/'manifest.json',{})
            dump(release/'summary.json',[{'benchmark':'gaia','protocol':'official exact'}])
            dump(pred/'conv/t1/r.json',{'question_id':'t1','question':'question A','status':'success','final_assistant_valid':True,'model_response':'Reasoning\nFINAL ANSWER: 42'})
            prepare_score(release,root/'packet',pred,'gaia')
            m=json.loads((root/'packet/manifest.json').read_text())
            self.assertEqual(m['assigned'],2);self.assertEqual(m['completed'],1);self.assertEqual(m['failures'],1)
            item=json.loads((root/'packet/submission.jsonl').read_text())
            self.assertEqual(item,{'task_id':'official-A','model_answer':'42'})
            verify_packet(root/'packet')
            (root/'packet/submission.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):verify_packet(root/'packet')

    def test_native_report_table_and_attribution_packet_contracts(self):
        specs={
            'drbench':{'id':1,'prompt':'question'},
            'liveresearchbench':{'checklists':[{'checklist_no_placeholder':'criterion'}]},
            'mind2web2':{},
            'widesearch':{'instance_id':'case','query':'question','language':'en','evaluation':'{"required":["name"]}'}}
        for name,reference in specs.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);release=root/'release';pred=root/'pred'
                official='1' if name=='drbench' else 'case'
                jsonl(release/'private/references.jsonl',[{'benchmark':name,'id':official,'opaque_id':'t1','question':'question','answer':'','reference':reference}])
                dump(release/'manifest.json',{});dump(release/'summary.json',[{'benchmark':name,'protocol':'native'}])
                dump(pred/'conv/t1/r.json',{'question_id':'t1','question':'question','status':'success','final_assistant_valid':True,'model_response':'result'})
                packet=root/'packet';prepare_score(release,packet,pred,name);verify_packet(packet)
                if name=='mind2web2':self.assertTrue((packet/'answers/glasseek/case/answer_1.md').is_file())
                if name=='liveresearchbench':self.assertEqual(json.loads((packet/'benchmark_data.json').read_text())['case']['checklists'],['criterion'])
                if name=='widesearch':self.assertIsInstance(json.loads((packet/'queries.jsonl').read_text())['evaluation'],dict)
                if name=='drbench':self.assertEqual(json.loads((packet/'raw.jsonl').read_text())['id'],1)


if __name__=='__main__':unittest.main()
