import json
import csv
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from reporting.examples import safe_window
from reporting.safety import report
from safety_rules import Scanner
import safety_pipeline


class SafetyReportTests(unittest.TestCase):
    def setUp(self):
        policy=json.loads(Path(__file__).with_name('safety_policy.json').read_text())
        policy['disabled_rules']=[]  # Regression of privacy-enabled historical reports.
        self.scanner=Scanner(policy,{},b'synthetic-report-key-000000000000000')

    def test_clipped_signed_url_stays_redacted(self):
        value='prefix https://example.org/file?X-Amz-Credential=AKIA'+'A'*16+'&X-Amz-Signature=private-test-signature suffix'
        start=value.index('AKIA');end=start+20
        rendered=safe_window(value,0,start,self.scanner)+safe_window(value,end,len(value),self.scanner)
        self.assertNotIn('private-test-signature',rendered)
        self.assertNotIn('AKIA',rendered)
        self.assertIn('prefix',rendered);self.assertIn('suffix',rendered)

    def test_partial_phone_and_email_stay_redacted(self):
        value='Phone 13800138000 email person@example.org done'
        self.assertNotIn('00138000',safe_window(value,9,len(value),self.scanner))
        self.assertNotIn('person@',safe_window(value,0,value.index('example'),self.scanner))

    def test_report_counts_causes_and_keeps_sensitive_values_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);key=b'synthetic-report-key-000000000000000';keyfile=root/'key';keyfile.write_bytes(key)
            contract={'schemas':{},'delegation':{'blocked_child_tools':[]}}
            scanner=Scanner(self.scanner.policy,contract,key)
            token='sk-'+'k'*32
            texts=['safe answer','contact person@example.org phone 13800138000',token+' person@example.org']
            items=[{'id':str(i),'source':'Synthetic','agent_role':'parent','tools':[],
                    'messages':[{'role':'user','content':'Question'},{'role':'assistant','content':text}]} for i,text in enumerate(texts)]
            source=root/'candidates.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in items))
            (root/'origins.jsonl').write_text(''.join(json.dumps({'canonical_id':str(i),'source':'Synthetic','agent_role':'parent'})+'\n' for i in range(3)))
            args=SimpleNamespace(input=source,output=root/'out',workers=1,resume=False,limit=None,max_record_bytes=100000)
            safety_pipeline.scan(args,self.scanner.policy,contract,scanner)
            report(args.output,keyfile)
            result=json.loads((args.output/'report/summary.json').read_text())
            self.assertEqual(result['totals'],{'scanned':3,'pass_rules':1,'review':1,'quarantine':1})
            causes={(r['state'],r['rule_id']):r['trajectories'] for r in result['disposition_reasons']}
            self.assertEqual(causes,{('review','privacy.email'):1,('review','privacy.phone'):1,('quarantine','credential.token'):1})
            with (args.output/'report/review_category_matrix.csv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle)),[{'dataset':'Synthetic','unique_trajectories':'1','privacy':'1'}])
            with (args.output/'report/quarantine_rule_matrix.csv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle)),[{'dataset':'Synthetic','unique_trajectories':'1','credential.token':'1'}])
            samples=(args.output/'report/audit_samples.json').read_text()
            self.assertNotIn(token,samples);self.assertNotIn('person@example.org',samples)


if __name__=='__main__':unittest.main()
