import json
import gzip
from pathlib import Path
import sqlite3
import tempfile
import unittest

from client import digest
from reporting.cleaning import report


class CleaningReportTests(unittest.TestCase):
    def test_final_outcomes_and_overlapping_rules_account_for_every_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for folder in ('intake','searchswarm','unified','safety/export'):(run/folder).mkdir(parents=True)
            def write(path,value): (run/path).write_text(json.dumps(value))
            def lines(path,values): (run/path).write_text(''.join(json.dumps(v)+'\n' for v in values))
            write('manifest.json',{})
            write('intake/cleaning_summary.json',{'status':'completed','sources':{'Example':{'scanned':3}}})
            write('searchswarm/summary.json',{'status':'completed','counts':{'bundles':0,'children_read':0}})
            write('safety/export/summary.json',{'status':'completed','strict_protocol_verified':True,
                  'input_records':1,'exported':1,'protocol_metrics':{'passed':1}})
            for path in ('searchswarm/held.jsonl','searchswarm/duplicates.jsonl',
                         'unified/exact_duplicates.jsonl','safety/export/withheld.jsonl'): lines(path,[])
            lines('intake/held.jsonl',[{'id':'b','source':'Example','migration_rules':['search','argument.source'],
                                      'reasons':['unmapped_search_argument:source']}])
            lines('intake/duplicates.jsonl',[{'id':'c','source':'Example','migration_rules':['search']}])
            item={'id':'a','source':'Example','migration_rules':['search','extract']}
            lines('unified/candidates.jsonl',[item]);lines('safety/export/candidates.jsonl',[item])
            db=sqlite3.connect(run/'safety/state.sqlite')
            db.execute('CREATE TABLE decisions(row_index INTEGER,result TEXT)')
            db.execute('INSERT INTO decisions VALUES(0,?)',(json.dumps({'source_row':0,'content_hash':digest(item),'findings':[]}),))
            db.commit();db.close()
            summary=report(run)
            self.assertEqual(summary['totals'],{'read':3,'migrated':1,'held':1,'quarantine':0,'duplicate':1})
            rules={r['rule']:r for r in summary['rule_counts']}
            self.assertEqual((rules['search']['migrated'],rules['search']['held'],rules['search']['duplicate']),(1,1,1))
            self.assertEqual((rules['argument.source']['migrated'],rules['argument.source']['held']),(0,1))
            self.assertEqual((run/'final/offline_passed.jsonl').resolve(),run/'safety/export/candidates.jsonl')
            with gzip.open(run/'safety/export/candidates.jsonl.gz','wt') as handle:
                handle.write((run/'safety/export/candidates.jsonl').read_text())
            export=json.loads((run/'safety/export/summary.json').read_text())
            export['candidate_file']='candidates.jsonl.gz';write('safety/export/summary.json',export)
            (run/'final/offline_passed.jsonl').unlink();(run/'safety/export/candidates.jsonl').unlink()
            self.assertEqual(report(run)['totals'],summary['totals'])
            self.assertEqual((run/'final/offline_passed.jsonl.gz').resolve(),run/'safety/export/candidates.jsonl.gz')
            # Missing rows must never be reported as a complete full-corpus run.
            write('intake/cleaning_summary.json',{'status':'completed','sources':{'Example':{'scanned':4}}})
            with self.assertRaises(AssertionError):report(run)
