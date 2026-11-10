"""Protocol evidence only: no result/page boundary or success is invented."""
import hashlib
import json
import unittest

from glasseek_shared.web_return_audit import analyze_search_return
from glasseek_shared.search_protocol import decode_query_array,validate_queries


class SearchReturnAuditTests(unittest.TestCase):
    def audit(self,query,body):
        original=body;result=analyze_search_return(query,body,{'call_id':'batch','id_origin':'native'})
        self.assertEqual(body,original)
        self.assertEqual(result['raw_response_sha256'],hashlib.sha256(body.encode()).hexdigest())
        self.assertEqual(result['association']['status'],'verified')
        return result

    def test_explicit_json_labels_and_nested_result_string(self):
        body='<tool_response>Tool: search\nResult: '+json.dumps([{'query':'A','result':json.dumps([{'title':'Article','url':'https://example.org'}])}])+'</tool_response>'
        r=self.audit(['A'],body)
        self.assertEqual(r['coverage']['status'],'complete_labels')
        self.assertEqual(r['execution']['status'],'unreported')

    def test_query_headings_preserve_batch_and_do_not_imply_success(self):
        body="A Example Engine search for 'A' found 2 results:\nEvidence A\nA Example Engine search for 'B' found 1 results:\nEvidence B"
        r=self.audit(['A','B'],body)
        self.assertTrue(r['coverage']['per_query_association_verified'])
        self.assertEqual(r['execution']['status'],'unreported')

    def test_shared_array_title_is_not_per_query_coverage(self):
        body="### A Example Engine search for '[\"A\", \"B\"]' found 4 results:\nOne shared list"
        r=self.audit(['A','B'],body)
        self.assertEqual(r['coverage']['status'],'shared_batch_echo')
        self.assertFalse(r['coverage']['per_query_association_verified'])

    def test_partial_and_rewritten_labels_keep_batch_identity(self):
        for label,expected in [('A','partial_labels'),('A credible source','rewritten_or_unmatched_labels')]:
            body="A Example Engine search for '"+label+"' found 9 results:\nHistorical evidence"
            r=self.audit(['A','B'],body)
            self.assertEqual(r['coverage']['status'],expected)
            self.assertTrue(r['coverage']['needs_review'])
            self.assertEqual(r['execution']['status'],'unreported')

    def test_error_is_reported_failure_even_when_association_is_valid(self):
        r=self.audit(['A','B'],'<tool_response>Error: Tool call is not a valid JSON.</tool_response>')
        self.assertEqual(r['coverage']['status'],'not_reported')
        self.assertEqual(r['execution']['status'],'reported_failure')

    def test_api_reports_can_be_mixed_without_losing_query_results(self):
        body=json.dumps({'results':[{'query':'A','result':{'success':True,'data':{'web':[]}}},
                                    {'query':'B','result':{'success':False,'error':'network error'}}]})
        r=self.audit(['A','B'],body)
        self.assertEqual(r['coverage']['status'],'complete_labels')
        self.assertEqual(r['execution']['status'],'mixed_reports')
        self.assertEqual(r['execution']['reported_failure_count'],1)

    def test_top_level_query_failure_is_not_counted_twice(self):
        r=self.audit('A',json.dumps({'query':'A','error':'failed'}))
        self.assertEqual(r['execution']['reported_failure_count'],1)

    def test_nested_json_strings_can_carry_labels_without_rewriting(self):
        body=json.dumps([json.dumps({'query':q,'result':'Evidence'}) for q in ('A','B')])
        r=self.audit(['A','B'],body)
        self.assertEqual(r['coverage']['status'],'complete_labels')
        r=self.audit(['A','B'],json.dumps(['Evidence A','Evidence B']))
        self.assertEqual(r['coverage']['status'],'not_reported')

    def test_duplicate_queries_do_not_create_unique_index_claims(self):
        r=self.audit(['A','A'],json.dumps([{'query':'A','result':'one'},{'query':'A','result':'two'}]))
        self.assertEqual(r['coverage']['status'],'duplicate_query_labels')

    def test_search_documents_error_fields_are_not_backend_status(self):
        r=self.audit('A',json.dumps([{'title':'Error handling','content':'An example','error':'quoted text','url':'https://example.org'}]))
        self.assertEqual(r['execution']['status'],'unreported')

    def test_array_strings_decode_but_quoted_scalar_queries_stay_exact(self):
        for value in ('["A", "B"]',json.dumps('["A", "B"]')):
            self.assertEqual(decode_query_array(value),['A','B'])
        self.assertEqual(decode_query_array('"exact phrase"'),'"exact phrase"')
        with self.assertRaises(ValueError):validate_queries(decode_query_array('[1,2]'))

    def test_concatenated_original_json_returns_keep_explicit_query_evidence(self):
        parts=[json.dumps({'query':q,'result':'Evidence '+q}) for q in ('A','B')]
        body='\n\n'.join(parts)
        r=analyze_search_return(['A','B'],body,{'call_id':'batch'},parts)
        self.assertEqual(r['coverage']['status'],'complete_labels')
        self.assertEqual(r['execution']['status'],'unreported')
        self.assertEqual(r['representation'],'concatenated_call_returns')
        self.assertEqual(r['raw_response_sha256'],hashlib.sha256(body.encode()).hexdigest())

    def test_zero_results_is_not_missing_association_or_reported_execution_failure(self):
        r=self.audit('A',"A Example Engine search for 'A' found 0 results:\n")
        self.assertEqual(r['coverage']['status'],'complete_labels')
        self.assertFalse(r['query_results']['all_queries_have_reported_nonempty_results'])
        self.assertEqual(r['execution']['status'],'unreported')

    def test_reported_execution_success_does_not_imply_nonempty_results(self):
        r=self.audit('A',json.dumps({'query':'A','result':{'success':True,'data':{'web':[]}}}))
        self.assertEqual(r['execution']['status'],'reported_success')
        self.assertFalse(r['query_results']['all_queries_have_reported_nonempty_results'])

    def test_shared_batch_result_counts_cannot_be_allocated_to_queries(self):
        r=self.audit(['A','B'],"A Example Engine search for '[\"A\", \"B\"]' found 4 results:\nOne list")
        self.assertIsNone(r['query_results']['all_queries_have_reported_nonempty_results'])

    def test_explicit_query_result_lists_prove_payload_presence_only(self):
        r=self.audit(['A','B'],json.dumps([{'query':q,'result':[{'title':'Evidence'}]} for q in ('A','B')]))
        self.assertTrue(r['query_results']['all_queries_have_reported_nonempty_results'])
        self.assertEqual(r['execution']['status'],'unreported')


if __name__=='__main__':unittest.main()
