import unittest
from analyze_browsecomp_url_transfer import classify,urlparts,output_status,change_type,changes_redirect_target

class URLTransferTests(unittest.TestCase):
    def test_exact_parameter_changes_and_query_order(self):
        old='https://example.org/path?a=1&b=2';seen=({old},{urlparts(old)['base']:[old]})
        self.assertEqual(classify(old,seen)[0],'exact')
        self.assertEqual(classify('https://example.org/path?b=2&a=1',seen)[0],'normalized_same_query')
        self.assertEqual(classify('https://example.org/path?a=9',seen)[0],'query_changed_same_host_path')
        self.assertEqual(change_type(old,'https://example.org/path?a=1'),'removed_only')
        self.assertEqual(classify('https://example.org/other',seen)[0],'not_seen')
        self.assertTrue(changes_redirect_target('https://google.com/goto?url=https%3A%2F%2Fa.com','https://google.com/goto?url=https%3A%2F%2Fb.com'))

    def test_never_attribute_a_different_returned_url_to_request(self):
        result={'results':[{'url':'https://other.test/','content':'valid content','error':None}]}
        self.assertEqual(output_status(result,'https://example.org/',0,1)[:2],('unverified_result_url','different_or_missing'))

    def test_unicode_percent_encoding_and_block_signals(self):
        u='https://example.org/字';r={'results':[{'url':'https://example.org/%E5%AD%97','content':'Example page','error':None}]}
        self.assertEqual(output_status(r,u,0,1)[:2],('content_without_block_signal','normalized'))
        r={'results':[{'url':u,'content':'Captcha check','error':None}]}
        self.assertEqual(output_status(r,u,0,1)[:2],('content_with_block_signal','exact'))
        self.assertEqual(output_status({'error':'bad'},u,0,1)[:2],('call_error','missing'))

if __name__=='__main__':unittest.main()
