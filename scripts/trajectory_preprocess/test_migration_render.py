import unittest
from migration_render import target_messages

class RenderTests(unittest.TestCase):
    def test_protocol_envelope_only(self):
        original='<think>Keep reasoning.</think>\n<report>\nEvidence https://example.org\n</report>'
        item={'agent_role':'child','messages':[{'role':'assistant','content':original}]}
        messages,changes=target_messages(item)
        self.assertEqual(messages[0]['content'],'<think>Keep reasoning.</think>\nEvidence https://example.org')
        self.assertEqual(item['messages'][0]['content'],original)
        self.assertEqual(changes,['child_final_report_envelope_removed'])
    def test_no_rewrite_of_parent_or_plain_report(self):
        for role,text in [('parent','<report>Answer</report>'),('child','Plain report'),('child','<think><report>quoted</report></think>Plain report')]:
            item={'agent_role':role,'messages':[{'role':'assistant','content':text}]}
            self.assertEqual(target_messages(item),(item['messages'],[]))
if __name__=='__main__':unittest.main()
