"""Protocol-level transformations, including negative and preservation cases."""
import copy
import json
import unittest

from glasseek_shared.hermes.qwen_xml_protocol import strip_batch_wrappers,has_batch_wrapper
from glasseek_shared.trajectory_release import validate_messages
from tool_xml import serialize,normalize_text
from tool_prompt import reconcile


class XMLFramingTests(unittest.TestCase):
    def test_complete_wrapper_removal_preserves_reasoning_and_parameter_data(self):
        value='literal\n<tool_calls_begin>\ninside a parameter'
        call={'name':'web_search','arguments':{'query':value}}
        body=serialize(call)
        text='<think>Quoted <tool_calls_begin> text.</think>\n<tool_calls_begin>\n'+body+'\n</tool_calls_end>\n'
        converted,events=strip_batch_wrappers(text)
        self.assertEqual(len(events),1);self.assertFalse(has_batch_wrapper(converted))
        self.assertIn('<think>Quoted <tool_calls_begin> text.</think>',converted)
        self.assertEqual(normalize_text(converted)[1],[call])
        self.assertEqual(normalize_text(text)[1],[call])

    def test_ambiguous_or_incomplete_frames_are_not_silently_discarded(self):
        call=serialize({'name':'web_search','arguments':{'query':'A'}})
        for text in ('<tool_calls_begin>\n'+call,
                     call+'\n</tool_calls_end>',
                     '<tool_calls_begin>\nImportant factual text\n'+call+'\n</tool_calls_end>',
                     '<tool_calls_begin>\n<tool_calls_begin>\n'+call+'\n</tool_calls_end>\n</tool_calls_end>'):
            with self.assertRaisesRegex(ValueError,'unbalanced_tool_batch_wrapper'):normalize_text(text)

    def test_strict_validator_rejects_legacy_frame_even_with_valid_inner_call(self):
        schema={'web_search':{'name':'web_search','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}}
        call={'name':'web_search','arguments':{'query':'A'},'call_id':'a'}
        text='<tool_calls_begin>\n'+serialize(call)+'\n</tool_calls_end>'
        messages=[{'role':'user','content':'Question'},{'role':'assistant','content':text,'call_links':[call]},
                  {'role':'user','content':'<tool_response>Evidence</tool_response>','tool_call_id':'a','response_to':'a'},
                  {'role':'assistant','content':'Answer'}]
        with self.assertRaisesRegex(ValueError,'legacy_tool_batch_wrapper'):validate_messages(messages,schema)

    def test_prompt_task_requirements_survive_schema_and_example_reconciliation(self):
        schema=json.dumps({'name':'old_search','parameters':{'type':'object','properties':{}}})
        task='Keep all final answers inside <answer></answer>.\nUse the supplied research state without discarding verified facts.'
        suffix='\nAlways preserve contradictory evidence and answer in the requested language.'
        prompt=task+'\n<tools>'+schema+'</tools>\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'+suffix
        item={'instruction_messages':[{'role':'system','content':prompt,'before_step':1,'source_message_index':0}]}
        flags=reconcile(item);text=item['instruction_messages'][0]['content']
        self.assertIn(task,text);self.assertTrue(text.endswith(suffix))
        self.assertNotIn('old_search',text);self.assertNotIn('<tool_call>\n{',text)
        self.assertIn('<function=example_function_name>',text)
        self.assertNotIn('tool_prompt_protocol_review_required',flags)
        self.assertEqual(item['tool_prompt_provenance']['events'][0]['original_content'],prompt)

    def test_unknown_tools_block_is_preserved_and_marked_for_review(self):
        text='Task instructions.\n<tools>This is not a JSON function declaration.</tools>'
        item={'instruction_messages':[{'role':'system','content':text,'before_step':1}]}
        self.assertIn('tool_prompt_protocol_review_required',reconcile(item))
        self.assertEqual(item['instruction_messages'][0]['content'],text)

    def test_xml_schema_declarations_are_metadata_and_task_text_is_preserved(self):
        schema='<function><name>lookup</name><description>Read records.</description><parameters><parameter><name>query</name><type>string</type></parameter></parameters></function>'
        text='Keep every task constraint.\n<tools>'+schema+'</tools>\nKeep the final answer requirement.'
        item={'instruction_messages':[{'role':'system','content':text,'before_step':1}]}
        flags=reconcile(item);converted=item['instruction_messages'][0]['content']
        self.assertNotIn('tool_prompt_protocol_review_required',flags)
        self.assertTrue(converted.startswith('Keep every task constraint.'))
        self.assertTrue(converted.endswith('Keep the final answer requirement.'))
        self.assertNotIn('<name>lookup</name>',converted)

    def test_literal_newlines_in_prompt_schema_descriptions_are_lossless_metadata(self):
        text='Task.\n<tools>{"name":"lookup","description":"line one\nline two","parameters":{"type":"object"}}</tools>\nAnswer requirement.'
        item={'instruction_messages':[{'role':'system','content':text,'before_step':1}]}
        self.assertNotIn('tool_prompt_protocol_review_required',reconcile(item))
        self.assertEqual(item['tool_prompt_provenance']['events'][0]['original_content'],text)

    def test_recognizable_obsolete_schema_is_replaced_without_claiming_it_parsed(self):
        text='Keep the task constraints.\n<tools>{"type":"function","function":{"name":"lookup","description":"an invalid "quoted" description","parameters":{"type":"object"}}}</tools>\nKeep the language requirement.'
        item={'instruction_messages':[{'role':'system','content':text,'before_step':1}]}
        flags=reconcile(item)
        self.assertIn('source_tool_declaration_syntax_unverified',flags)
        self.assertNotIn('tool_prompt_protocol_review_required',flags)
        self.assertTrue(item['instruction_messages'][0]['content'].startswith('Keep the task constraints.'))
        self.assertTrue(item['instruction_messages'][0]['content'].endswith('Keep the language requirement.'))
        self.assertEqual(item['tool_prompt_provenance']['events'][0]['original_content'],text)


if __name__=='__main__':unittest.main()
