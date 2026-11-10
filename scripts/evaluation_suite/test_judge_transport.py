import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

from judge_transport import adapt


class TransportTests(unittest.TestCase):
    def test_modality_is_never_removed(self):
        with self.assertRaises(ValueError):adapt({'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:...'}}]}]}, {'model':'flash','text_only':True})

    @unittest.skipUnless(importlib.util.find_spec('openai') and importlib.util.find_spec('pydantic'),'score environment required')
    def test_sdk_parse_uses_author_schema_and_accounts_invalid_json(self):
        import httpx
        from openai import OpenAI
        from pydantic import BaseModel
        from typing import Literal
        from judge_transport import install
        class Verdict(BaseModel):
            correct:Literal['yes','no']
            strict:Literal[True]
        captured=[]
        def handle(req):
            captured.append(json.loads(req.content))
            return httpx.Response(200,json={'id':'offline','object':'chat.completion','created':0,'model':'flash',
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{"correct":"yes","strict":true}'}}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})
        with tempfile.TemporaryDirectory() as d:
            usage=Path(d)/'api_usage.jsonl';install({'model':'flash','transport':'json_object_local_schema','usage_log':str(usage),'responses_as_chat':True})
            client=OpenAI(api_key='offline-test-only',http_client=httpx.Client(transport=httpx.MockTransport(handle)))
            result=client.beta.chat.completions.parse(model='original-model',messages=[{'role':'user','content':'unchanged native rubric'}],response_format=Verdict,max_completion_tokens=4096)
            self.assertEqual(result.choices[0].message.parsed.correct,'yes')
            self.assertEqual(captured[0]['messages'][-1]['content'],'unchanged native rubric')
            self.assertEqual(captured[0]['model'],'flash');self.assertEqual(captured[0]['response_format'],{'type':'json_object'})
            self.assertEqual(json.loads(usage.read_text())['usage']['total_tokens'],120)
            result=client.responses.create(model='original',input='unchanged BC+ rubric',max_output_tokens=4096)
            self.assertEqual(result.output_text,'{"correct":"yes","strict":true}')
            self.assertEqual(captured[-1]['messages'],[{'role':'user','content':'unchanged BC+ rubric'}])
            client.close()


if __name__=='__main__':unittest.main()
