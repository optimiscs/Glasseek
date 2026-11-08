"""Explain mechanical gates without equating them to semantic quality verdicts."""


def categorize(reasons, error=''):
    categories = set()
    for reason in reasons:
        if reason.startswith(('unsupported_tool:', 'tool_runtime:')):
            categories.add('unsupported_tools')
        elif reason.startswith('tool_schema:'):
            categories.add('format_review')
        elif reason == 'annotation_size_budget_exceeded':
            categories.add('length_budget')
        elif reason == 'hierarchical_bundle_adapter_required':
            categories.add('adapter_holds')
        elif reason in ('missing_final_answer', 'missing_observation', 'unanswered_user_message'):
            categories.add('incomplete_episodes')
        elif reason == 'normalization_error':
            if any(s in error for s in ('Empty trajectory','Empty question','Empty tool call','Invalid message object')):
                categories.add('quality_issues')
            elif 'AdapterUnsupported' in error or any(s in error for s in ('needs adapter','adapter required','collides with XML','Unsupported multipart')):
                categories.add('adapter_holds')
            else:
                categories.add('format_review')
        elif reason in ('invalid_search_query','invalid_search_limit','invalid_calls','invalid_call','invalid_arguments',
                        'conflicting_search_limits','conflicting_extract_urls','duplicate_tool_call_id',
                        'unmatched_tool_response_id','orphan_tool_observation','tool_response_name_mismatch'):
            categories.add('quality_issues')
        else:
            categories.add('format_review')
    return sorted(categories)
