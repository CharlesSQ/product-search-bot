SCHEMA_INSTRUCTIONS = """
Using the Data source generate a code snippet with a JSON object formatted in the following schema:

```json 
{{
    "query": a simple text string to compare to 
document contents
    "filter":  `comparison_statement | or(comparison_statement_1, comparison_statement_2, comparison_statement_3, ...) | and(comparison_statement_1, comparison_statement_2, .comparison_statement_3, ..)`
    "limit": optional. int, the number of documents to retrieve
}}
```.

"comparison_statement":
 `eq | ne | lt | lte | gt | gte(string name of attribute to apply the comparison to in double quotes, the comparison value")`:

Make sure that query doesn't refere to the filter operators.

Make sure that you only use the comparators and logical operators listed above and no others.

Make sure that filters only refer to attributes that exist in the data source.

Make sure that filters only use the attributed names with its function names if there are functions applied on them.

Make sure that filters only use format `YYYY-MM-DD` when handling timestamp data typed values.

If there are no filters that should be applied the filter value must be "NO_FILTER"

Data Source:
```json
{{
    "content": "{content}",
    "attributes": {attributes}
}}
```
"""

SUFFIX = """\
{query}
Respond only in Json format!
"""
