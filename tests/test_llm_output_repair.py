"""
Demonstration of JSON Repair Capability
This script shows how the system repairs malformed LLM JSON output
"""

import json
import re

def fix_regex_patterns(text):
    """Fix regex patterns in JSON by properly escaping backslashes"""
    def escape_regex(match):
        regex_value = match.group(2)
        # Replace single backslashes with double backslashes
        escaped_regex = regex_value.replace('\\', '\\\\')
        return f'{match.group(1)}"{escaped_regex}"{match.group(3)}'
    
    # Pattern to match: "regex": "pattern"
    return re.sub(r'("regex"\s*:\s*)"([^"]*)"(\s*[,}])', escape_regex, text)


# Simulated malformed LLM output
malformed_json = '''
{
  "column": "email",
  "expectations": [
    { "type": "expect_column_values_to_match_regex",
      "regex": "^[^@]+@[^@]+$"
'''

print("="*80)
print("JSON REPAIR DEMONSTRATION")
print("="*80)
print("\n1️⃣  MALFORMED LLM OUTPUT (Missing closing brackets):")
print("-" * 80)
print(malformed_json)
print("-" * 80)

# Try to parse - will fail
print("\n2️⃣  ATTEMPTING TO PARSE MALFORMED JSON...")
try:
    json.loads(malformed_json)
    print("✅ Parsed successfully (unexpected)")
except json.JSONDecodeError as e:
    print(f"❌ JSON Parse Error: {e}")
    print(f"   Error at position {e.pos}, line {e.lineno}, column {e.colno}")

# Repair attempt 1: Add missing closing brackets
print("\n3️⃣  REPAIR ATTEMPT: Adding missing brackets...")
repaired_json = malformed_json.strip()
if not repaired_json.endswith('}'):
    # Count opening vs closing brackets
    open_braces = repaired_json.count('{')
    close_braces = repaired_json.count('}')
    open_brackets = repaired_json.count('[')
    close_brackets = repaired_json.count(']')
    
    # Add missing closures
    for _ in range(open_brackets - close_brackets):
        repaired_json += '\n    ]'
    for _ in range(open_braces - close_braces):
        repaired_json += '\n  }'

print("REPAIRED JSON:")
print("-" * 80)
print(repaired_json)
print("-" * 80)

# Try parsing again
print("\n4️⃣  PARSING REPAIRED JSON...")
try:
    parsed = json.loads(repaired_json)
    print("✅ SUCCESSFULLY PARSED!")
    print("\nParsed structure:")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print(f"❌ Still failed: {e}")

# Example 2: Regex escape issue
print("\n" + "="*80)
print("EXAMPLE 2: REGEX ESCAPE ISSUE")
print("="*80)

malformed_regex = '''
{
  "column": "phone",
  "expectations": [
    {
      "type": "expect_column_values_to_match_regex",
      "regex": "^\d{3}-\d{3}-\d{4}$"
    }
  ]
}
'''

print("\n1️⃣  MALFORMED JSON (Unescaped backslashes in regex):")
print("-" * 80)
print(malformed_regex)
print("-" * 80)

print("\n2️⃣  ATTEMPTING TO PARSE...")
try:
    json.loads(malformed_regex)
    print("❌ This might parse but regex is incorrect")
except json.JSONDecodeError as e:
    print(f"❌ JSON Parse Error: {e}")

print("\n3️⃣  APPLYING REGEX FIX...")
fixed_regex = fix_regex_patterns(malformed_regex)
print("REPAIRED JSON:")
print("-" * 80)
print(fixed_regex)
print("-" * 80)

print("\n4️⃣  PARSING FIXED JSON...")
try:
    parsed = json.loads(fixed_regex)
    print("✅ SUCCESSFULLY PARSED!")
    print("\nParsed structure:")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print(f"❌ Failed: {e}")

print("\n" + "="*80)
print("📸 EVIDENCE: Screenshot this output showing:")
print("   ✔ Malformed JSON examples")
print("   ✔ Repair process")
print("   ✔ Successfully parsed output")
print("="*80)
