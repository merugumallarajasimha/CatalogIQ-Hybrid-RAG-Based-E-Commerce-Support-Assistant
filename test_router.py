import sys
sys.path.insert(0, 'src')
from query_router import classify_queries

# Test queries spanning all 6 categories from test_questions.csv plus new ones
test_queries = [
    # exact_lookup (from test_questions.csv)
    "What is the torque spec for screw S-4012-SCRW?",
    "Which part number is the gas lift cylinder for CHAIR-ERG-X99?",
    "What does error E02 mean on DESK-STD-MOTO?",
    
    # product_search (from test_questions.csv)
    "What is the warranty on CHAIR-ERG-X99?",
    "What are the specifications of CHAIR-ERG-X99?",
    
    # product_comparison (new)
    "Compare CHAIR-ERG-X99 vs DESK-STD-MOTO",
    "Which is better: CHAIR-ERG-X99 or DESK-STD-MOTO?",
    "Difference between MON-ARM-DUAL and MON-ARM-SINGLE",
    
    # troubleshooting (from test_questions.csv)
    "My chair squeaks when I lean back, how do I fix it?",
    "Standing desk won't move up or down, shows error E01",
    "Left side of desk lags behind right when raising",
    "Mouse cursor jumps around on glass desk",
    
    # general_question (from test_questions.csv + new)
    "How do I cook a turkey?",
    "What is the return policy?",
    
    # unsupported (new)
    "Write a poem about chairs",
    "What is the meaning of life?",
]

print("=" * 100)
print("QUERY ROUTER CLASSIFICATION - 17 Queries Spanning All 6 Categories")
print("=" * 100)

results = classify_queries(test_queries)
for r in results:
    print()
    print("Query:", r['query'])
    print("  Category:", r['category'])
    ev = r['evidence']
    if ev['has_identifier']:
        print("  Identifiers:", ev['identifiers'])
    if ev['comparison_signals']:
        print("  Comparison signals:", ev['comparison_signals'])
    if ev['troubleshooting_signals']:
        print("  Troubleshooting signals:", ev['troubleshooting_signals'])
    if ev['product_signals']:
        print("  Product signals:", ev['product_signals'])