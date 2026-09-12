import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import os
from reranker import rerank


def test_reranker_alone():
    query = "chair makes squeaking noise when leaning back"

    candidates = [
        {
            "doc_id": "doc_1",
            "content": "RECLINE SQUEAKING NOISE: Audible squeaking when leaning back in the chair. Root Cause: Loose screw set S-4012-SCRW under base P-9913-BASE. Resolution: Tighten all 4 screws to 12.5 Nm torque.",
            "sku": "CHAIR-ERG-X99",
            "product_name": "ErgoFlex Pro Office Chair",
            "part_numbers": ["P-9913-BASE", "S-4012-SCRW"],
            "combined_score": 0.02,
        },
        {
            "doc_id": "doc_2",
            "content": "HEIGHT ADJUSTMENT FAILURE: Chair sinks slowly or will not hold height position. Root Cause: Gas lift cylinder C-7721-GL internal seal failure. Replace cylinder C-7721-GL.",
            "sku": "CHAIR-ERG-X99",
            "product_name": "ErgoFlex Pro Office Chair",
            "part_numbers": ["C-7721-GL"],
            "combined_score": 0.018,
        },
        {
            "doc_id": "doc_3",
            "content": "DESK WON'T MOVE UP/DOWN: Control panel shows error E01 or no response. Root Cause: Motor M-8841-MOT overload or control box CB-2201-CTL fault. Run calibration.",
            "sku": "DESK-STD-MOTO",
            "product_name": "Motorized Standing Desk Standard",
            "part_numbers": ["M-8841-MOT", "CB-2201-CTL"],
            "combined_score": 0.005,
        },
        {
            "doc_id": "doc_4",
            "content": "CURSOR JITTER / ERRATIC MOVEMENT: Cursor jumps, stutters, or moves randomly. Root Cause: Sensor lens dirty, surface incompatibility, or sensor SN-8801-SNS fault.",
            "sku": "MOUSE-VERT-PRO",
            "product_name": "Vertical Ergonomic Mouse Pro",
            "part_numbers": ["SN-8801-SNS"],
            "combined_score": 0.003,
        },
        {
            "doc_id": "doc_5",
            "content": "MICROPHONE NOT WORKING: Mic muted, low volume, or static. Root Cause: Mic boom MB-3301-BM not seated, mic capsule MC-4401-MIC fault. Re-pair dongle DG-2201-DGL.",
            "sku": "HEADSET-WL-PRO",
            "product_name": "Wireless Pro Headset",
            "part_numbers": ["MC-4401-MIC", "DG-2201-DGL"],
            "combined_score": 0.001,
        },
    ]

    print("=" * 70)
    print("RERANKER ALONE TEST")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"\nTesting {len(candidates)} hand-crafted candidates (mix of relevant/irrelevant)")
    print("-" * 70)

    results = rerank(query, candidates, top_n=5)

    print(f"\n{'Rank':<4} {'Score':<8} {'Doc ID':<8} {'SKU':<20} {'Content Preview'}")
    print("-" * 70)
    for i, r in enumerate(results):
        preview = r["content"][:80].replace("\n", " ")
        print(f"{i+1:<4} {r['rerank_score']:<8.4f} {r['doc_id']:<8} {r['sku']:<20} {preview}...")

    print("\n" + "=" * 70)
    print("EXPECTED: doc_1 (squeaking chair) should rank #1 with highest score")
    print("         doc_2 (height adjustment) should rank #2 (same chair, related)")
    print("         doc_3, doc_4, doc_5 (desk, mouse, headset) should rank lower")
    print("=" * 70)


if __name__ == "__main__":
    test_reranker_alone()