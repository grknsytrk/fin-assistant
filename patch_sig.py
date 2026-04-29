import re
with open('frontend/src/pages/MarketsView.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix IndexLineChart Signature
old_sig = """function IndexLineChart({

    points,
    prevClose,
    changePct,
}: {
    points: MarketIndexDetailResponse['line_points'];
    prevClose: number | null;
    changePct: number | null;
}) {"""
new_sig = """function IndexLineChart({
    symbol,
    points,
    prevClose,
    changePct,
}: {
    symbol?: string;
    points: MarketIndexDetailResponse['line_points'];
    prevClose: number | null;
    changePct: number | null;
}) {"""
content = content.replace(old_sig, new_sig)

# Write back
with open('frontend/src/pages/MarketsView.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Sig replacement done.")
