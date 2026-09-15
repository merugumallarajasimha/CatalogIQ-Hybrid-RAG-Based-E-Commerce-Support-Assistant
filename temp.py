# Check file sizes - real data should be:
# products: ~1.1 GB, examples: ~50 MB, sources: ~1.7 MB

import os
for f in os.listdir('data'):
    size = os.path.getsize('data/' + f)
    print(f'{f}: {size/1e6:.1f} MB' if size > 1e6 else f'{f}: {size/1e3:.1f} KB')
