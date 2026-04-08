import random

# Load training data
with open('data/splits_no_test/train_sources.txt', 'r', encoding='utf-8') as f:
    train_sources = [line.strip() for line in f]

with open('data/splits_no_test/train_targets.txt', 'r', encoding='utf-8') as f:
    train_targets = [line.strip() for line in f]

# Load validation data
with open('data/splits_no_test/val_sources.txt', 'r', encoding='utf-8') as f:
    val_sources = [line.strip() for line in f]

with open('data/splits_no_test/val_targets.txt', 'r', encoding='utf-8') as f:
    val_targets = [line.strip() for line in f]

# Create indices and shuffle
random.seed(42)
train_indices = list(range(len(train_sources)))
val_indices = list(range(len(val_sources)))

random.shuffle(train_indices)
random.shuffle(val_indices)

# Take 20% of each
train_20_percent = int(len(train_sources) * 0.2)
val_20_percent = int(len(val_sources) * 0.2)

train_indices_20 = train_indices[:train_20_percent]
val_indices_20 = val_indices[:val_20_percent]

# Write 20% subsets
with open('data/splits_no_test/train_sources_20pct.txt', 'w', encoding='utf-8') as f:
    for idx in sorted(train_indices_20):
        f.write(train_sources[idx] + '\n')

with open('data/splits_no_test/train_targets_20pct.txt', 'w', encoding='utf-8') as f:
    for idx in sorted(train_indices_20):
        f.write(train_targets[idx] + '\n')

with open('data/splits_no_test/val_sources_20pct.txt', 'w', encoding='utf-8') as f:
    for idx in sorted(val_indices_20):
        f.write(val_sources[idx] + '\n')

with open('data/splits_no_test/val_targets_20pct.txt', 'w', encoding='utf-8') as f:
    for idx in sorted(val_indices_20):
        f.write(val_targets[idx] + '\n')

print(f'Train: {len(train_sources)} → {train_20_percent} samples (20%)')
print(f'Val: {len(val_sources)} → {val_20_percent} samples (20%)')
print('✓ Created 20% stratified subsets')
