import os

dataset_path = "dataset/train"

class_counts = {}

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[class_name] = count

# Sort classes
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])

print("\n📊 CLASS DISTRIBUTION:\n")
for cls, count in sorted_classes:
    print(f"{cls}: {count}")

print("\n⚠️ LOW DATA CLASSES (<100 images):")
for cls, count in sorted_classes:
    if count < 100:
        print(f"{cls}: {count}")