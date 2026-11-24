# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###

    # 读取开发集，计算样本数量
    with open('birth_dev.tsv', encoding='utf-8') as f:
        n = sum(1 for line in f if line.strip())

    # 假设所有人的出生地都是 London
    predictions = ['London'] * n

    # 计算准确率
    total, correct = utils.evaluate_places('birth_dev.tsv', predictions)

    accuracy = correct / total * 100
    print(f'London Baseline Accuracy: {accuracy:.2f}%')

    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
