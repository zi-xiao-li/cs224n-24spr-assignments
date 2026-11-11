
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024：作业 2
parser_transitions.py：完成部分句法分析的算法。
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""

import sys

class PartialParse(object):
    def __init__(self, sentence):
        """初始化当前的部分句法分析。

        @param sentence (list of str): 待解析的句子，以单词列表的形式表示。
                                        你的代码不应修改原始句子。
        """
        # 保存待解析的句子，用于记录相关信息。不要在代码中修改它。
        self.sentence = sentence

        ### YOUR CODE HERE (3 Lines)
        ### 你的代码应初始化以下属性：
        ###     self.stack: 当前的栈，以列表形式表示，栈顶是列表的
        ###                 最后一个元素。
        ###     self.buffer: 当前的缓冲区，以列表形式表示，缓冲区的第一个元素
        ###                  是列表的第一个元素。
        ###     self.dependencies: 到目前为止生成的依存关系列表，以元组列表的形式表示，
        ###             每个元组的格式为 (head, dependent)。
        ###             该列表中元素的顺序不重要。
        ###
        ### 注意：根节点应使用字符串 "ROOT" 表示。
        ### 注意：如果需要使用 sentence 对象初始化任何属性，请确保不要直接
        ###       引用 sentence 对象。也就是说，切记不要修改原始 sentence 对象。

        self.stack = ["ROOT"]
        self.buffer = sentence.copy()
        self.dependencies = []


        ### END YOUR CODE


    def parse_step(self, transition):
        """通过将给定的转换操作应用于当前部分句法分析，执行一次解析步骤。

        @param transition (str): 一个值为 "S"、"LA" 或 "RA" 的字符串，分别表示
                                SHIFT（移进）、LEFT-ARC（左弧）和 RIGHT-ARC（右弧）操作。
                                可以假设传入的转换操作是合法的。
        """
        ### YOUR CODE HERE (~7-12 Lines)
        ### TODO:
        ###     实现一次句法分析步骤，即按照 PDF 作业说明中的描述，
        ###     实现以下转换操作的逻辑：
        ###         1. SHIFT（移进）
        ###         2. LEFT-ARC（左弧）
        ###         3. RIGHT-ARC（右弧）

        if transition == "S":
            self.stack.append(self.buffer.pop(0))

        elif transition == "LA":
            head = self.stack[-1]
            dependent = self.stack.pop(-2)
            self.dependencies.append((head, dependent))

        elif transition == "RA":
            dependent = self.stack.pop()
            head = self.stack[-1] # 因为已经移除dependent了，所以变成栈中的最后一个词
            self.dependencies.append((head, dependent))



        ### END YOUR CODE

    def parse(self, transitions):
        """将给定的一系列转换操作应用于当前 PartialParse 对象。

        @param transitions (list of str): 按照执行顺序排列的转换操作列表。

        @return dependencies (list of string tuples): 解析句子时生成的依存关系列表，
                                                       以元组列表的形式表示，
                                                       每个元组的格式为 (head, dependent)。
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies





def minibatch_parse(sentences, model, batch_size):
    """使用模型以小批量方式解析句子列表。

    @param sentences (list of list of str): 待解析的句子列表。
                                            （每个句子都是一个单词列表，每个单词都是字符串类型）
    @param model (ParserModel): 用于决定句法转换操作的模型。假设该模型具有
                                model.predict(partial_parses) 方法，该方法接收
                                PartialParse 对象列表作为输入，并返回每个解析对象
                                对应的预测转换操作列表。也就是说，调用
                                    transitions = model.predict(partial_parses)
                                之后，transitions[i] 就是要应用于
                                partial_parses[i] 的下一步转换操作。
    @param batch_size (int): 每个小批量中包含的 PartialParse 对象数量。

    @return dependencies (list of dependency lists): 一个列表，其中每个元素都是
                                                     对应句子解析得到的依存关系列表。
                                                     返回结果的顺序应与 sentences 中
                                                     的句子顺序一致（即 dependencies[i]
                                                     应包含 sentences[i] 的解析结果）。
    """
    dependencies = []

    ### YOUR CODE HERE (~8-10 Lines)
    ### TODO:
    ###     实现小批量句法分析算法。注意，该算法的伪代码已在 PDF 作业说明中给出。
    ###
    ###     注意：浅拷贝（如 PDF 中所示）可以通过 Python 的切片操作实现，例如：
    ###                 unfinished_parses = partial_parses[:]
    ###             这里，`unfinished_parses` 是 `partial_parses` 的浅拷贝。
    ###             在 Python 中，像 `unfinished_parses` 这样的浅拷贝列表不会创建
    ###             `partial_parses` 中所存储对象的新实例，而是两个列表引用相同的对象。
    ###             在本题中，`partial_parses` 包含一个部分句法分析对象列表，
    ###             `unfinished_parses` 引用的也是这些相同的对象。因此，不应使用
    ###             `del` 操作符删除 `unfinished_parses` 列表中的对象。
    ###             这可能释放 `partial_parses` 仍在访问的底层内存，
    ###             从而导致代码崩溃。

    # 为每个句子创建一个 PartialParse 对象，保存其当前解析状态
    partial_parses = [PartialParse(sentence) for sentence in sentences]

    # 浅拷贝列表：两个列表中的元素指向相同的 PartialParse 对象
    # 浅拷贝后，外层列表独立，内部对象共享
    unfinished_parses = partial_parses[:]

    # 只要还有未完成解析的句子，就继续循环
    while unfinished_parses: # 列表具有布尔值

        # 取前 batch_size 个未完成的解析对象，组成一个 minibatch
        minibatch = unfinished_parses[:batch_size]
        transitions = model.predict(minibatch)

        # zip 将每个解析对象与其对应的预测操作配对
        for pp, transition in zip(minibatch, transitions):
            pp.parse_step(transition)

        # 筛选仍未完成解析的对象，移除已经完成的对象
        unfinished_parses = [
            pp for pp in unfinished_parses
            if pp.buffer or len(pp.stack) > 1
        ] #列表推导式

    # 按原始句子顺序，提取每个解析对象最终生成的依存关系列表
    dependencies = [pp.dependencies for pp in partial_parses]



    ### END YOUR CODE

    return dependencies




def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    """
    def __init__(self, mode = "unidirectional"):
        self.mode = mode

    def predict(self, partial_parses):
        if self.mode == "unidirectional":
            return self.unidirectional_predict(partial_parses)
        elif self.mode == "interleave":
            return self.interleave_predict(partial_parses)
        else:
            raise NotImplementedError()

    def unidirectional_predict(self, partial_parses):
        """First shifts everything onto the stack and then does exclusively right arcs if the first word of
        the sentence is "right", "left" if otherwise.
        """
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

    def interleave_predict(self, partial_parses):
        """First shifts everything onto the stack and then interleaves "right" and "left".
        """
        return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """

    # Unidirectional arcs test
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))

    # Out-of-bound test
    sentences = [["right"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))

    # Mixed arcs test
    sentences = [["this", "is", "interleaving", "dependency", "test"]]
    deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'is'), ('dependency', 'interleaving'),
                      ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
