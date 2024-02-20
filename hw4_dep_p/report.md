# 作业技术报告

## `parser_utils.py`分析

### `class Config`

```python
class Config(object):
    ...
    embedding_file = './data/en-cw.txt'
```

在`Config`类我添加了`embedding_file`为word2vec文件路径。

### `class Parser`

```python
class Parser(object):
    ...
    def __init__(self, dataset):
        ...
        # deprel = [self.root_label] + list(set([w for ex in dataset
        #                                        for w in ex['label']
        #                                        if w != self.root_label]))
        deprel = ['root', 'ccomp', 'parataxis', 'acl', 'dep', 'conj', 'cc', 'det:predet', 'mark', 'nmod:poss', 'compound:prt', 'amod', 'nmod:tmod', 'mwe', 'expl', 
'csubjpass', 'punct', 'nummod', 'nsubjpass', 'nmod', 'case', 'dobj','compound', 'advmod', 'appos', 'auxpass', 'neg', 'acl:relcl', 'iobj', 'aux', 'nmod:npmod', 'cc:preconj', 'discourse', 'cop', 'csubj', 'advcl', 'nsubj', 'det', 'xcomp']
```

#### `__init__(self, dataset):`

定义了`root label`,`dependency relation`,`token to id`的映射，这里我为了确保调用`predicte.py`的`Parser`的`tran2id`保持一致，以确保label正确对应，LAS得到正确的数据，固定了`deprel`为常量；构建了`transition`到`id`的映射。包括`:LEFT(L), RIGHT(R), SHIFT(S) `以及 `labeled`版本(`L-deprel, R-deprel)`。

- `root_label`: 根标签，表示句法依存分析中的根节点的标签。
- `unlabeled`: 布尔值，表示是否进行无标签依存分析。
- `with_punct`: 布尔值，表示是否考虑标点符号。
- `use_pos`: 布尔值，表示是否使用词性标签。
- `use_dep`: 布尔值，表示是否使用依存关系标签。
- `n_deprel`: 依存关系标签的数量（不包括无标签情况）。
- `n_trans`: 转移动作的数量。
- `tran2id`: 将转移动作映射到索引的字典。
- `id2tran`: 将索引映射到转移动作的字典。
- `tok2id`: 将词语、词性标签和转移动作映射到索引的字典。
- `id2tok`: 将索引映射到词语、词性标签和转移动作的字典。
- `n_tokens`: 词语、词性标签和转移动作的数量。
- `model`: 存储句法依存分析模型的对象。

#### `vectorize(self, examples):`

将数据转换为向量表示，接受一个列表作为输入，并将每个示例中的词语、词性标签、头部索引和依存关系标签转换为对应的索引值存入列表 `vec_examples`，其中每个示例包含以下字段：

- `word`: 包含词语索引的列表，列表的第一个元素为根标记（ROOT），后续元素是词语在词汇表中的索引，如果词语不在词汇表中，则使用未知标记（UNK）的索引。
- `pos`: 包含词性标签索引的列表，列表的第一个元素为根标记（P_ROOT），后续元素是词性标签在词性标签词汇表中的索引，如果词性标签不在词性标签词汇表中，则使用未知标记（P_UNK）的索引。
- `head`: 包含头部索引的列表，列表的第一个元素为-1，后续元素是对应词语的头部索引。
- `label`: 包含依存关系标签索引的列表，列表的第一个元素为-1，后续元素是依存关系标签在依存关系标签词汇表中的索引，如果依存关系标签不在词性标签词汇表中，则使用-1表示。

#### `extract_features(self, stack, buf, arcs, ex):`

根据当前的堆栈和缓冲区状态，以及示例中的词语、词性标签和依存关系标签，提取一系列特征。特征包括词语本身、词性标签等。每个特征都可以表示为一个字符串，并根据 `self.tok2id` 将字符串转换为相应的整数标识。

首先初始化一个空的特征列表 `features`，并根据堆栈和缓冲区的状态，将相关的词语索引添加到特征列表中。如果使用词性标签（`self.use_pos` 为真），则还会将相关的词性标签索引添加到特征列表中。如果使用依存关系标签（`self.use_dep` 为真），则会将相关的依存关系标签索引添加到特征列表中。

接下来，通过获取左侧孩子（`get_lc`）和右侧孩子（`get_rc`）的索引，以及左侧孩子的左侧孩子（`get_lc(lc[0])`）和右侧孩子的右侧孩子（`get_rc(rc[0])`）的索引，提取更多的特征。这些特征包括左侧孩子的词语、右侧孩子的词语、左侧孩子的第二个词语、右侧孩子的第二个词语、左侧孩子的左侧孩子的词语和右侧孩子的右侧孩子的词语。

最后，将词语特征、词性标签特征和依存关系标签特征连接在一起，并返回特征列表 `features`。

#### `get_oracle(self, stack, buf, ex):`

`get_oracle` 根据当前状态确定应该执行的操作编号（对应于`__init__`中的`trans`列表）。它接受当前的堆栈（stack）、缓冲区（buf）和示例（ex）作为输入，并根据当前状态返回应该执行的操作编号。

首先检查堆栈的大小。如果堆栈的大小小于 2，表示无法执行移动或弧操作，此时返回最后一个操作编号，对应于转换列表中的空操作。

接下来，根据堆栈中最后两个词语的索引（`i0` 和 `i1`），以及它们的头部索引（`h0` 和 `h1`）和依存关系标签（`l0` 和 `l1`），确定应该执行的操作编号。

如果`self.unlabeled` 为真，则执行以下判断：

- 如果 `i1` 大于 0 且 `h1` 等于 `i0`，表示应该执行左弧操作，返回操作编号 0。
- 如果 `i1` 大于等于 0 且 `h0` 等于 `i1`，且缓冲区中没有任何词语的头部是 `i0`，表示应该执行右弧操作，返回操作编号 1。
- 否则，表示应该执行移动操作，返回操作编号 2。如果缓冲区为空，返回 None。

如果`self.unlabeled` 为假，则执行以下判断：

- 如果 `i1` 大于 0 且 `h1` 等于 `i0`，表示应该执行带有标签 `l1` 的左弧操作，返回操作编号 `l1`。如果 `l1` 的取值范围在 0 到 `self.n_deprel` 之间，返回相应的操作编号；否则返回 None。
- 如果 `i1` 大于等于 0 且 `h0` 等于 `i1`，且缓冲区中没有任何词语的头部是 `i0`，表示应该执行带有标签 `l0` 的右弧操作，返回操作编号 `l0 + self.n_deprel`。如果 `l0` 的取值范围在 0 到 `self.n_deprel` 之间，返回相应的操作编号；否则返回 None。
- 否则，表示应该执行移动操作，返回操作编号 `self.n_trans - 1`。如果缓冲区为空，返回 None。

#### `create_instances(self, examples):`

从列表中创建训练实例。它接受一个示例列表作为输入，并返回一个包含所有实例的列表。

在方法内部，对于每个示例，首先获取示例中词语的数量（`n_words`）。然后初始化堆栈（stack）、缓冲区（buf）和弧（arcs）为空，实例列表（instances）为空。

接下来，通过迭代执行转换操作来创建实例。每次迭代中，调用 `get_oracle` 方法根据当前的堆栈和缓冲区状态确定应该执行的操作编号（gold_t）。如果返回的 `gold_t` 为 None，则终止迭代。否则，调用 `legal_labels` 方法获取合法的标签列表（legal_labels）。

然后，调用 `extract_features` 方法提取特征，将特征、合法标签列表和 `gold_t` 组成一个元组，并添加到实例列表中。

接下来，根据 `gold_t` 执行相应的转换操作，更新堆栈、弧和缓冲区。如果 `gold_t` 等于转换的最后一个操作编号（`self.n_trans - 1`），表示应该执行移动操作，将缓冲区的第一个词语移动到堆栈顶部，并从缓冲区中移除。如果 `gold_t` 小于 `self.n_deprel`，表示应该执行左弧操作，将堆栈的最后两个词语之间建立一个弧，并从堆栈中移除最后两个词语。否则，表示应该执行右弧操作，将堆栈倒数第二个词语作为头部，最后一个词语作为依赖，并从堆栈中移除最后一个词语。

增加成功的示例数量，并将列表添加到总实例列表中。

#### ` legal_labels(self, stack, buf):`

用于确定当前状态下的合法标签列表。它接受堆栈（stack）和缓冲区（buf）的状态作为输入，并返回一个表示合法标签的列表。

首先创建长度为 `self.n_deprel` 的列表，用于表示左弧操作的合法标签。如果堆栈的长度大于2，则将左弧操作的合法标签设置为1，否则设置为0。

然后，创建长 `self.n_deprel` 的列表，用于表示右弧操作的合法标签。如果堆栈的长度大于等于2，则将右弧操作的合法标签设置为1，否则设置为0。

最后，创建长度为1的列表，用于表示移动操作的合法标签。如果缓冲区的长度大于0，则将移动操作的合法标签设置为1，否则设置为0。

将左弧操作的合法标签列表、右弧操作的合法标签列表和移动操作的合法标签列表连接起来，形成最终的合法标签列表.

#### ` parse(self, dataset, eval_batch_size):` 

首先，根据数据集中每个例子的单词数量，构建句子列表 `sentences`，其中每个句子是一个包含单词索引的列表。同时，使用句子的内存地址作为键，将句子索引映射到句子的索引位置，以便在后续处理中能够通过句子的地址快速找到对应的索引。

然后，创建 `ModelWrapper` 对象，该对象封装了当前 `Parser` 对象、数据集和句子索引的映射。

接下来，使用 `minibatch_parse` 方法对句子列表进行批量解析，传入模型对象和评估批量大小作为参数。该方法将使用模型进行句法依存解析，并返回解析的结果。

在解析过程中，计算未标记附加准确率（UAS）。首先，通过遍历数据集中的每个例子，将解析结果的依存关系应用到 `head` 列表中，其中 `head` 列表的索引表示单词的位置，值表示该单词的头部位置。然后，将预测的头部位置、标注的头部位置、标注的依存标签和词性进行比较，如果预测的头部位置和标注的头部位置相同，则 UAS 值加1，同时统计所有标记的单词数量，并将其除以所有标记的单词数量，得到最终的 UAS 值。

### `class ModelWrapper(object):`

```python
class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        ...

    def predict(self, partial_parses):
        ...
        if self.parser.unlabeled:
            pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred]
        else:
            pred = [self.parser.id2tran[p] for p in pred]
        return pred
```

`ModelWrapper` 类用于封装 `Parser` 对象、数据集和句子索引的映射。

在初始化时，需要传入 `parser`（`Parser` 对象）、`dataset`（数据集）和 `sentence_id_to_idx`（句子索引映射）作为参数。

`ModelWrapper` 类提供了 `predict` 方法，用于对给定的部分解析进行预测。该方法接受一个部分解析列表 `partial_parses` 作为输入。对于每个部分解析，它提取特征并将其转换为模型可接受的格式。然后，通过模型进行预测，得到预测结果。如果模型使用了无标签的转移系统，则根据预测结果将其转换为对应的操作字符串（"S"、"LA"、"RA"）。否则，将预测结果转换为对应的转移操作字符串。（这一部分我进行了如上的修改）

### `read_conll(in_file, lowercase=False, max_example=None):`

用于从 CoNLL 格式的文件中读取数据并解析为列表。

函数的输入参数包括：

- `in_file`：输入文件的路径。
- `lowercase`：一个布尔值，表示是否将单词转换为小写（默认为 False）。
- `max_example`：一个整数，表示最大读取的示例数量（默认为 None，即读取全部示例）。

函数的输出是一个示例列表，每个示例由单词（word）、词性（pos）、依存头（head）和依存关系标签（label）组成。示例列表中的每个示例是一个字典，包含这些属性。

在函数内部，它打开输入文件，并逐行读取文件内容。对于每一行，它按制表符分割，并根据 CoNLL 格式的列位置提取单词、词性、依存头和依存关系标签。如果行的长度为 10，则说明它是有效的数据行，不包含连字符的行号。此时，将提取的内容添加到当前示例的相应列表中。

如果行的长度不为 10，说明该行是空行或注释行，或者当前示例已经完整读取。在这种情况下，将当前示例添加到示例列表中，并清空临时列表。如果设置了 `max_example`，并且示例数量达到了 `max_example` 的限制，则停止读取，提前返回结果。

### `build_dict(keys, n_max=None, offset=0):`

用于构建词典，将输入的键列表转换为以键为键和以索引为值的字典。

函数的输入参数包括：

- `keys`：键的列表。
- `n_max`：一个整数，表示词典中最常见的键的数量（默认为 None，表示包含所有键）。
- `offset`：一个整数，表示索引的偏移量（默认为 0）。

函数内部首先使用 `Counter` 对键列表进行计数，统计每个键出现的次数。然后，根据 `n_max` 参数选择保留最常见的键及其对应的次数，生成一个元组列表 `ls`。如果 `n_max` 为 None，则保留所有键。

之后，函数使用列表推导式遍历 `ls` 中的元组，根据键和索引的偏移量生成一个字典，其中键为键列表中的键，值为对应的索引加上偏移量。

#### ` load_and_preprocess_data(reduced):`

函数加载训练集、开发集和测试集的数据。通过调用 `read_conll` 函数，读取相应的文件并将其转换为例子列表。训练集、开发集和测试集分别存储在 `train_set`、`dev_set` 和 `test_set` 中。如果 `reduced` 参数为 `True`，则对数据集进行缩减，仅保留部分样本。

接下来，函数创建一个 `Parser` 对象，用于构建解析器。通过传入训练集数据 `train_set` 来初始化解析器。

然后，函数加载预训练的词嵌入。这一部分是我添加的。它读取嵌入文件的每一行，提取单词和对应的嵌入向量，并将其存储在 `word_vectors` 字典中。接着，根据解析器的词汇表，构建一个表示嵌入矩阵的 `embeddings_matrix`。对于词汇表中的每个单词，如果其在 `word_vectors` 中存在对应的嵌入向量，则将该嵌入向量复制到 `embeddings_matrix` 中相应位置；否则，如果其小写形式存在于 `word_vectors` 中，则将其对应的嵌入向量复制到 `embeddings_matrix` 中相应位置。我使用的额外资源在`./data/en_cw.txt`。这部分的代码是：

```python
    print("Loading pretrained embeddings...",)
    word_vectors = {}
    for line in open(config.embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
```



接着，函数对数据集进行向量化处理。通过调用解析器的 `vectorize` 方法，将训练集、开发集和测试集的例子列表转换为向量表示的形式。

最后，函数调用解析器的 `create_instances` 方法，基于训练集数据创建训练实例。

函数返回解析器对象 `parser`、嵌入矩阵 `embeddings_matrix`、训练实例列表 `train_examples`、向量化后的开发集 `dev_set` 和测试集 `test_set`。

这些是我添加的函数：

```python
def minibatches(data, batch_size, size):
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    one_hot = np.zeros((y.size, size))
    one_hot[np.arange(y.size), y] = 1
    return get_minibatches([x, one_hot], batch_size)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def get_minibatches(data, minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [_minibatch(d, minibatch_indices) for d in data] if list_data \
            else _minibatch(data, minibatch_indices)


def _minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]
```



`minibatches(data, batch_size, size)` 函数用于将数据集划分为小批量数据。它接收 `data`（包含特征和标签）, `batch_size` 和 `size` 参数。函数首先将特征和标签分别提取出来，然后将标签进行`one hot`编码转换。最后，它调用 `get_minibatches` 函数来获取实际的小批量数据。

`AverageMeter` 类用于计算和存储平均值和当前值。它具有 `reset` 方法用于重置计数器，`update` 方法用于更新计数器和累积值，以及 `avg` 属性用于获取平均值。

`get_minibatches(data, minibatch_size, shuffle=True)` 函数用于生成小批量数据。它接收 `data`（可能是一个列表，其中包含特征和标签的列表或 NumPy 数组）作为输入，以及 `minibatch_size` 和 `shuffle` 参数。函数首先根据数据的大小生成索引，然后根据 `minibatch_size` 将索引划分为小批量。通过迭代这些小批量索引，函数返回相应的小批量数据。

`_minibatch(data, minibatch_idx)` 函数用于根据给定的小批量索引从数据中提取相应的数据。如果 `data` 是一个 NumPy 数组，它将根据索引返回相应的子数组。如果 `data` 是一个列表，它将根据索引返回子列表。

这些辅助函数和类用于支持数据的划分和批处理操作，便于训练过程中对数据进行小批量处理，提高训练效率和模型性能。

### 实现代码的解释

`parser_utils.py`在第一部分已做解释，不再赘述。

#### `parser_transitions.py`

```py
class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.stack = ['ROOT']
        self.buffer = [] + sentence
        self.dependencies = []

    def parse_step(self, transition):
        if transition.startswith("L"):
            label = transition[2:] if len(transition) > 2 else None
            self.dependencies.append((self.stack[-1], self.stack[-2], label))
            self.stack.pop(-2)
        elif transition.startswith("R"):
            label = transition[2:] if len(transition) > 2 else None
            self.dependencies.append((self.stack[-2], self.stack[-1], label))
            self.stack.pop(-1)
        elif transition == "S":
            self.stack.append(self.buffer.pop(0))

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies

def minibatch_parse(sentences, model, batch_size):
    dependencies = []

    partial_parses = [PartialParse(sentence) for sentence in sentences]

    unfinished_parses = partial_parses[:]
    while unfinished_parses:
        batch_parses = unfinished_parses[:batch_size]
        transitions = model.predict(batch_parses)
        for parse,transition in zip(batch_parses,transitions):
            parse.parse_step(transition)
            if len(parse.stack) == 1 and len(parse.buffer) == 0:
                unfinished_parses.remove(parse)
    dependencies = [parse.dependencies for parse in partial_parses]
    return dependencies
```



`PartialParse` 类表示部分解析的状态。在初始化时，它接收一个句子作为输入，并设置初始的堆栈（`stack`）为只包含根节点（`ROOT`）的列表，缓冲区（`buffer`）为句子的副本，依存关系列表（`dependencies`）为空。

`parse_step(transition)` 方法接收一个转移（`transition`）作为输入，并根据转移的类型执行相应的操作。同时考虑带标签和不带标签的情况，如果转移以 "L" 开头，表示进行左弧操作，将栈顶两个元素的依存关系添加到依存关系列表中，并将栈中第二个元素弹出。如果转移以 "R" 开头，表示进行右弧操作，将栈顶两个元素的依存关系添加到依存关系列表中，并将栈顶元素弹出。如果转移是 "S"，表示进行移进操作，将缓冲区的第一个元素移动到栈顶。通过多次调用 `parse_step` 方法，可以逐步解析句子。

`minibatch_parse(sentences, model, batch_size)` 函数接收句子列表（`sentences`）、模型对象（`model`）和批量大小（`batch_size`）作为输入。它通过创建部分解析对象的列表，并将其存储在 `partial_parses` 中。然后，它使用循环来不断执行解析步骤，直到所有部分解析都完成。在每个步骤中，它从模型中预测出转移序列，并逐个应用于相应的部分解析对象。如果某个部分解析对象完成解析（即堆栈中只剩下根节点且缓冲区为空），则将其从未完成解析列表中移除。最终，它返回所有部分解析对象生成的依存关系列表。

### `trainer.py`

```python
    def train(self, parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005): 
        ...
        ### TODO: Initialize `self.optimizer`, i.e., specify parameters to optimize
        self.optimizer = optim.Adam(params=parser.model.parameters(),lr=self.lr)
        self.loss_func = loss_func = nn.CrossEntropyLoss()

        for epoch in range(self.n_epochs):
            ...
            dev_UAS, dev_LAS = self._train_for_epoch(parser, train_data, dev_data, self.optimizer, self.loss_func, self.batch_size)
            # TODO: you can change this part, to use either uas or las to select best model
            if dev_LAS > best_dev_LAS:
                best_dev_LAS = dev_LAS
                print("New best dev LAS! Saving model.")
                torch.save(parser.model.state_dict(), self.output_path)
            print("")
```

这部分没有太多的修改：

1. 增加了参数。用于接受data，给出输出路径，调整训练`batch_size`,`epoch`和学习率。
2. 初始化了`self.optimizer`和`self.loss_func`.
3. 根据`_train_for_epoch`函数参数的传入参数。
4. 调整根据`dev_LAS`选择最优模型。

```python
 def _train_for_epoch(self, parser, train_data, dev_data, optimizer, loss_func, batch_size): 
        parser.model.train() # Places model in "train" mode, e.g., apply dropout layer, etc.
        ### TODO: Train all batches of train_data in an epoch.
        ### Remember to shuffle before training the first batch (You can use Dataloader of PyTorch)

        n_minibatches = math.ceil(len(train_data) / batch_size)
        loss_meter = AverageMeter()
        with tqdm(total=n_minibatches) as prog:
            size = 3 if parser.unlabeled else 79
            for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size,size)):
                optimizer.zero_grad()
                loss = 0.
                train_x = torch.from_numpy(train_x).long()
                train_y = torch.from_numpy(train_y.nonzero()[1]).long()
                logits = parser.model.forward(train_x)
                loss = loss_func(logits,train_y)
                loss.backward()
                optimizer.step()
                prog.update(1)
                loss_meter.update(loss.item())
        print ("Average Train Loss: {}".format(loss_meter.avg))
        print("Evaluating on dev set",)
        parser.model.eval() # Places model in "eval" mode, e.g., don't apply dropout layer, etc.
        _,dependencies = parser.parse(self.dev_data)
        all_head = []
        all_ex_head = []
        for i, ex in enumerate(self.dev_data):
            head = [-1] * len(ex['word'])
            for dependency in dependencies[i]:
                h, t, label = dependency
                head[t] = [h, label]
            ex_label = [parser.id2tok[w].replace('<l>:', '') for w in ex['label'][1:]]
            all_head.append(head[1:]) 
            all_ex_head.append(list(zip(ex['head'][1:],ex_label)))
        uas,las = evaluate(all_head, all_ex_head)  # To check the format of the input, please refer to the utils.py
        print("- dev UAS: {:.2f}".format(uas * 100.0), "- dev LAS: {:.2f}".format(las * 100.0))
        return uas, las
```

该函数是进行单个`epoch`的训练，具体执行以下步骤：

1. 遍历训练数据的小批次：
   - 重置优化器梯度。
   - 计算模型的输出。
   - 计算损失。
   - 反向传播并更新模型参数。
   - 更新进度条和损失计量器。
2. 解析开发集数据，获取依赖关系。与模型预测值比较，计算开发集数据的 UAS 和 LAS。

### `parsing_model.py`

```python
class ParsingModel(nn.Module):

    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        super(ParsingModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))
        self.embed_to_hidden_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_features*self.embed_size,self.hidden_size)))
        self.embed_to_hidden_bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1,self.hidden_size)))

        self.dropout = nn.Dropout(self.dropout_prob)

        self.hidden_to_logits_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.hidden_size,self.n_classes)))
        self.hidden_to_logits_bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1,self.n_classes)))




    def forward(self, t):
        x = self.embeddings[t].view(t.shape[0],-1)
        h = F.relu(x.matmul(self.embed_to_hidden_weight)+self.embed_to_hidden_bias)
        h = self.dropout(h)
        logits = h.matmul(self.hidden_to_logits_weight)+self.hidden_to_logits_bias
        
        return logits

```

**输入参数:**

- `embeddings`: 预训练的词嵌入矩阵。
- `n_features`: 特征维度。
- `hidden_size` : 隐藏层大小。
- `n_classes` : 类别数量。
- `dropout_prob`: Dropout 概率。

该类定义了神经依存解析器的前向传播过程。

- `__init__` 方法用于初始化模型的参数和层。
- `forward` 方法用于执行模型的前向传播过程，执行步骤如下：
  1. 从预训练的词嵌入矩阵 `embeddings` 中获取输入张量 `t` 对应的嵌入向量。嵌入向量的形状为 `(batch_size, n_features, embed_size)`，其中 `embed_size` 是词嵌入的维度。
  2. 将嵌入向量展平为形状为 `(batch_size, n_features * embed_size)` 的张量 `x`。
  3. 将 `x` 乘以权重矩阵 `embed_to_hidden_weight`，并加上偏置向量 `embed_to_hidden_bias`。然后应用 ReLU 激活函数，得到隐藏层输出张量 `h`。隐藏层输出的形状为 `(batch_size, hidden_size)`。
  4. 对隐藏层输出 `h` 进行 Dropout 操作，根据指定的概率随机丢弃部分神经元，以防止过拟合。
  5. 将 Dropout 后的隐藏层输出 `h` 乘以权重矩阵 `hidden_to_logits_weight`，并加上偏置向量 `hidden_to_logits_bias`，得到预测张量 `logits`。预测张量的形状为 `(batch_size, n_classes)`。

### `main.py`

```python
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)
    features = 36 if parser.unlabeled else 48
    classes = 3 if parser.unlabeled else 79
    if parser.unlabeled and debug:
        classes = 77
    parser.model = ParsingModel(embeddings,n_features=features,n_classes=classes)
    ...
    optimizer = optim.Adam(params=parser.model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    ...
    
```

这一部分是根据解析器的配置，确定特征数量 `features` 和类别数量 `classes`。之后创建优化器 `optimizer` 和损失函数 `loss_func`。

```python
if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        _, dependencies = parser.parse(test_data)
        all_head = []
        all_ex_head = []
        for i, ex in enumerate(test_data):
            head = [-1] * len(ex['word'])
            for dependency in dependencies[i]:
                h, t, label = dependency
                head[t] = [h, label]
            ex_label = [parser.id2tok[w].replace('<l>:', '') for w in ex['label'][1:]]
            all_head.append(head[1:]) 
            all_ex_head.append(list(zip(ex['head'][1:],ex_label)))
        with open('./prediction.json', 'w') as fh:
            json.dump(dependencies, fh)
        uas,las = evaluate(all_head, all_ex_head)  # To check the format of the input, please refer to the utils.py
        print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))
        print("Done!")
```

这里对`test_data`评测之前进行了解析数据，获取依赖关系。

### `predicte.py`

这段代码主要用于加载训练好的依存句法解析模型，并在测试集上进行评估和打印解析结果。下面是代码的主要流程：

1. 设置随机种子：使用相同的随机种子确保实验的可重复性和一致性。
2. 加载数据：使用`read_conll`函数从文件中读取训练集和测试集的数据。
3. 创建解析器对象：使用训练集数据创建一个`Parser`对象，该对象将用于解析测试集数据。
4. 加载模型权重：创建一个`ParsingModel`对象，并加载预训练的模型权重。
5. 执行测试集解析：对测试集数据进行依存句法解析，并获取解析结果。
6. 构造解析结果：根据解析结果和测试集数据，构造完整的依存关系和标签列表。
7. 使用`evaluate`函数对解析结果进行评估，计算UAS和LAS指标。

## 实验结果

- test UAS: 89.13 - test las: 87.17
- dev UAS: 88.81 - dev LAS: 86.74

## Parsing展示

Sentence: <ROOT> no , it was n't black monday .

| Word   | Predicted Dependency | Actual Dependency |
| ------ | -------------------- | ----------------- |
| no     | (7, 'discourse')     | (7, 'discourse')  |
| ,      | (7, 'punct')         | (7, 'punct')      |
| it     | (7, 'nsubj')         | (7, 'nsubj')      |
| was    | (7, 'cop')           | (7, 'cop')        |
| n't    | (7, 'neg')           | (7, 'neg')        |
| black  | (7, 'amod')          | (7, 'compound')   |
| monday | (0, 'root')          | (0, 'root')       |
| .      | (7, 'punct')         | (7, 'punct')      |

Sentence: <ROOT> but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged 190.58 points -- most of it in the final hour -- it barely managed to stay this side of chaos .

| Word       | Predicted Dependency | Actual Dependency |
| ---------- | -------------------- | ----------------- |
| but        | (0, 'root')          | (33, 'cc')        |
| while      | (10, 'mark')         | (10, 'mark')      |
| the        | (7, 'det')           | (7, 'det')        |
| new        | (7, 'compound')      | (7, 'compound')   |
| york       | (7, 'compound')      | (7, 'compound')   |
| stock      | (7, 'compound')      | (7, 'compound')   |
| exchange   | (10, 'nsubj')        | (10, 'nsubj')     |
| did        | (10, 'aux')          | (10, 'aux')       |
| n't        | (10, 'neg')          | (10, 'neg')       |
| fall       | (33, 'advcl')        | (33, 'advcl')     |
| apart      | (10, 'advcl')        | (10, 'advcl')     |
| friday     | (10, 'nmod:tmod')    | (10, 'nmod:tmod') |
| as         | (19, 'mark')         | (19, 'mark')      |
| the        | (18, 'det')          | (18, 'det')       |
| dow        | (18, 'compound')     | (18, 'compound')  |
| jones      | (18, 'compound')     | (18, 'compound')  |
| industrial | (18, 'compound')     | (18, 'compound')  |
| average    | (19, 'nsubj')        | (19, 'nsubj')     |
| plunged    | (10, 'advcl')        | (10, 'advcl')     |
| 190.58     | (21, 'nummod')       | (21, 'nummod')    |
| points     | (19, 'dobj')         | (19, 'dobj')      |
| --         | (23, 'punct')        | (23, 'punct')     |
| most       | (19, 'dep')          | (19, 'dep')       |
| of         | (25, 'case')         | (25, 'case')      |
| it         | (23, 'nmod')         | (23, 'nmod')      |
| in         | (29, 'case')         | (29, 'case')      |
| the        | (29, 'det')          | (29, 'det')       |
| final      | (29, 'amod')         | (29, 'amod')      |
| hour       | (23, 'nmod')         | (23, 'nmod')      |
| --         | (33, 'punct')        | (33, 'punct')     |
| it's       | (33, 'nsubj')        | (33, 'nsubj')     |
| barely     | (33, 'advmod')       | (33, 'advmod')    |
| managed    | (23, 'dep')          | (0, 'root')       |
| to         | (35, 'mark')         | (35, 'mark')      |
| stay       | (33, 'xcomp')        | (33, 'xcomp')     |
| this       | (37, 'det')          | (37, 'det')       |
| side       | (35, 'xcomp')        | (35, 'dobj')      |
| of         | (39, 'case')         | (39, 'case')      |
| chaos      | (37, 'nmod')         | (37, 'nmod')      |
| .          | (1, 'punct')         | (33, 'punct')     |
